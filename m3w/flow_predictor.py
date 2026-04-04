"""
Flow Matching Latent Predictor for EDELINE-MARL.

Implements continuous flow matching with shortcut forcing for predicting
dual latent states [z; d] in the multi-agent world model.

References:
    - Lipman et al. 2022 — Flow Matching for Generative Modeling
    - Frans et al. 2024 — Shortcut Forcing for one-step generation
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from m3w.interfaces import FlowPredictorOutput, LATENT_DIM


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(tau: torch.Tensor, embed_dim: int = 64) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar time tau.

    Args:
        tau: (B, 1) time values in [0, 1].
        embed_dim: Dimensionality of the embedding.

    Returns:
        Embedding tensor of shape (B, embed_dim).
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    half = embed_dim // 2
    # Frequency bands: exp(-log(10000) * i / (half - 1)) for i in [0, half)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=tau.dtype) / max(half - 1, 1)
    )  # (half,)
    args = tau * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)


# ---------------------------------------------------------------------------
# B1 — VelocityField
# ---------------------------------------------------------------------------

class VelocityField(nn.Module):
    """Neural network v_theta(x_tau, tau, h, m) predicting the velocity in the
    flow ODE.  3-hidden-layer MLP with Mish activation and LayerNorm.

    Conditioning is done by concatenating [x_tau; tau_embed; h; m].
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 512,
        cond_dim: int = 512,
        tau_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.TAU_EMBED_DIM = tau_embed_dim

        input_dim = latent_dim + tau_embed_dim + cond_dim + cond_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # MoE bias buffer — defaults to zero, can be injected externally.
        self.register_buffer("_moe_bias", torch.zeros(latent_dim))

    # -- MoE bias hooks ------------------------------------------------

    def set_moe_bias(self, bias: torch.Tensor) -> None:
        """Inject an MoE velocity bias.  The bias participates in the
        computational graph (no detach).
        """
        self._moe_bias = bias

    def clear_moe_bias(self) -> None:
        """Reset MoE bias to zero (removes from computational graph)."""
        self._moe_bias = torch.zeros(
            self.latent_dim, device=next(self.parameters()).device
        )

    # -- Forward -------------------------------------------------------

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        h: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_tau: (B, latent_dim) noisy interpolant.
            tau:   (B, 1) time scalar in [0, 1].
            h:     (B, cond_dim) recurrent hidden state.
            m:     (B, cond_dim) cross-agent message.

        Returns:
            v: (B, latent_dim) predicted velocity (including MoE bias).
        """
        tau_embed = sinusoidal_embedding(tau, self.TAU_EMBED_DIM)  # (B, 64)
        inp = torch.cat([x_tau, tau_embed, h, m], dim=-1)          # (B, input_dim)
        v = self.net(inp)                                          # (B, latent_dim)
        v = v + self._moe_bias                                     # add MoE bias
        return v


# ---------------------------------------------------------------------------
# B2 — FlowMatchingLoss
# ---------------------------------------------------------------------------

class FlowMatchingLoss:
    """Computes flow matching and shortcut forcing losses.

    The linear interpolant is x_tau = (1-tau)*x_0 + tau*x_1 with constant
    target velocity u = x_1 - x_0.
    """

    LAMBDA_SHORTCUT: float = 0.1

    @staticmethod
    def compute(
        v_pred: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        tau: torch.Tensor,
        velocity_field: Optional["VelocityField"] = None,
        h: Optional[torch.Tensor] = None,
        m: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute flow matching loss and shortcut loss.

        Args:
            v_pred:  (B, latent_dim) predicted velocity from the network.
            x_0:     (B, latent_dim) noise sample.
            x_1:     (B, latent_dim) target latent.
            tau:     (B, 1) interpolation time.
            velocity_field: VelocityField module (needed for shortcut loss).
            h:       (B, cond_dim) recurrent state (needed for shortcut loss).
            m:       (B, cond_dim) message (needed for shortcut loss).

        Returns:
            Dict with "flow_loss" and "shortcut_loss" (scalars).
        """
        # Target velocity for the linear interpolant is constant: u = x_1 - x_0
        target_v = x_1 - x_0  # (B, latent_dim)

        # Flow matching loss: MSE(v_pred, target_v)
        flow_loss = F.mse_loss(v_pred, target_v)

        # Shortcut loss: single-step estimate from tau=0
        if velocity_field is not None and h is not None and m is not None:
            tau_zero = torch.zeros(x_0.shape[0], 1, device=x_0.device, dtype=x_0.dtype)
            v_at_zero = velocity_field(x_0, tau_zero, h, m)
            x_hat = x_0 + v_at_zero
            shortcut_loss = FlowMatchingLoss.LAMBDA_SHORTCUT * F.mse_loss(x_hat, x_1)
        else:
            shortcut_loss = torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)

        return {"flow_loss": flow_loss, "shortcut_loss": shortcut_loss}


# ---------------------------------------------------------------------------
# B3 — FlowPredictor
# ---------------------------------------------------------------------------

class FlowPredictor(nn.Module):
    """Top-level flow matching latent predictor.

    Training:  ``forward(x_1_target, h, m)`` — samples noise & time, returns
               prediction and velocity for loss computation.
    Inference: ``sample(h, m, steps)`` — generates a latent via Euler
               integration (or single-step shortcut when steps=1).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 512,
        cond_dim: int = 512,
        tau_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._velocity_field = VelocityField(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            tau_embed_dim=tau_embed_dim,
        )

    # -- Public property for external MoE bias injection ---------------

    @property
    def velocity_field(self) -> VelocityField:
        return self._velocity_field

    # -- Training forward ----------------------------------------------

    def forward(
        self,
        x_1_target: torch.Tensor,
        h: torch.Tensor,
        m: torch.Tensor,
    ) -> FlowPredictorOutput:
        """Training-mode forward pass.

        Args:
            x_1_target: (B, latent_dim) target dual latent [z; d].
            h:          (B, cond_dim) recurrent hidden state.
            m:          (B, cond_dim) cross-agent message.

        Returns:
            FlowPredictorOutput with x_hat and v_field.
        """
        B = x_1_target.shape[0]
        device = x_1_target.device
        dtype = x_1_target.dtype

        # Sample tau ~ U(0, 1)
        tau = torch.rand(B, 1, device=device, dtype=dtype)

        # Sample x_0 ~ N(0, I)
        x_0 = torch.randn(B, self.latent_dim, device=device, dtype=dtype)

        # Linear interpolant
        x_tau = (1.0 - tau) * x_0 + tau * x_1_target  # (B, latent_dim)

        # Predict velocity
        v_pred = self._velocity_field(x_tau, tau, h, m)  # (B, latent_dim)

        # Single-step shortcut estimate
        x_hat = x_0 + v_pred  # (B, latent_dim)

        return FlowPredictorOutput(x_hat=x_hat, v_field=v_pred)

    # -- Inference sampling --------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        steps: int = 1,
    ) -> FlowPredictorOutput:
        """Generate a latent sample via Euler integration (or single-step
        shortcut).

        Args:
            h:     (B, cond_dim) recurrent hidden state.
            m:     (B, cond_dim) cross-agent message.
            steps: Number of Euler integration steps.  1 = shortcut mode.

        Returns:
            FlowPredictorOutput with x_hat and v_field (velocity at final step).
        """
        B = h.shape[0]
        device = h.device
        dtype = h.dtype

        x = torch.randn(B, self.latent_dim, device=device, dtype=dtype)

        if steps == 1:
            # Shortcut mode: single step from tau=0
            tau = torch.zeros(B, 1, device=device, dtype=dtype)
            v = self._velocity_field(x, tau, h, m)
            x_hat = x + v
            return FlowPredictorOutput(x_hat=x_hat, v_field=v)

        # Multi-step Euler integration
        dt = 1.0 / steps
        v = None
        for i in range(steps):
            tau = torch.full((B, 1), i * dt, device=device, dtype=dtype)
            v = self._velocity_field(x, tau, h, m)
            x = x + v * dt

        # v is the velocity at the last step
        return FlowPredictorOutput(x_hat=x, v_field=v)
