"""
MoE modules for EDELINE-MARL.

- SoftMoEVelocityBias: SoftMoE that outputs task-conditioned velocity bias
  for injection into the flow predictor.
- SparseMoEReward: SparseMoE reward predictor conditioned on dual latent state.
- ContinuePredictor: Binary continue-flag predictor.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from m3w.interfaces import LATENT_DIM
from m3w.models.world_models import NoisyTopKRouter, SelfAttnExpert


# ---------------------------------------------------------------------------
# D1 -- SoftMoE Velocity Bias
# ---------------------------------------------------------------------------

class SoftMoEVelocityBias(nn.Module):
    """Produces a task-conditioned velocity bias via SoftMoE dispatch/combine.

    Follows the SoftMoE pattern from ``CenMoEDynamicsModel`` but operates on a
    single-token-per-agent recurrent state and outputs a bias vector in latent
    space rather than a next-latent prediction.

    Args:
        hidden_dim: Dimensionality of recurrent state h_t^i.
        latent_dim: Dimensionality of the output velocity bias.
        num_experts: Number of SoftMoE experts.
        num_slots: Number of slots per expert (kept at 1 for single-token input).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        latent_dim: int = LATENT_DIM,
        num_experts: int = 16,
        num_slots: int = 1,
        expert_hidden_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.num_slots = num_slots

        # Expert MLPs: hidden_dim -> expert_hidden_dim -> latent_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, latent_dim),
            )
            for _ in range(num_experts)
        ])

        # Dispatch/combine weight matrix (learned phi), mirroring CenMoEDynamicsModel
        self.phi = nn.Parameter(
            torch.randn(hidden_dim, num_experts, num_slots)
            * (1.0 / math.sqrt(hidden_dim))
        )

    def compute_bias(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the velocity bias for a batch of agent recurrent states.

        Args:
            h: Recurrent hidden state, shape ``(B, hidden_dim)``.

        Returns:
            Velocity bias, shape ``(B, latent_dim)``.
        """
        # Reshape to (B, 1, hidden_dim) -- single "token" per agent
        x = h.unsqueeze(1)  # [B, 1, hidden_dim]

        # Router weights: [B, 1, num_experts, num_slots]
        weights = torch.einsum("b n d, d e s -> b n e s", x, self.phi)

        # Dispatch: softmax over token dim (dim=1)
        dispatch_weights = F.softmax(weights, dim=1)  # [B, 1, E, S]
        expert_inputs = torch.einsum(
            "b n e s, b n d -> b e s d", dispatch_weights, x
        )  # [B, E, S, hidden_dim]

        # Expert forward
        expert_outputs = torch.stack([
            self.experts[i](expert_inputs[:, i])
            for i in range(self.num_experts)
        ])  # [E, B, S, latent_dim]
        expert_outputs = einops.rearrange(
            expert_outputs, "e b s d -> b (e s) d"
        )  # [B, E*S, latent_dim]

        # Combine: softmax over expert*slot dim
        combine_weights = einops.rearrange(
            weights, "b n e s -> b n (e s)"
        )  # [B, 1, E*S]
        combine_weights = F.softmax(combine_weights, dim=-1)

        out = torch.einsum(
            "b n z, b z d -> b n d", combine_weights, expert_outputs
        )  # [B, 1, latent_dim]

        return out.squeeze(1)  # [B, latent_dim]


# ---------------------------------------------------------------------------
# D2 -- SparseMoE Reward (dual-latent conditioned)
# ---------------------------------------------------------------------------

class SparseMoEReward(nn.Module):
    """SparseMoE reward predictor conditioned on the full dual-latent state.

    Input ``s_t`` is the concatenation over all agents of
    ``[z_flat^i; d^i; h^i]``, giving per-agent dim = LATENT_DIM + hidden_dim.

    The router gates on the flattened input while the self-attention experts
    operate on the per-agent token view ``(B, n_agents, per_agent_dim)``.

    Args:
        n_agents: Number of agents.
        hidden_dim: Dimensionality of recurrent state h per agent.
        num_experts: Number of experts.
        k: Top-K for the NoisyTopKRouter.
        num_reward_bins: Number of discrete reward bins for two-hot encoding.
        n_heads: Number of attention heads in each self-attention expert.
        expert_ffn_hidden: FFN hidden size inside each expert.
        expert_dropout: Dropout rate inside experts.
        head_hidden: Hidden size for the reward MLP head.
        noisy_gating: Whether to use noisy gating in the router.
    """

    def __init__(
        self,
        n_agents: int,
        hidden_dim: int = 512,
        num_experts: int = 16,
        k: int = 2,
        num_reward_bins: int = 101,
        n_heads: int = 1,
        expert_ffn_hidden: int = 1024,
        expert_dropout: float = 0.0,
        head_hidden: int = 512,
        noisy_gating: bool = True,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()
        self.per_agent_dim = (latent_dim if latent_dim is not None else LATENT_DIM) + hidden_dim
        self.n_agents = n_agents
        self.num_experts = num_experts
        self.k = k
        self.num_reward_bins = num_reward_bins

        # Router gates on flattened s_t
        self.router = NoisyTopKRouter(
            in_dim=n_agents * self.per_agent_dim,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
        )

        # Self-attention experts operating on (B, n_agents, per_agent_dim)
        self.experts = nn.ModuleList([
            SelfAttnExpert(
                d_model=self.per_agent_dim,
                n_heads=n_heads,
                ffn_hidden=expert_ffn_hidden,
                dropout=expert_dropout,
            )
            for _ in range(num_experts)
        ])

        # Reward head: flattened expert output -> reward bins
        self.reward_head = nn.Sequential(
            nn.Linear(n_agents * self.per_agent_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_reward_bins),
        )

    def forward(self, s_t: torch.Tensor):
        """Forward pass.

        Args:
            s_t: Concatenated dual-latent state for all agents,
                 shape ``(B, n_agents * (LATENT_DIM + hidden_dim))``.

        Returns:
            r_logits: Reward bin logits, shape ``(B, num_reward_bins)``.
            aux: Dictionary with ``loss_balancing``, ``gates``, ``logits``
                 from the router.
        """
        B = s_t.shape[0]

        # Reshape to per-agent tokens
        x = s_t.view(B, self.n_agents, self.per_agent_dim)  # [B, N_a, D]

        x_flat = s_t  # [B, N_a * D]
        gates, _load, _logits, aux_router = self.router(x_flat)  # gates: [B, N_e]

        # Weighted expert aggregation (same pattern as CenMoERewardModel)
        y = torch.zeros_like(x)  # [B, N_a, D]
        for e_idx, expert in enumerate(self.experts):
            mask = gates[:, e_idx] > 0
            if not mask.any():
                continue
            x_sel = x[mask]                          # [b_i, N_a, D]
            out_e = expert(x_sel)                    # [b_i, N_a, D]
            w = gates[mask, e_idx].view(-1, 1, 1)
            y[mask] = y[mask] + w * out_e

        y_flat = y.reshape(B, -1)                    # [B, N_a * D]
        r_logits = self.reward_head(y_flat)          # [B, num_reward_bins]

        aux = dict(
            loss_balancing=aux_router["loss_balancing"],
            gates=gates,
            logits=aux_router["logits"],
        )
        return r_logits, aux


# ---------------------------------------------------------------------------
# D3 -- Continue Predictor
# ---------------------------------------------------------------------------

class ContinuePredictor(nn.Module):
    """Binary continue-flag predictor (2-layer MLP).

    Args:
        input_dim: Dimensionality of the input (same s_t concatenation as
                   ``SparseMoEReward``).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            s_t: Concatenated dual-latent state, shape ``(B, input_dim)``.

        Returns:
            Continue logit, shape ``(B, 1)``.
        """
        return self.mlp(s_t)


__all__ = [
    "SoftMoEVelocityBias",
    "SparseMoEReward",
    "ContinuePredictor",
]
