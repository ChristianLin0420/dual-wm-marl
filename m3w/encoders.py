"""
Dual encoder branches for EDELINE-MARL.

Implements the visual (CategoricalVAE) and semantic (SemanticEncoder) latent
representations, plus the unified DualEncoder that produces AgentLatent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from m3w.interfaces import AgentLatent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP: Linear -> ReLU -> Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# Subtask A1 -- Categorical VAE
# ---------------------------------------------------------------------------

class CategoricalVAE(nn.Module):
    """Categorical VAE that produces discrete visual latents via Gumbel-Softmax.

    The encoder maps observations to logits over ``num_cats`` independent
    categorical distributions each with ``cat_dim`` classes.  Sampling uses
    straight-through Gumbel-Softmax so gradients flow through the one-hot
    samples.  The decoder reconstructs the observation from the flattened
    one-hot vector.

    Args:
        obs_dim:  Dimensionality of the observation vector.
        num_cats: Number of categorical distributions.
        cat_dim:  Number of classes per categorical distribution.
        hidden_dim: Hidden layer width for encoder and decoder MLPs.
        tau_start:  Initial Gumbel-Softmax temperature.
        tau_end:    Final Gumbel-Softmax temperature after annealing.
        anneal_steps: Number of forward calls over which to anneal temperature.
    """

    def __init__(
        self,
        obs_dim: int,
        num_cats: int = 32,
        cat_dim: int = 32,
        hidden_dim: int = 512,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        anneal_steps: int = 100_000,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_cats = num_cats
        self.cat_dim = cat_dim
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.anneal_steps = anneal_steps

        self.encoder = _build_mlp(obs_dim, hidden_dim, num_cats * cat_dim)
        self.decoder = _build_mlp(num_cats * cat_dim, hidden_dim, obs_dim)

        # Step counter for temperature annealing (not a learned parameter).
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    # -- temperature helpers ------------------------------------------------

    @property
    def tau(self) -> float:
        """Current Gumbel-Softmax temperature (linearly annealed)."""
        frac = min(self._step.item() / max(self.anneal_steps, 1), 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * frac

    def _advance_step(self) -> None:
        if self.training:
            self._step.add_(1)

    # -- forward ------------------------------------------------------------

    def forward(self, obs: torch.Tensor):
        """Encode, sample, and decode.

        Args:
            obs: Observation tensor of shape ``(B, obs_dim)``.

        Returns:
            z:     One-hot samples, shape ``(B, num_cats, cat_dim)``.
            logits: Raw logits,     shape ``(B, num_cats, cat_dim)``.
            recon:  Reconstruction, shape ``(B, obs_dim)``.
        """
        self._advance_step()

        logits = self.encoder(obs).view(-1, self.num_cats, self.cat_dim)

        # Gumbel-Softmax with straight-through gradient estimator.
        if self.training:
            z = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        else:
            # At eval time, take the argmax directly.
            idx = logits.argmax(dim=-1, keepdim=True)
            z = torch.zeros_like(logits).scatter_(-1, idx, 1.0)

        recon = self.decoder(z.reshape(obs.shape[0], -1))
        return z, logits, recon

    # -- loss ---------------------------------------------------------------

    def reconstruction_loss(self, obs: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss (L_rec).

        Args:
            obs:   Ground-truth observations ``(B, obs_dim)``.
            recon: Reconstructed observations ``(B, obs_dim)``.

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(recon, obs)


# ---------------------------------------------------------------------------
# Subtask A2 -- Semantic Encoder + Barlow Twins Loss
# ---------------------------------------------------------------------------

class BarlowTwinsLoss(nn.Module):
    """Barlow Twins redundancy-reduction loss.

    Encourages the cross-correlation matrix of two batches of embeddings to
    approach the identity matrix.

    L = sum_i (1 - C_ii)^2  +  lambda * sum_{i != j} C_ij^2

    Args:
        lam: Off-diagonal penalty weight (default 0.005).
    """

    def __init__(self, lam: float = 0.005):
        super().__init__()
        self.lam = lam

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute the Barlow Twins loss.

        Args:
            z_a: First view embeddings, shape ``(B, D)``.
            z_b: Second view embeddings, shape ``(B, D)``.

        Returns:
            Scalar loss.
        """
        B, D = z_a.shape

        # Normalize along batch dimension to zero mean, unit variance.
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-5)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-5)

        # Cross-correlation matrix C: (D, D)
        c = (z_a_norm.T @ z_b_norm) / B

        # Diagonal (invariance) term.
        diag = torch.diagonal(c)
        loss_diag = ((1.0 - diag) ** 2).sum()

        # Off-diagonal (redundancy reduction) term.
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=c.device)
        loss_off = (c[off_diag_mask] ** 2).sum()

        return loss_diag + self.lam * loss_off


class SemanticEncoder(nn.Module):
    """MLP projector that produces L2-normalised semantic embeddings.

    During training, two augmented views are created via independent dropout
    masks and the Barlow Twins loss drives the representation.

    Args:
        obs_dim:    Dimensionality of the observation vector.
        sem_dim:    Dimensionality of the semantic embedding.
        hidden_dim: Hidden layer width.
        aug_drop_p: Dropout probability used for view augmentation.
        bt_lambda:  Off-diagonal weight for BarlowTwinsLoss.
    """

    def __init__(
        self,
        obs_dim: int,
        sem_dim: int = 512,
        hidden_dim: int = 512,
        aug_drop_p: float = 0.1,
        bt_lambda: float = 0.005,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.sem_dim = sem_dim

        self.projector = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sem_dim),
        )
        self.aug_dropout = nn.Dropout(p=aug_drop_p)
        self.bt_loss_fn = BarlowTwinsLoss(lam=bt_lambda)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Project observation to an L2-normalised semantic embedding.

        Args:
            obs: Observation tensor of shape ``(B, obs_dim)``.

        Returns:
            d: L2-normalised embedding, shape ``(B, sem_dim)``.
        """
        d = self.projector(obs)
        d = F.normalize(d, p=2, dim=-1)
        return d

    def barlow_twins_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the Barlow Twins loss using dropout-augmented views.

        Two forward passes with independent dropout masks create the two
        views required by the loss.

        Args:
            obs: Observation tensor ``(B, obs_dim)``.

        Returns:
            Scalar Barlow Twins loss.
        """
        view_a = self.aug_dropout(obs)
        view_b = self.aug_dropout(obs)
        d_a = self.forward(view_a)
        d_b = self.forward(view_b)
        return self.bt_loss_fn(d_a, d_b)


# ---------------------------------------------------------------------------
# Subtask A3 -- Dual Encoder
# ---------------------------------------------------------------------------

class DualEncoder(nn.Module):
    """Unified encoder that produces the dual-latent AgentLatent.

    Wraps a CategoricalVAE (visual branch) and a SemanticEncoder (semantic
    branch) and returns a single :class:`AgentLatent` per observation.  The
    recurrent hidden state ``h`` is left as ``None``; it will be populated
    downstream by the sequence model (Agent C).

    This module is instantiated **once** and called per agent -- no weight
    duplication across agents.

    Args:
        obs_dim:  Dimensionality of the observation vector.
        num_cats: Number of categorical distributions for the visual branch.
        cat_dim:  Classes per categorical distribution.
        sem_dim:  Dimensionality of the semantic embedding.
        hidden_dim: Hidden layer width shared by both sub-encoders.
    """

    def __init__(
        self,
        obs_dim: int,
        num_cats: int = 32,
        cat_dim: int = 32,
        sem_dim: int = 512,
        hidden_dim: int = 512,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        anneal_steps: int = 100_000,
        bt_lambda: float = 0.005,
        aug_drop_p: float = 0.1,
    ):
        super().__init__()
        self.visual_encoder = CategoricalVAE(
            obs_dim=obs_dim,
            num_cats=num_cats,
            cat_dim=cat_dim,
            hidden_dim=hidden_dim,
            tau_start=tau_start,
            tau_end=tau_end,
            anneal_steps=anneal_steps,
        )
        self.semantic_encoder = SemanticEncoder(
            obs_dim=obs_dim,
            sem_dim=sem_dim,
            hidden_dim=hidden_dim,
            aug_drop_p=aug_drop_p,
            bt_lambda=bt_lambda,
        )

    def forward(self, obs: torch.Tensor) -> AgentLatent:
        """Encode an observation into the dual-latent representation.

        Args:
            obs: Observation tensor, shape ``(B, obs_dim)``.

        Returns:
            :class:`AgentLatent` with ``z``, ``d`` populated and ``h=None``.
        """
        z, _logits, _recon = self.visual_encoder(obs)
        d = self.semantic_encoder(obs)
        return AgentLatent(z=z, d=d, h=None)

    # -- convenience accessors for loss computation -------------------------

    def encode_visual(self, obs: torch.Tensor):
        """Run only the visual branch.

        Returns:
            z, logits, recon -- same as :meth:`CategoricalVAE.forward`.
        """
        return self.visual_encoder(obs)

    def encode_semantic(self, obs: torch.Tensor) -> torch.Tensor:
        """Run only the semantic branch.

        Returns:
            d: L2-normalised embedding ``(B, sem_dim)``.
        """
        return self.semantic_encoder(obs)

    def compute_reconstruction_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """L_rec: MSE reconstruction loss from the visual branch.

        Performs a full visual-encoder forward pass internally.
        """
        _z, _logits, recon = self.visual_encoder(obs)
        return self.visual_encoder.reconstruction_loss(obs, recon)

    def compute_latent_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """L_latent: Barlow Twins loss from the semantic branch.

        Uses dropout-based augmentation internally.
        """
        return self.semantic_encoder.barlow_twins_loss(obs)
