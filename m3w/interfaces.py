"""
Shared interface contracts for EDELINE-MARL.
All agents produce and consume these dataclasses.
"""
from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class AgentLatent:
    """Per-agent dual latent state. All agents produce and consume this."""
    z: torch.Tensor        # visual latent,   shape: (B, num_cats, cat_dim)  float
    d: torch.Tensor        # semantic latent, shape: (B, sem_dim)            float
    h: torch.Tensor        # recurrent state, shape: (B, hidden_dim)         float
    # z is the straight-through one-hot from Categorical VAE
    # d is the L2-normalized Barlow Twins embedding
    # h is the Transformer SSM hidden state


@dataclass
class FlowPredictorOutput:
    x_hat: torch.Tensor    # predicted dual latent [z_hat; d_hat], shape (B, latent_dim)
    v_field: torch.Tensor  # velocity field at t=1 (for loss), shape (B, latent_dim)


@dataclass
class WorldModelOutput:
    latents: List[AgentLatent]   # predicted next latents for all N agents
    rewards: torch.Tensor        # predicted team reward, shape (B,)
    continues: torch.Tensor      # predicted episode continue flag, shape (B,)


# Dimension constants
LATENT_DIM: int = 1536           # = num_cats(32) * cat_dim(32) + sem_dim(512)
VISUAL_LATENT_DIM: int = 1024    # num_cats=32, cat_dim=32
SEM_LATENT_DIM: int = 512
