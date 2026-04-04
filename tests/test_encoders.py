"""Unit tests for the dual encoder branches (CategoricalVAE, SemanticEncoder, DualEncoder)."""

import torch
import pytest

from m3w.interfaces import AgentLatent
from m3w.encoders import (
    CategoricalVAE,
    SemanticEncoder,
    BarlowTwinsLoss,
    DualEncoder,
)

# Test constants
B = 8
OBS_DIM = 64
NUM_CATS = 32
CAT_DIM = 32
SEM_DIM = 512


# ---------------------------------------------------------------------------
# CategoricalVAE tests
# ---------------------------------------------------------------------------

class TestCategoricalVAE:
    @pytest.fixture
    def vae(self):
        return CategoricalVAE(obs_dim=OBS_DIM, num_cats=NUM_CATS, cat_dim=CAT_DIM)

    @pytest.fixture
    def obs(self):
        return torch.randn(B, OBS_DIM)

    def test_output_shapes(self, vae, obs):
        """z, logits, and recon should have the documented shapes."""
        z, logits, recon = vae(obs)
        assert z.shape == (B, NUM_CATS, CAT_DIM)
        assert logits.shape == (B, NUM_CATS, CAT_DIM)
        assert recon.shape == (B, OBS_DIM)

    def test_gumbel_softmax_one_hot(self, vae, obs):
        """In training mode, z should be (approximately) one-hot along cat_dim."""
        vae.train()
        z, _, _ = vae(obs)
        # Each row along the last dim should sum to 1.
        row_sums = z.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        # Hard one-hot: values should be 0 or 1.
        assert torch.allclose(z, z.round(), atol=1e-5)

    def test_eval_mode_argmax(self, vae, obs):
        """In eval mode, z should be strictly one-hot (argmax)."""
        vae.eval()
        with torch.no_grad():
            z, _, _ = vae(obs)
        row_sums = z.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))
        assert set(z.unique().tolist()).issubset({0.0, 1.0})

    def test_reconstruction_loss_scalar(self, vae, obs):
        """reconstruction_loss should return a scalar."""
        _, _, recon = vae(obs)
        loss = vae.reconstruction_loss(obs, recon)
        assert loss.dim() == 0

    def test_temperature_annealing(self, vae):
        """Temperature should decrease from tau_start towards tau_end."""
        tau_0 = vae.tau
        # Simulate many steps.
        obs = torch.randn(2, OBS_DIM)
        vae.train()
        for _ in range(1000):
            vae(obs)
        tau_after = vae.tau
        assert tau_after < tau_0


# ---------------------------------------------------------------------------
# SemanticEncoder tests
# ---------------------------------------------------------------------------

class TestSemanticEncoder:
    @pytest.fixture
    def encoder(self):
        return SemanticEncoder(obs_dim=OBS_DIM, sem_dim=SEM_DIM)

    @pytest.fixture
    def obs(self):
        return torch.randn(B, OBS_DIM)

    def test_output_shape(self, encoder, obs):
        """Output should be (B, sem_dim)."""
        d = encoder(obs)
        assert d.shape == (B, SEM_DIM)

    def test_l2_normalised(self, encoder, obs):
        """Each embedding vector should have unit L2 norm."""
        d = encoder(obs)
        norms = torch.norm(d, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_barlow_twins_loss_scalar(self, encoder, obs):
        """barlow_twins_loss should return a scalar."""
        encoder.train()
        loss = encoder.barlow_twins_loss(obs)
        assert loss.dim() == 0
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# BarlowTwinsLoss tests
# ---------------------------------------------------------------------------

class TestBarlowTwinsLoss:
    def test_returns_scalar(self):
        """Loss should be a scalar tensor."""
        loss_fn = BarlowTwinsLoss()
        z_a = torch.randn(B, SEM_DIM)
        z_b = torch.randn(B, SEM_DIM)
        loss = loss_fn(z_a, z_b)
        assert loss.dim() == 0

    def test_identical_inputs_low_loss(self):
        """When both views are identical the invariance term should be ~0."""
        loss_fn = BarlowTwinsLoss()
        z = torch.randn(B, SEM_DIM)
        loss_same = loss_fn(z, z)
        loss_diff = loss_fn(z, torch.randn_like(z))
        # Identical inputs should yield lower (or equal) loss than random.
        assert loss_same.item() <= loss_diff.item() + 1e-3


# ---------------------------------------------------------------------------
# DualEncoder tests
# ---------------------------------------------------------------------------

class TestDualEncoder:
    @pytest.fixture
    def dual(self):
        return DualEncoder(obs_dim=OBS_DIM, num_cats=NUM_CATS, cat_dim=CAT_DIM, sem_dim=SEM_DIM)

    @pytest.fixture
    def obs(self):
        return torch.randn(B, OBS_DIM)

    def test_forward_returns_agent_latent(self, dual, obs):
        """forward should return an AgentLatent dataclass."""
        out = dual(obs)
        assert isinstance(out, AgentLatent)

    def test_z_shape(self, dual, obs):
        out = dual(obs)
        assert out.z.shape == (B, NUM_CATS, CAT_DIM)

    def test_d_shape(self, dual, obs):
        out = dual(obs)
        assert out.d.shape == (B, SEM_DIM)

    def test_h_is_none(self, dual, obs):
        """h should be None (placeholder for the sequence model)."""
        out = dual(obs)
        assert out.h is None

    def test_compute_reconstruction_loss(self, dual, obs):
        loss = dual.compute_reconstruction_loss(obs)
        assert loss.dim() == 0

    def test_compute_latent_loss(self, dual, obs):
        dual.train()
        loss = dual.compute_latent_loss(obs)
        assert loss.dim() == 0

    def test_no_shared_weights(self, dual):
        """Visual and semantic branches should have independent parameters."""
        vis_params = set(id(p) for p in dual.visual_encoder.parameters())
        sem_params = set(id(p) for p in dual.semantic_encoder.parameters())
        assert vis_params.isdisjoint(sem_params)
