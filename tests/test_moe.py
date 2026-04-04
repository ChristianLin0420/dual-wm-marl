"""Unit tests for m3w.moe: SoftMoEVelocityBias, SparseMoEReward, ContinuePredictor."""

import pytest
import torch

from m3w.interfaces import LATENT_DIM
from m3w.moe import ContinuePredictor, SoftMoEVelocityBias, SparseMoEReward

# ---- Shared fixtures -------------------------------------------------------

B = 8
HIDDEN_DIM = 512
N_AGENTS = 2
NUM_EXPERTS = 16
K = 2
NUM_REWARD_BINS = 101
PER_AGENT_DIM = LATENT_DIM + HIDDEN_DIM
S_T_DIM = N_AGENTS * PER_AGENT_DIM


@pytest.fixture
def velocity_bias_model():
    return SoftMoEVelocityBias(
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_experts=NUM_EXPERTS,
        num_slots=1,
    )


@pytest.fixture
def reward_model():
    return SparseMoEReward(
        n_agents=N_AGENTS,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        k=K,
        num_reward_bins=NUM_REWARD_BINS,
    )


@pytest.fixture
def continue_model():
    return ContinuePredictor(input_dim=S_T_DIM)


# ---- D1: SoftMoEVelocityBias -----------------------------------------------


class TestSoftMoEVelocityBias:
    def test_output_shape(self, velocity_bias_model):
        h = torch.randn(B, HIDDEN_DIM)
        out = velocity_bias_model.compute_bias(h)
        assert out.shape == (B, LATENT_DIM), (
            f"Expected ({B}, {LATENT_DIM}), got {out.shape}"
        )

    def test_gradient_flow(self, velocity_bias_model):
        h = torch.randn(B, HIDDEN_DIM, requires_grad=True)
        out = velocity_bias_model.compute_bias(h)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None, "Gradient did not flow back to input h"
        assert h.grad.shape == h.shape

    def test_batch_size_one(self, velocity_bias_model):
        h = torch.randn(1, HIDDEN_DIM)
        out = velocity_bias_model.compute_bias(h)
        assert out.shape == (1, LATENT_DIM)

    def test_deterministic_eval(self, velocity_bias_model):
        velocity_bias_model.eval()
        h = torch.randn(B, HIDDEN_DIM)
        out1 = velocity_bias_model.compute_bias(h)
        out2 = velocity_bias_model.compute_bias(h)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# ---- D2: SparseMoEReward ---------------------------------------------------


class TestSparseMoEReward:
    def test_output_shape(self, reward_model):
        s_t = torch.randn(B, S_T_DIM)
        r_logits, aux = reward_model(s_t)
        assert r_logits.shape == (B, NUM_REWARD_BINS), (
            f"Expected ({B}, {NUM_REWARD_BINS}), got {r_logits.shape}"
        )

    def test_aux_keys(self, reward_model):
        s_t = torch.randn(B, S_T_DIM)
        _r_logits, aux = reward_model(s_t)
        assert "loss_balancing" in aux
        assert "gates" in aux
        assert "logits" in aux

    def test_gates_shape(self, reward_model):
        s_t = torch.randn(B, S_T_DIM)
        _r_logits, aux = reward_model(s_t)
        assert aux["gates"].shape == (B, NUM_EXPERTS)

    def test_gradient_flow(self, reward_model):
        s_t = torch.randn(B, S_T_DIM, requires_grad=True)
        r_logits, aux = reward_model(s_t)
        loss = r_logits.sum() + aux["loss_balancing"]
        loss.backward()
        assert s_t.grad is not None, "Gradient did not flow back to input s_t"


# ---- D3: ContinuePredictor -------------------------------------------------


class TestContinuePredictor:
    def test_output_shape(self, continue_model):
        s_t = torch.randn(B, S_T_DIM)
        out = continue_model(s_t)
        assert out.shape == (B, 1), f"Expected ({B}, 1), got {out.shape}"

    def test_gradient_flow(self, continue_model):
        s_t = torch.randn(B, S_T_DIM, requires_grad=True)
        out = continue_model(s_t)
        out.sum().backward()
        assert s_t.grad is not None, "Gradient did not flow back to input s_t"

    def test_batch_size_one(self, continue_model):
        s_t = torch.randn(1, S_T_DIM)
        out = continue_model(s_t)
        assert out.shape == (1, 1)
