"""
Unit tests for the unified WorldModel (m3w/world_model.py).
"""

import pytest
import torch

from m3w.world_model import WorldModel, LinearDynamicsPredictor, _default_cfg
from m3w.interfaces import WorldModelOutput, AgentLatent, LATENT_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B = 4
OBS_DIM = 64
ACTION_DIM = 6
NUM_AGENTS = 2
HIDDEN_DIM = 128  # small for fast tests
# num_cats * cat_dim + sem_dim MUST equal LATENT_DIM (1536) because
# SparseMoEReward hardcodes LATENT_DIM for its internal per_agent_dim.
NUM_CATS = 32
CAT_DIM = 32
SEM_DIM = 512
NUM_EXPERTS = 4
NUM_REWARD_BINS = 11


def make_cfg(**overrides) -> dict:
    """Create a test-friendly cfg dict with small dimensions."""
    cfg = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        num_agents=NUM_AGENTS,
        hidden_dim=HIDDEN_DIM,
        num_cats=NUM_CATS,
        cat_dim=CAT_DIM,
        sem_dim=SEM_DIM,
        num_experts=NUM_EXPERTS,
        num_slots=1,
        num_reward_bins=NUM_REWARD_BINS,
        reward_vmin=-20,
        reward_vmax=20,
        cpc_steps_ahead=3,
        shortcut_lambda=0.1,
        loss_weights={
            "L_pred": 1.0,
            "L_rec": 1.0,
            "L_rep": 0.1,
            "L_dyn": 0.5,
            "L_cpc": 0.1,
            "L_latent": 1.0,
            "L_flow": 1.0,
            "L_shortcut": 1.0,
        },
    )
    cfg.update(overrides)
    return cfg


def make_inputs(cfg: dict, batch_size: int = B):
    """Create dummy inputs matching the cfg."""
    N = cfg["num_agents"]
    obs_dim = cfg["obs_dim"]
    action_dim = cfg["action_dim"]

    obs_list = [torch.randn(batch_size, obs_dim) for _ in range(N)]
    action_list = [torch.randn(batch_size, action_dim) for _ in range(N)]
    next_obs_list = [torch.randn(batch_size, obs_dim) for _ in range(N)]
    rewards = torch.randn(batch_size)
    continues = torch.ones(batch_size)
    return obs_list, action_list, next_obs_list, rewards, continues


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWorldModelInit:
    """E1: WorldModel.__init__ succeeds with valid cfg."""

    def test_init_default(self):
        cfg = make_cfg()
        model = WorldModel(cfg)
        assert model is not None
        assert model.num_agents == NUM_AGENTS
        assert model.hidden_dim == HIDDEN_DIM

    def test_init_preserves_cfg_keys(self):
        cfg = make_cfg()
        model = WorldModel(cfg)
        assert model.cfg["obs_dim"] == OBS_DIM
        assert model.cfg["action_dim"] == ACTION_DIM

    def test_submodules_exist(self):
        cfg = make_cfg()
        model = WorldModel(cfg)
        assert hasattr(model, "encoder")
        assert hasattr(model, "sequence_model")
        assert hasattr(model, "moe_velocity_bias")
        assert hasattr(model, "flow_predictor")
        assert hasattr(model, "reward_predictor")
        assert hasattr(model, "continue_predictor")
        assert hasattr(model, "accpc")
        assert hasattr(model, "dynamics_predictor")


class TestWorldModelForward:
    """E2: WorldModel.forward returns WorldModelOutput with correct shapes."""

    def test_forward_training(self):
        cfg = make_cfg()
        model = WorldModel(cfg)
        model.train()

        obs_list, action_list, next_obs_list, _, _ = make_inputs(cfg)
        out = model(obs_list, action_list, target_obs_list=next_obs_list)

        assert isinstance(out, WorldModelOutput)
        assert len(out.latents) == NUM_AGENTS
        assert out.rewards.shape == (B,)
        assert out.continues.shape == (B,)

    def test_forward_eval(self):
        cfg = make_cfg()
        model = WorldModel(cfg)
        model.eval()

        obs_list, action_list, _, _, _ = make_inputs(cfg)
        with torch.no_grad():
            out = model(obs_list, action_list)

        assert isinstance(out, WorldModelOutput)
        assert len(out.latents) == NUM_AGENTS
        assert out.rewards.shape == (B,)
        assert out.continues.shape == (B,)

    def test_latent_shapes(self):
        cfg = make_cfg()
        model = WorldModel(cfg)
        model.eval()

        obs_list, action_list, _, _, _ = make_inputs(cfg)
        with torch.no_grad():
            out = model(obs_list, action_list)

        for lat in out.latents:
            assert isinstance(lat, AgentLatent)
            assert lat.z.shape == (B, NUM_CATS, CAT_DIM)
            assert lat.d.shape == (B, SEM_DIM)
            # h is None for predicted latents (split from x_hat)
            assert lat.h is None


class TestWorldModelLoss:
    """E3: WorldModel.compute_loss returns dict with expected keys, all scalar."""

    EXPECTED_KEYS = {
        "L_total", "L_pred", "L_rec", "L_rep",
        "L_dyn", "L_cpc", "L_latent", "L_flow", "L_shortcut",
    }

    def test_loss_keys(self):
        cfg = make_cfg()
        model = WorldModel(cfg)
        model.train()

        obs_list, action_list, next_obs_list, rewards, continues = make_inputs(cfg)
        losses = model.compute_loss(obs_list, action_list, next_obs_list, rewards, continues)

        assert set(losses.keys()) == self.EXPECTED_KEYS

    def test_losses_are_scalars(self):
        cfg = make_cfg()
        model = WorldModel(cfg)
        model.train()

        obs_list, action_list, next_obs_list, rewards, continues = make_inputs(cfg)
        losses = model.compute_loss(obs_list, action_list, next_obs_list, rewards, continues)

        for key, val in losses.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a tensor"
            assert val.dim() == 0, f"{key} is not scalar, shape={val.shape}"

    def test_losses_finite(self):
        cfg = make_cfg()
        model = WorldModel(cfg)
        model.train()

        obs_list, action_list, next_obs_list, rewards, continues = make_inputs(cfg)
        losses = model.compute_loss(obs_list, action_list, next_obs_list, rewards, continues)

        for key, val in losses.items():
            assert torch.isfinite(val), f"{key} is not finite: {val.item()}"

    def test_gradient_flows(self):
        """L_total.backward() succeeds without error."""
        cfg = make_cfg()
        model = WorldModel(cfg)
        model.train()

        obs_list, action_list, next_obs_list, rewards, continues = make_inputs(cfg)
        losses = model.compute_loss(obs_list, action_list, next_obs_list, rewards, continues)

        losses["L_total"].backward()

        # Check that at least some parameters received gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameter received a non-zero gradient"


class TestLinearDynamicsPredictor:
    """E4: LinearDynamicsPredictor."""

    def test_output_shape(self):
        predictor = LinearDynamicsPredictor(hidden_dim=HIDDEN_DIM, latent_dim=128)
        h = torch.randn(B, HIDDEN_DIM)
        out = predictor(h)
        assert out.shape == (B, 128)

    def test_gradient(self):
        predictor = LinearDynamicsPredictor(hidden_dim=HIDDEN_DIM, latent_dim=128)
        h = torch.randn(B, HIDDEN_DIM, requires_grad=True)
        out = predictor(h)
        out.sum().backward()
        assert h.grad is not None


class TestDefaultCfg:
    """Verify _default_cfg returns a valid configuration."""

    def test_default_cfg_keys(self):
        cfg = _default_cfg()
        required_keys = {
            "obs_dim", "action_dim", "num_agents", "hidden_dim",
            "num_cats", "cat_dim", "sem_dim", "num_experts", "num_slots",
            "num_reward_bins", "reward_vmin", "reward_vmax",
            "cpc_steps_ahead", "shortcut_lambda", "loss_weights",
        }
        assert required_keys.issubset(set(cfg.keys()))

    def test_default_cfg_builds_model(self):
        cfg = _default_cfg()
        model = WorldModel(cfg)
        assert model is not None
