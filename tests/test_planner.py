"""Tests for the MPPI Planner."""

import torch
import torch.nn as nn
import pytest

from m3w.interfaces import AgentLatent, LATENT_DIM
from m3w.planner import MPPIPlanner


# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------

class MockVelocityField(nn.Module):
    """Minimal velocity field that returns zeros."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("_moe_bias", torch.zeros(latent_dim))

    def set_moe_bias(self, bias):
        self._moe_bias = bias

    def clear_moe_bias(self):
        self._moe_bias = torch.zeros(self.latent_dim)

    def forward(self, x_tau, tau, h, m):
        return torch.zeros_like(x_tau)


class MockFlowPredictor(nn.Module):
    """Mock flow predictor that returns random latents."""

    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self._velocity_field = MockVelocityField(latent_dim)

    @property
    def velocity_field(self):
        return self._velocity_field

    @torch.no_grad()
    def sample(self, h, m, steps=1):
        B = h.shape[0]
        from m3w.interfaces import FlowPredictorOutput
        x_hat = torch.randn(B, self.latent_dim, device=h.device)
        v_field = torch.zeros(B, self.latent_dim, device=h.device)
        return FlowPredictorOutput(x_hat=x_hat, v_field=v_field)


class MockSoftMoE(nn.Module):
    """Mock SoftMoE that returns zero bias."""

    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def compute_bias(self, h):
        B = h.shape[0]
        return torch.zeros(B, self.latent_dim, device=h.device)


class MockRewardHead(nn.Module):
    """Mock reward head that returns uniform logits."""

    def __init__(self, n_agents, hidden_dim, num_bins=101):
        super().__init__()
        self.num_bins = num_bins
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.per_agent_dim = hidden_dim  # mock: no latent padding needed

    def forward(self, s_t):
        B = s_t.shape[0]
        logits = torch.zeros(B, self.num_bins, device=s_t.device)
        aux = {"loss_balancing": torch.tensor(0.0, device=s_t.device),
               "gates": None, "logits": None}
        return logits, aux


class MockRewardProcessor:
    """Mock two-hot processor that decodes logits to scalar."""

    def logits_decode_scalar(self, logits):
        B = logits.shape[0]
        return torch.zeros(B, 1, device=logits.device)


class MockWorldModel:
    """Composes all mock sub-components."""

    def __init__(self, num_agents, hidden_dim, latent_dim=LATENT_DIM, num_bins=101):
        self.flow_predictor = MockFlowPredictor(latent_dim, hidden_dim)
        self.soft_moe = MockSoftMoE(hidden_dim, latent_dim)
        self.reward_head = MockRewardHead(num_agents, hidden_dim, num_bins)
        self.reward_processor = MockRewardProcessor()


class MockActor:
    """Mock actor that returns random actions of correct shape."""

    def __init__(self, action_dim, device="cpu"):
        self.action_dim = action_dim
        self.device = device

    def get_actions(self, obs, stochastic=True, available_actions=None):
        B = obs.shape[0]
        return torch.randn(B, self.action_dim, device=obs.device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def planner_cfg():
    return {
        "horizon": 3,
        "num_samples": 32,
        "num_elites": 8,
        "temperature": 0.5,
        "num_pi_trajs": 4,
        "max_std": 1.0,
        "min_std": 0.05,
        "iterations": 3,
        "gamma": 0.99,
        "flow_steps_infer": 1,
    }


@pytest.fixture
def setup():
    """Create test fixtures with B=4, num_agents=2, action_dim=6, hidden_dim=512."""
    B = 4
    num_agents = 2
    action_dim = 6
    hidden_dim = 512
    num_cats = 32
    cat_dim = 32
    sem_dim = 512
    device = "cpu"

    latents = []
    messages = []
    for i in range(num_agents):
        lat = AgentLatent(
            z=torch.randn(B, num_cats, cat_dim),
            d=torch.randn(B, sem_dim),
            h=torch.randn(B, hidden_dim),
        )
        latents.append(lat)
        messages.append(torch.randn(B, hidden_dim))

    world_model = MockWorldModel(num_agents, hidden_dim)
    actors = [MockActor(action_dim, device) for _ in range(num_agents)]
    action_dims = [action_dim] * num_agents

    return {
        "B": B,
        "num_agents": num_agents,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "latents": latents,
        "messages": messages,
        "world_model": world_model,
        "actors": actors,
        "action_dims": action_dims,
    }


class TestMPPIPlanner:

    def test_plan_returns_correct_shape(self, planner_cfg, setup):
        """plan() should return (B, num_agents, action_dim)."""
        planner = MPPIPlanner(planner_cfg)
        actions = planner.plan(
            world_model=setup["world_model"],
            latents=setup["latents"],
            messages=setup["messages"],
            actors=setup["actors"],
            action_dims=setup["action_dims"],
        )
        assert isinstance(actions, type(actions))
        assert actions.shape == (
            setup["B"],
            setup["num_agents"],
            setup["action_dim"],
        )

    def test_plan_actions_in_range(self, planner_cfg, setup):
        """Actions should be clamped to [-1, 1]."""
        planner = MPPIPlanner(planner_cfg)
        actions = planner.plan(
            world_model=setup["world_model"],
            latents=setup["latents"],
            messages=setup["messages"],
            actors=setup["actors"],
            action_dims=setup["action_dims"],
        )
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_plan_eval_mode(self, planner_cfg, setup):
        """Eval mode should still produce correct shapes."""
        planner = MPPIPlanner(planner_cfg)
        actions = planner.plan(
            world_model=setup["world_model"],
            latents=setup["latents"],
            messages=setup["messages"],
            actors=setup["actors"],
            action_dims=setup["action_dims"],
            eval_mode=True,
        )
        assert actions.shape == (
            setup["B"],
            setup["num_agents"],
            setup["action_dim"],
        )

    def test_plan_with_t0_flags(self, planner_cfg, setup):
        """t0 flags should not change output shape."""
        planner = MPPIPlanner(planner_cfg)
        # First call to establish running mean
        planner.plan(
            world_model=setup["world_model"],
            latents=setup["latents"],
            messages=setup["messages"],
            actors=setup["actors"],
            action_dims=setup["action_dims"],
            t0=[True] * setup["B"],
        )
        # Second call with mixed t0 (warm-start for some envs)
        t0 = [False, True, False, True]
        actions = planner.plan(
            world_model=setup["world_model"],
            latents=setup["latents"],
            messages=setup["messages"],
            actors=setup["actors"],
            action_dims=setup["action_dims"],
            t0=t0,
        )
        assert actions.shape == (
            setup["B"],
            setup["num_agents"],
            setup["action_dim"],
        )

    def test_plan_no_pi_trajs(self, planner_cfg, setup):
        """Planner should work with num_pi_trajs=0."""
        planner_cfg["num_pi_trajs"] = 0
        planner = MPPIPlanner(planner_cfg)
        actions = planner.plan(
            world_model=setup["world_model"],
            latents=setup["latents"],
            messages=setup["messages"],
            actors=setup["actors"],
            action_dims=setup["action_dims"],
        )
        assert actions.shape == (
            setup["B"],
            setup["num_agents"],
            setup["action_dim"],
        )

    def test_running_mean_updated(self, planner_cfg, setup):
        """Running mean should be set after planning."""
        planner = MPPIPlanner(planner_cfg)
        assert planner._running_mean is None
        planner.plan(
            world_model=setup["world_model"],
            latents=setup["latents"],
            messages=setup["messages"],
            actors=setup["actors"],
            action_dims=setup["action_dims"],
        )
        assert planner._running_mean is not None
        assert len(planner._running_mean) == setup["num_agents"]
        for i in range(setup["num_agents"]):
            assert planner._running_mean[i].shape == (
                planner_cfg["horizon"],
                setup["B"],
                setup["action_dim"],
            )

    def test_single_iteration(self, planner_cfg, setup):
        """Planner should work with a single iteration."""
        planner_cfg["iterations"] = 1
        planner = MPPIPlanner(planner_cfg)
        actions = planner.plan(
            world_model=setup["world_model"],
            latents=setup["latents"],
            messages=setup["messages"],
            actors=setup["actors"],
            action_dims=setup["action_dims"],
        )
        assert actions.shape == (
            setup["B"],
            setup["num_agents"],
            setup["action_dim"],
        )

    def test_batch_size_one(self, planner_cfg, setup):
        """Planner should work with batch size 1."""
        B = 1
        num_agents = setup["num_agents"]
        hidden_dim = setup["hidden_dim"]
        num_cats, cat_dim, sem_dim = 32, 32, 512

        latents = [
            AgentLatent(
                z=torch.randn(B, num_cats, cat_dim),
                d=torch.randn(B, sem_dim),
                h=torch.randn(B, hidden_dim),
            )
            for _ in range(num_agents)
        ]
        messages = [torch.randn(B, hidden_dim) for _ in range(num_agents)]

        planner = MPPIPlanner(planner_cfg)
        actions = planner.plan(
            world_model=setup["world_model"],
            latents=latents,
            messages=messages,
            actors=setup["actors"],
            action_dims=setup["action_dims"],
        )
        assert actions.shape == (B, num_agents, setup["action_dim"])
