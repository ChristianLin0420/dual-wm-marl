"""
Unit tests for m3w.sequence_model (TSSM-MA components).

Covers:
  - C1: ProjectDualLatent
  - C2: TransformerSSM
  - C3: CrossAgentAttention
  - C4: TransformerSSM_MA
  - C5: ACCPC_MA
"""

import pytest
import torch

from m3w.interfaces import AgentLatent
from m3w.sequence_model import (
    ProjectDualLatent,
    TransformerSSM,
    CrossAgentAttention,
    TransformerSSM_MA,
    ACCPC_MA,
)

# Test constants
B = 8            # batch size
N = 2            # number of agents
T = 5            # sequence length
ACTION_DIM = 6
HIDDEN_DIM = 512
NUM_CATS = 32
CAT_DIM = 32
SEM_DIM = 512
VISUAL_DIM = NUM_CATS * CAT_DIM  # 1024


def _make_latent(batch_size: int = B, h: torch.Tensor = None) -> AgentLatent:
    """Create a dummy AgentLatent."""
    return AgentLatent(
        z=torch.randn(batch_size, NUM_CATS, CAT_DIM),
        d=torch.randn(batch_size, SEM_DIM),
        h=h if h is not None else torch.zeros(batch_size, HIDDEN_DIM),
    )


# -----------------------------------------------------------------------
# C1: ProjectDualLatent
# -----------------------------------------------------------------------

class TestProjectDualLatent:
    def test_output_shape(self):
        proj = ProjectDualLatent(
            visual_dim=VISUAL_DIM, sem_dim=SEM_DIM, hidden_dim=HIDDEN_DIM
        )
        z = torch.randn(B, NUM_CATS, CAT_DIM)
        d = torch.randn(B, SEM_DIM)
        out = proj(z, d)
        assert out.shape == (B, HIDDEN_DIM)

    def test_gradient_flows(self):
        proj = ProjectDualLatent(
            visual_dim=VISUAL_DIM, sem_dim=SEM_DIM, hidden_dim=HIDDEN_DIM
        )
        z = torch.randn(B, NUM_CATS, CAT_DIM, requires_grad=True)
        d = torch.randn(B, SEM_DIM, requires_grad=True)
        out = proj(z, d)
        out.sum().backward()
        assert z.grad is not None
        assert d.grad is not None

    def test_different_batch_sizes(self):
        proj = ProjectDualLatent(
            visual_dim=VISUAL_DIM, sem_dim=SEM_DIM, hidden_dim=HIDDEN_DIM
        )
        for bs in [1, 4, 16]:
            z = torch.randn(bs, NUM_CATS, CAT_DIM)
            d = torch.randn(bs, SEM_DIM)
            out = proj(z, d)
            assert out.shape == (bs, HIDDEN_DIM)


# -----------------------------------------------------------------------
# C2: TransformerSSM
# -----------------------------------------------------------------------

class TestTransformerSSM:
    def test_output_shape(self):
        ssm = TransformerSSM(
            hidden_dim=HIDDEN_DIM, num_heads=8, num_layers=3, action_dim=ACTION_DIM
        )
        projected = torch.randn(B, T, HIDDEN_DIM)
        actions = torch.randn(B, T, ACTION_DIM)
        out = ssm(projected, actions)
        assert out.shape == (B, T, HIDDEN_DIM)

    def test_single_step(self):
        ssm = TransformerSSM(
            hidden_dim=HIDDEN_DIM, num_heads=8, num_layers=3, action_dim=ACTION_DIM
        )
        projected = torch.randn(B, 1, HIDDEN_DIM)
        actions = torch.randn(B, 1, ACTION_DIM)
        out = ssm(projected, actions)
        assert out.shape == (B, 1, HIDDEN_DIM)

    def test_causal_mask(self):
        """Verify that the output at time t does not depend on future inputs."""
        ssm = TransformerSSM(
            hidden_dim=HIDDEN_DIM, num_heads=8, num_layers=3, action_dim=ACTION_DIM
        )
        ssm.eval()
        projected = torch.randn(B, T, HIDDEN_DIM)
        actions = torch.randn(B, T, ACTION_DIM)

        # Full sequence
        out_full = ssm(projected, actions)

        # Truncated to first 3 steps
        out_trunc = ssm(projected[:, :3], actions[:, :3])

        # The first 3 outputs should be the same (causal)
        assert torch.allclose(out_full[:, :3], out_trunc, atol=1e-5)

    def test_padding_mask(self):
        ssm = TransformerSSM(
            hidden_dim=HIDDEN_DIM, num_heads=8, num_layers=3, action_dim=ACTION_DIM
        )
        ssm.eval()
        projected = torch.randn(B, T, HIDDEN_DIM)
        actions = torch.randn(B, T, ACTION_DIM)
        padding_mask = torch.zeros(B, T, dtype=torch.bool)
        padding_mask[:, -2:] = True  # Last 2 steps are padded

        out = ssm(projected, actions, padding_mask=padding_mask)
        assert out.shape == (B, T, HIDDEN_DIM)

    def test_gradient_flows(self):
        ssm = TransformerSSM(
            hidden_dim=HIDDEN_DIM, num_heads=8, num_layers=3, action_dim=ACTION_DIM
        )
        projected = torch.randn(B, T, HIDDEN_DIM, requires_grad=True)
        actions = torch.randn(B, T, ACTION_DIM)
        out = ssm(projected, actions)
        out.sum().backward()
        assert projected.grad is not None


# -----------------------------------------------------------------------
# C3: CrossAgentAttention
# -----------------------------------------------------------------------

class TestCrossAgentAttention:
    def test_output_shape(self):
        cross_attn = CrossAgentAttention(hidden_dim=HIDDEN_DIM, num_heads=4)
        agent_hiddens = torch.randn(B, N, HIDDEN_DIM)
        messages = cross_attn(agent_hiddens)
        assert messages.shape == (B, N, HIDDEN_DIM)

    def test_three_agents(self):
        cross_attn = CrossAgentAttention(hidden_dim=HIDDEN_DIM, num_heads=4)
        agent_hiddens = torch.randn(B, 3, HIDDEN_DIM)
        messages = cross_attn(agent_hiddens)
        assert messages.shape == (B, 3, HIDDEN_DIM)

    def test_single_agent_zero_message(self):
        cross_attn = CrossAgentAttention(hidden_dim=HIDDEN_DIM, num_heads=4)
        agent_hiddens = torch.randn(B, 1, HIDDEN_DIM)
        messages = cross_attn(agent_hiddens)
        # Single agent should get zero message
        assert messages.shape == (B, 1, HIDDEN_DIM)
        assert torch.allclose(messages, torch.zeros_like(messages))

    def test_padding_mask(self):
        cross_attn = CrossAgentAttention(hidden_dim=HIDDEN_DIM, num_heads=4)
        agent_hiddens = torch.randn(B, 3, HIDDEN_DIM)
        padding_mask = torch.zeros(B, 3, dtype=torch.bool)
        padding_mask[:, 2] = True  # Third agent is padded/absent
        messages = cross_attn(agent_hiddens, padding_mask=padding_mask)
        assert messages.shape == (B, 3, HIDDEN_DIM)

    def test_gradient_flows(self):
        cross_attn = CrossAgentAttention(hidden_dim=HIDDEN_DIM, num_heads=4)
        agent_hiddens = torch.randn(B, N, HIDDEN_DIM, requires_grad=True)
        messages = cross_attn(agent_hiddens)
        messages.sum().backward()
        assert agent_hiddens.grad is not None


# -----------------------------------------------------------------------
# C4: TransformerSSM_MA
# -----------------------------------------------------------------------

class TestTransformerSSM_MA:
    def test_forward_output_types(self):
        model = TransformerSSM_MA(
            hidden_dim=HIDDEN_DIM,
            num_heads=8,
            num_layers=3,
            action_dim=ACTION_DIM,
            num_agents=N,
        )
        latents = [_make_latent() for _ in range(N)]
        actions = torch.randn(B, N, ACTION_DIM)

        updated, messages = model(latents, actions)

        assert isinstance(updated, list)
        assert len(updated) == N
        assert isinstance(messages, list)
        assert len(messages) == N

    def test_h_filled(self):
        model = TransformerSSM_MA(
            hidden_dim=HIDDEN_DIM,
            num_heads=8,
            num_layers=3,
            action_dim=ACTION_DIM,
            num_agents=N,
        )
        latents = [_make_latent() for _ in range(N)]
        actions = torch.randn(B, N, ACTION_DIM)

        updated, messages = model(latents, actions)

        for i in range(N):
            assert updated[i].h is not None
            assert updated[i].h.shape == (B, HIDDEN_DIM)
            assert messages[i].shape == (B, HIDDEN_DIM)

    def test_z_d_preserved(self):
        model = TransformerSSM_MA(
            hidden_dim=HIDDEN_DIM,
            num_heads=8,
            num_layers=3,
            action_dim=ACTION_DIM,
            num_agents=N,
        )
        latents = [_make_latent() for _ in range(N)]
        actions = torch.randn(B, N, ACTION_DIM)

        updated, _ = model(latents, actions)

        for i in range(N):
            assert torch.equal(updated[i].z, latents[i].z)
            assert torch.equal(updated[i].d, latents[i].d)

    def test_actions_as_list(self):
        model = TransformerSSM_MA(
            hidden_dim=HIDDEN_DIM,
            num_heads=8,
            num_layers=3,
            action_dim=ACTION_DIM,
            num_agents=N,
        )
        latents = [_make_latent() for _ in range(N)]
        actions_list = [torch.randn(B, ACTION_DIM) for _ in range(N)]

        updated, messages = model(latents, actions_list)
        assert len(updated) == N
        assert len(messages) == N

    def test_gradient_flows(self):
        model = TransformerSSM_MA(
            hidden_dim=HIDDEN_DIM,
            num_heads=8,
            num_layers=3,
            action_dim=ACTION_DIM,
            num_agents=N,
        )
        z = torch.randn(B, NUM_CATS, CAT_DIM, requires_grad=True)
        d = torch.randn(B, SEM_DIM, requires_grad=True)
        latents = [
            AgentLatent(z=z, d=d, h=torch.zeros(B, HIDDEN_DIM)),
            _make_latent(),
        ]
        actions = torch.randn(B, N, ACTION_DIM)

        updated, messages = model(latents, actions)
        loss = sum(m.sum() for m in messages) + sum(u.h.sum() for u in updated)
        loss.backward()
        assert z.grad is not None
        assert d.grad is not None


# -----------------------------------------------------------------------
# C5: ACCPC_MA
# -----------------------------------------------------------------------

class TestACCPC_MA:
    def test_compute_loss_scalar(self):
        num_steps_ahead = 3
        cpc = ACCPC_MA(
            hidden_dim=HIDDEN_DIM,
            latent_dim=VISUAL_DIM + SEM_DIM,
            num_steps_ahead=num_steps_ahead,
            action_dim=ACTION_DIM,
        )

        # N agents, T timesteps, T + num_steps_ahead future slots
        total_t = T + num_steps_ahead
        agent_states = [
            [_make_latent() for _ in range(T)] for _ in range(N)
        ]
        agent_actions = torch.randn(B, N, T, ACTION_DIM)
        agent_future_latents = [
            [_make_latent() for _ in range(total_t)] for _ in range(N)
        ]

        loss = cpc.compute_loss(agent_states, agent_actions, agent_future_latents)

        assert loss.shape == ()  # scalar
        assert loss.item() > 0  # InfoNCE loss is always positive

    def test_gradient_flows(self):
        num_steps_ahead = 2
        cpc = ACCPC_MA(
            hidden_dim=HIDDEN_DIM,
            latent_dim=VISUAL_DIM + SEM_DIM,
            num_steps_ahead=num_steps_ahead,
            action_dim=ACTION_DIM,
        )

        total_t = T + num_steps_ahead
        agent_states = [
            [_make_latent() for _ in range(T)] for _ in range(N)
        ]
        agent_actions = torch.randn(B, N, T, ACTION_DIM)
        agent_future_latents = [
            [_make_latent() for _ in range(total_t)] for _ in range(N)
        ]

        loss = cpc.compute_loss(agent_states, agent_actions, agent_future_latents)
        loss.backward()

        # Check that projector weights received gradients
        assert cpc.projector.weight.grad is not None
        # Check that predictor weights received gradients
        for predictor in cpc.predictors:
            for param in predictor.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    break

    def test_single_step_ahead(self):
        cpc = ACCPC_MA(
            hidden_dim=HIDDEN_DIM,
            latent_dim=VISUAL_DIM + SEM_DIM,
            num_steps_ahead=1,
            action_dim=ACTION_DIM,
        )

        agent_states = [
            [_make_latent() for _ in range(T)] for _ in range(N)
        ]
        agent_actions = torch.randn(B, N, T, ACTION_DIM)
        agent_future_latents = [
            [_make_latent() for _ in range(T + 1)] for _ in range(N)
        ]

        loss = cpc.compute_loss(agent_states, agent_actions, agent_future_latents)
        assert loss.shape == ()
        assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
