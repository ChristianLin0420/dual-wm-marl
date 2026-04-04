"""Unit tests for the flow matching latent predictor."""

import pytest
import torch

from m3w.interfaces import FlowPredictorOutput, LATENT_DIM
from m3w.flow_predictor import (
    FlowMatchingLoss,
    FlowPredictor,
    VelocityField,
    sinusoidal_embedding,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B = 8
HIDDEN_DIM = 512
COND_DIM = 512
DEVICE = "cpu"


@pytest.fixture
def velocity_field() -> VelocityField:
    return VelocityField(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, cond_dim=COND_DIM)


@pytest.fixture
def flow_predictor() -> FlowPredictor:
    return FlowPredictor(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, cond_dim=COND_DIM)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(batch: int = B):
    x_tau = torch.randn(batch, LATENT_DIM)
    tau = torch.rand(batch, 1)
    h = torch.randn(batch, COND_DIM)
    m = torch.randn(batch, COND_DIM)
    return x_tau, tau, h, m


# ---------------------------------------------------------------------------
# B1 — VelocityField tests
# ---------------------------------------------------------------------------

class TestVelocityField:
    def test_output_shape(self, velocity_field: VelocityField):
        x_tau, tau, h, m = _make_inputs()
        v = velocity_field(x_tau, tau, h, m)
        assert v.shape == (B, LATENT_DIM)

    def test_moe_bias_injection(self, velocity_field: VelocityField):
        x_tau, tau, h, m = _make_inputs()

        # Output without bias
        v_no_bias = velocity_field(x_tau, tau, h, m).detach().clone()

        # Set a constant bias and check it shifts the output
        bias = torch.ones(LATENT_DIM) * 3.0
        velocity_field.set_moe_bias(bias)
        v_with_bias = velocity_field(x_tau, tau, h, m).detach().clone()

        diff = v_with_bias - v_no_bias
        assert torch.allclose(diff, bias, atol=1e-5)

    def test_clear_moe_bias(self, velocity_field: VelocityField):
        x_tau, tau, h, m = _make_inputs()

        v_before = velocity_field(x_tau, tau, h, m).detach().clone()

        velocity_field.set_moe_bias(torch.ones(LATENT_DIM) * 5.0)
        velocity_field.clear_moe_bias()

        v_after = velocity_field(x_tau, tau, h, m).detach().clone()
        assert torch.allclose(v_before, v_after, atol=1e-5)


# ---------------------------------------------------------------------------
# B2 — FlowMatchingLoss tests
# ---------------------------------------------------------------------------

class TestFlowMatchingLoss:
    def test_losses_are_scalar(self, velocity_field: VelocityField):
        x_0 = torch.randn(B, LATENT_DIM)
        x_1 = torch.randn(B, LATENT_DIM)
        tau = torch.rand(B, 1)
        v_pred = torch.randn(B, LATENT_DIM)
        h = torch.randn(B, COND_DIM)
        m = torch.randn(B, COND_DIM)

        losses = FlowMatchingLoss.compute(
            v_pred, x_0, x_1, tau,
            velocity_field=velocity_field, h=h, m=m,
        )
        assert losses["flow_loss"].dim() == 0
        assert losses["shortcut_loss"].dim() == 0

    def test_flow_loss_zero_when_perfect(self):
        """If v_pred == x_1 - x_0, flow loss should be zero."""
        x_0 = torch.randn(B, LATENT_DIM)
        x_1 = torch.randn(B, LATENT_DIM)
        tau = torch.rand(B, 1)
        v_pred = x_1 - x_0

        losses = FlowMatchingLoss.compute(v_pred, x_0, x_1, tau)
        assert losses["flow_loss"].item() < 1e-6

    def test_shortcut_loss_without_velocity_field(self):
        """Shortcut loss should be zero when velocity_field is not provided."""
        x_0 = torch.randn(B, LATENT_DIM)
        x_1 = torch.randn(B, LATENT_DIM)
        tau = torch.rand(B, 1)
        v_pred = torch.randn(B, LATENT_DIM)

        losses = FlowMatchingLoss.compute(v_pred, x_0, x_1, tau)
        assert losses["shortcut_loss"].item() == 0.0


# ---------------------------------------------------------------------------
# B3 — FlowPredictor tests
# ---------------------------------------------------------------------------

class TestFlowPredictor:
    def test_forward_output_type_and_shapes(self, flow_predictor: FlowPredictor):
        x_1 = torch.randn(B, LATENT_DIM)
        h = torch.randn(B, COND_DIM)
        m = torch.randn(B, COND_DIM)

        out = flow_predictor(x_1, h, m)
        assert isinstance(out, FlowPredictorOutput)
        assert out.x_hat.shape == (B, LATENT_DIM)
        assert out.v_field.shape == (B, LATENT_DIM)

    def test_sample_single_step(self, flow_predictor: FlowPredictor):
        h = torch.randn(B, COND_DIM)
        m = torch.randn(B, COND_DIM)

        out = flow_predictor.sample(h, m, steps=1)
        assert isinstance(out, FlowPredictorOutput)
        assert out.x_hat.shape == (B, LATENT_DIM)
        assert out.v_field.shape == (B, LATENT_DIM)

    def test_sample_multi_step(self, flow_predictor: FlowPredictor):
        h = torch.randn(B, COND_DIM)
        m = torch.randn(B, COND_DIM)

        out = flow_predictor.sample(h, m, steps=4)
        assert isinstance(out, FlowPredictorOutput)
        assert out.x_hat.shape == (B, LATENT_DIM)
        assert out.v_field.shape == (B, LATENT_DIM)

    def test_velocity_field_property(self, flow_predictor: FlowPredictor):
        assert isinstance(flow_predictor.velocity_field, VelocityField)


# ---------------------------------------------------------------------------
# Gradient flow through MoE bias
# ---------------------------------------------------------------------------

class TestMoEBiasGradient:
    def test_gradient_flows_through_bias(self, flow_predictor: FlowPredictor):
        """MoE bias must participate in the computational graph so that
        gradients propagate back to the bias tensor."""
        bias = torch.randn(LATENT_DIM, requires_grad=True)
        flow_predictor.velocity_field.set_moe_bias(bias)

        x_1 = torch.randn(B, LATENT_DIM)
        h = torch.randn(B, COND_DIM)
        m = torch.randn(B, COND_DIM)

        out = flow_predictor(x_1, h, m)
        loss = out.x_hat.sum()
        loss.backward()

        assert bias.grad is not None, "Gradient did not flow through the MoE bias"
        assert bias.grad.shape == (LATENT_DIM,)


# ---------------------------------------------------------------------------
# Sinusoidal embedding
# ---------------------------------------------------------------------------

class TestSinusoidalEmbedding:
    def test_shape(self):
        tau = torch.rand(B, 1)
        emb = sinusoidal_embedding(tau, embed_dim=64)
        assert emb.shape == (B, 64)

    def test_deterministic(self):
        tau = torch.tensor([[0.5]])
        e1 = sinusoidal_embedding(tau, 64)
        e2 = sinusoidal_embedding(tau, 64)
        assert torch.allclose(e1, e2)
