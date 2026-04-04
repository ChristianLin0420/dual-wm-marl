# EDELINE-MARL Implementation Notes

## Cross-Agent Coordination Notes

### Agent B ↔ Agent D: MoE Velocity Bias Injection
- `SoftMoEVelocityBias.compute_bias(h)` returns `(B, LATENT_DIM)` tensor
- Injected via `FlowPredictor.velocity_field.set_moe_bias(bias)` before forward
- Bias MUST remain part of the computational graph (no `.detach()`)
- Bias is computed ONCE per step (at τ=0) and added to all ODE steps

## File Ownership
- Agent A: `m3w/encoders.py`
- Agent B: `m3w/flow_predictor.py`
- Agent C: `m3w/sequence_model.py`
- Agent D: `m3w/moe.py`
- Agent E: `m3w/world_model.py`
- Agent F: `m3w/planner.py`, `examples/train.py`, `configs/`
- Shared: `m3w/interfaces.py`

## Design Deviations
(To be filled in during implementation)
