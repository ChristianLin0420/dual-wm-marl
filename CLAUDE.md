# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This repository contains two systems:

1. **M3W** (base) — "Learning and Planning Multi-Agent Tasks via an MoE-based World Model" (NeurIPS 2025). MoE world model for multi-task MARL.
2. **EDELINE-MARL** (extension) — Fuses dual-latent representation (Categorical VAE + Barlow Twins), flow matching latent predictor with shortcut forcing, and cross-agent Transformer SSM into M3W's MoE framework.

## Setup

```bash
conda create -n m3w python=3.8 -y
conda activate m3w
pip install -r requirements.txt
pip install einops pytest  # additional deps for EDELINE-MARL
pip install -e .           # install m3w package in dev mode
```

MuJoCo setup (required for MuJoCo environments):
```bash
# Download MuJoCo 210
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz && rm mujoco210-linux-x86_64.tar.gz

# Add to .bashrc
echo 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH' >> ~/.bashrc

# System deps (Ubuntu)
sudo apt-get install -y libx11-dev libgl1-mesa-dev libosmesa6-dev libglew-dev patchelf

# mujoco_py requires Cython<3
pip install "cython<3" mujoco-py
```

External dependencies that may require manual installation: [MA-MuJoCo](https://github.com/schroederdewitt/multiagent_mujoco), [Isaac Gym](https://developer.nvidia.com/isaac-gym), [Bi-DexHands](https://github.com/PKU-MARL/DexterousHands).

## Training

Both training scripts require MuJoCo's `LD_LIBRARY_PATH` (add to `.bashrc` or prefix each command):
```bash
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
```

### M3W (original baseline)
```bash
python examples/train.py --load-config configs/mujoco/installtest/m3w/config.json
```
### EDELINE-MARL (multi-task MuJoCo)
```bash
python examples/train_edeline.py --load-config configs/edeline_marl/mujoco/config.json
```

Both log to the same wandb project `wm-marl` with descriptive run names:
- M3W: `m3w_mujoco_2_cheetah_2ag_2task_s1`
- EDELINE: `edeline_mujoco_2_cheetah_2ag_2task_s1`

EDELINE-MARL logs detailed metrics: all 7 loss components, gradient norms per module, reward accuracy, eval returns with videos. M3W logs basic training metrics. See the Wandb Logging section below.

### Config Structure

Configs are JSON files with three sections:
- `main_args` — algo (`"m3w"` or `"edeline_marl"`), env, experiment name
- `algo_args` — model/training hyperparameters, world model config, planner config
- `env_args` — agent count, task count, episode limit, environment specification

EDELINE-MARL adds these keys under `algo_args.world_model`:
- `num_cats`, `cat_dim` — Categorical VAE dimensions (visual latent = num_cats * cat_dim)
- `sem_dim` — Barlow Twins semantic latent dimension (total latent = visual + sem)
- `hidden_dim` — Transformer SSM and conditioning dimension
- `flow_steps_train` (8), `flow_steps_infer` (1) — ODE steps (1 = shortcut mode)
- `shortcut_lambda` (0.1) — weight for shortcut forcing loss
- `num_experts`, `num_slots` — MoE configuration
- `num_reward_bins` — distributional reward prediction bins
- `loss_weights` — per-component loss weights: `pred`, `rec`, `rep`, `dyn`, `cpc`, `latent`, `flow`

Dimension sizing guideline: scale dimensions to observation complexity. For MuJoCo scalar obs (dim ~20), use small values (num_cats=8, cat_dim=8, sem_dim=64, hidden_dim=128). The default config at `configs/edeline_marl/mujoco/config.json` is tuned for 2-agent HalfCheetah (obs_dim=21, action_dim=3).

## Tests

```bash
python -m pytest tests/ -v                         # all tests (84)
python -m pytest tests/test_encoders.py -v         # dual encoder only
python -m pytest tests/test_flow_predictor.py -v   # flow matching only
python -m pytest tests/test_sequence_model.py -v   # Transformer SSM + AC-CPC
python -m pytest tests/test_moe.py -v              # MoE velocity bias + reward
python -m pytest tests/test_world_model.py -v      # orchestrator + losses
python -m pytest tests/test_planner.py -v          # MPPI planner
```

## Architecture

### Environment Observations

MuJoCo multi-agent environments use **scalar state observations** (not images):
- 2-agent HalfCheetah: obs_dim=21 (joint positions + velocities + agent ID), action_dim=3 per agent
- Observations are continuous floats, normalized in the env wrapper

### M3W (Original) Data Flow

```
Env observations → MLPEncoder (per agent) → latent [B, 128]
  → SoftMoE Dynamics → predicted next latent
  → SparseMoE Reward → reward logits (TwoHot distributional)
  → MPPI Planner → action sequences
  → Actor (Gaussian + tanh) → actions
  → Dual Q-network Critic → Q-values
```

### EDELINE-MARL Framework Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EDELINE-MARL  World Model                              │
│                                                                                 │
│  ┌──────────── Per Agent i (shared weights) ──────────────────────────────────┐ │
│  │                                                                            │ │
│  │   obs_t^i ──┬──► CategoricalVAE ──► z_t^i [B, 8, 8]  (Gumbel-Softmax)   │ │
│  │   [B, 21]   │    (encoder.py)        visual latent                        │ │
│  │             │                                                              │ │
│  │             └──► SemanticEncoder ──► d_t^i [B, 64]     (L2-normalized)    │ │
│  │                  (encoder.py)        semantic latent                       │ │
│  │                                                                            │ │
│  │   x_t^i = [z_flat; d] ─────────────────────────── latent_dim = 128       │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                              │                                                  │
│                              ▼                                                  │
│  ┌─────────────── Transformer SSM-MA (sequence_model.py) ───────────────────┐  │
│  │                                                                           │  │
│  │  For each agent i:                                                        │  │
│  │    ProjectDualLatent([z_flat; d]) ──► proj_i [B, 128]                    │  │
│  │    TransformerSSM(proj_i + action_emb) ──► h_t^i [B, 128]               │  │
│  │                          │                                                │  │
│  │  Cross-Agent Attention:  │                                                │  │
│  │    query = h_t^i         ▼                                                │  │
│  │    keys  = {h_t^j : j≠i} ──► m_t^i [B, 128]  (teammate message)         │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                      │ h_t^i              │ m_t^i                               │
│                      ▼                    ▼                                     │
│  ┌──── MoE Velocity Bias (moe.py) ────┐  │                                    │
│  │  SoftMoE dispatch/combine on h_t^i  │  │                                    │
│  │  Expert MLPs ──► bias [B, 128]      │  │                                    │
│  └──────────────┬──────────────────────┘  │                                    │
│                 │ velocity_bias           │                                     │
│                 ▼                         ▼                                     │
│  ┌──── Flow Matching Latent Predictor (flow_predictor.py) ──────────────────┐  │
│  │                                                                           │  │
│  │  Training:                                                                │  │
│  │    x_0 ~ N(0, I)           sample noise                                  │  │
│  │    τ  ~ U(0, 1)            sample time                                   │  │
│  │    x_τ = (1-τ)x_0 + τx_1  linear interpolant                            │  │
│  │    v = VelocityField(x_τ, τ, h, m) + velocity_bias                      │  │
│  │    L_flow = MSE(v, x_1 - x_0)                                           │  │
│  │                                                                           │  │
│  │  Inference (single-step shortcut):                                        │  │
│  │    x_0 ~ N(0, I)                                                         │  │
│  │    x̂_1 = x_0 + VelocityField(x_0, 0, h, m) + velocity_bias             │  │
│  │                                                                           │  │
│  └──────────────┬────────────────────────────────────────────────────────────┘  │
│                 │ x̂_t^i [B, 128]                                               │
│                 ▼                                                               │
│  ┌──── State Aggregation ──────────────────────────────────────────────────┐   │
│  │  s_t = concat over agents: [z_flat^1; d^1; h^1; z_flat^2; d^2; h^2]   │   │
│  │        shape: [B, N * (latent_dim + hidden_dim)]                        │   │
│  └──────────────┬──────────────────────────┬───────────────────────────────┘   │
│                 │                          │                                    │
│                 ▼                          ▼                                    │
│  ┌── SparseMoE Reward (moe.py) ──┐  ┌── ContinuePredictor ──┐                │
│  │  NoisyTopK Router + SelfAttn   │  │  MLP ──► p(continue)  │                │
│  │  Experts ──► r̂_t [B, bins]     │  │          [B, 1]       │                │
│  └────────────────────────────────┘  └────────────────────────┘                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

                              │ predicted latents, rewards
                              ▼
┌─────────────────── MPPI Planner (planner.py) ─────────────────────────────────┐
│                                                                                │
│  For K iterations:                                                             │
│    Sample S action trajectories over horizon H                                 │
│    For each trajectory, roll out world model in imagination:                   │
│      single-step flow sampling ──► next latent ──► reward                     │
│    Accumulate discounted returns R^k                                           │
│    Select top-E elites, refit mean/std                                         │
│  Return first action from best trajectory                                      │
│                                                                                │
└───────────────────────────────┬────────────────────────────────────────────────┘
                                │ actions [N, action_dim]
                                ▼
┌───────── Actor-Critic (algorithms/) ──────────────────────────────────────────┐
│  Actor:  π(a|h) = Gaussian + tanh squashing   (per agent, trained on Q)      │
│  Critic: Q(s,a) = Dual distributional Q-nets   (shared, with target nets)    │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Training Pseudocode

```python
# ============================================================
# EDELINE-MARL Training Loop  (examples/train_edeline.py)
# ============================================================

# --- Initialization ---
encoder     = DualEncoder(obs_dim, num_cats, cat_dim, sem_dim, hidden_dim)
ssm         = TransformerSSM_MA(hidden_dim, action_dim, num_agents)
flow_pred   = FlowPredictor(latent_dim, hidden_dim)
soft_moe    = SoftMoEVelocityBias(hidden_dim, latent_dim, num_experts)
reward_head = SparseMoEReward(num_agents, hidden_dim, latent_dim)
continue_pred = ContinuePredictor(num_agents * (latent_dim + hidden_dim))
actors      = [WorldModelActor(hidden_dim, action_dim) for _ in num_agents]
critic      = WorldModelCritic(hidden_dim * num_agents, action_spaces)
buffer      = ReplayBuffer(buffer_size)
planner     = MPPIPlanner(horizon, num_samples, num_elites, temperature)

# --- Phase 1: Warmup (collect random data + pretrain world model) ---
for step in range(warmup_steps):
    actions = sample_random_actions()
    obs, rewards, dones = envs.step(actions)
    buffer.insert(obs, actions, rewards, dones)

for step in range(warmup_train_steps):
    batch = buffer.sample(batch_size)
    world_model_update(batch)  # same as main loop model update below

# --- Phase 2: Main Training Loop ---
for step in range(num_env_steps):

    # ---- 1. Plan actions via MPPI over learned world model ----
    with torch.no_grad():
        for agent_i in range(N):
            latent_i = encoder(obs[:, i])       # AgentLatent(z, d, h=None)
        latents, messages = ssm(latents, prev_actions)  # fill h, compute m

        actions = planner.plan(
            world_model, latents, messages, actors
            # internally rolls out H steps:
            #   for t in horizon:
            #     bias = soft_moe.compute_bias(h)
            #     next_latent = flow_pred.sample(h, m)   # single-step shortcut
            #     reward = reward_head(concat [z; d; h])
            #   select best trajectory via CEM elite selection
        )

    # ---- 2. Environment step & buffer insert ----
    next_obs, rewards, dones = envs.step(actions)
    buffer.insert(obs, actions, rewards, dones, next_obs)
    obs = next_obs

    # ---- 3. World model + critic update ----
    batch = buffer.sample_horizon(batch_size, horizon)
    # batch contains: obs[N,H,B,D], actions[N,H,B,A], rewards[H,B,1], next_obs[N,H,B,D]

    # -- Critic target (no grad) --
    with torch.no_grad():
        next_latents = [encoder(nstep_next_obs[i]) for i in range(N)]
        next_latents, _ = ssm(next_latents, dummy_actions)
        next_actions = [actor_i(next_h_i) for i in range(N)]
        next_q = critic.target(concat_h, concat_a)
        q_target = nstep_reward + gamma * next_q * (1 - done)

    # -- Forward through horizon --
    for t in range(horizon):
        # Encode current observations
        latents_t = [encoder(obs[i, t]) for i in range(N)]

        # Reconstruction loss (VAE)
        L_rec += encoder.reconstruction_loss(obs[i, t])

        # Barlow Twins loss (semantic encoder)
        L_latent += encoder.barlow_twins_loss(obs[i, t])

        # Sequence model: get h and cross-agent messages
        latents_h, messages = ssm(latents_t, actions[:, t])

        # Flow matching loss (per agent)
        for i in range(N):
            x_1_target = concat(next_latent[i].z_flat, next_latent[i].d)
            bias = soft_moe.compute_bias(latents_h[i].h)
            flow_pred.velocity_field.set_moe_bias(bias)
            flow_out = flow_pred(x_1_target, h=latents_h[i].h, m=messages[i])

            # L_flow = E[||v_pred - (x_1 - x_0)||^2]
            # L_shortcut = lambda * ||x_0 + v(x_0, 0, h, m) - x_1||^2
            L_flow += flow_matching_loss(flow_out, x_1_target)

        # Reward prediction loss
        s_t = concat([z_flat_i; d_i; h_i] for all agents)
        r_logits = reward_head(s_t)
        L_reward += two_hot_loss(r_logits, rewards[t])

        # Q-function loss
        q1, q2 = critic(joint_h, joint_actions_t)
        L_q += (two_hot_loss(q1, q_target) + two_hot_loss(q2, q_target)) / 2

    # -- Aggregate and backprop --
    L_total = (w_rec * L_rec + w_latent * L_latent + w_flow * L_flow
               + w_reward * L_reward + w_q * L_q + w_balance * L_balance)
    optimizer.zero_grad()
    L_total.backward()
    clip_grad_norm(all_params, grad_clip_norm)
    optimizer.step()
    critic.soft_update_target()

    # ---- 4. Actor update (every policy_freq steps) ----
    if step % policy_freq == 0:
        with torch.no_grad():
            latents = [encoder(obs_batch[i]) for i in range(N)]
            latents_h, _ = ssm(latents, dummy_actions)
            h_per_agent = [latents_h[i].h for i in range(N)]

        for agent_i in shuffled(range(N)):
            actions_i, log_prob_i = actor_i(h_per_agent[i])
            q_value = critic(concat_h, concat_actions)
            actor_loss = (entropy_coef * log_prob_i - q_value).mean()
            actor_i.optimizer.step(actor_loss)

    # ---- 5. Logging & Evaluation ----
    if step % log_interval == 0:
        log_to_wandb(L_rec, L_latent, L_flow, L_reward, L_q,
                     grad_norms, reward_acc, vae_temperature, ...)

    if step % eval_interval == 0:
        eval_returns = run_eval_episodes(planner, encoder, ssm)
        log_to_wandb(eval_avg_rew, eval_video, ...)
```

### Key Modules — M3W (original, unchanged)

- **`m3w/models/world_models.py`** — Base MLP/MoE components: `MLPEncoder`, `NoisyTopKRouter`, `SelfAttnExpert`, `CenMoEDynamicsModel`, `CenMoERewardModel`, `TwoHotProcessor`.
- **`m3w/runners/world_model_runner.py`** — Original training loop with MPPI planning.
- **`m3w/algorithms/actors/world_model_actor.py`** — `WorldModelPolicy` + `WorldModelActor`.
- **`m3w/algorithms/critics/world_model_critic.py`** — `DisRegQNet`, dual Q-networks with target networks.
- **`m3w/common/buffers/world_model_buffer.py`** — Off-policy replay buffer with multi-step returns.
- **`m3w/envs/`** — `MultitaskMujoco` and `MultitaskDexHands` wrappers.

### Key Modules — EDELINE-MARL (new)

- **`m3w/interfaces.py`** — Shared dataclasses: `AgentLatent` (z, d, h), `FlowPredictorOutput`, `WorldModelOutput`. Default constants: `LATENT_DIM=1536`, `VISUAL_LATENT_DIM=1024`, `SEM_LATENT_DIM=512` (actual dims depend on config).
- **`m3w/encoders.py`** — `CategoricalVAE` (Gumbel-Softmax with temperature annealing), `SemanticEncoder` (Barlow Twins with dropout augmentation), `DualEncoder` (unified forward → `AgentLatent`). Parameter-shared across agents.
- **`m3w/flow_predictor.py`** — `VelocityField` (3-layer MLP with sinusoidal time embedding + MoE bias hook), `FlowPredictor` (training: random τ interpolant; inference: single-step shortcut or multi-step Euler), `FlowMatchingLoss`.
- **`m3w/sequence_model.py`** — `TransformerSSM` (causal self-attention per agent), `CrossAgentAttention` (agent i attends over teammates' h), `TransformerSSM_MA` (orchestrates per-agent SSM + cross-agent messages), `ACCPC_MA` (multi-agent AC-CPC with cross-agent InfoNCE).
- **`m3w/moe.py`** — `SoftMoEVelocityBias` (SoftMoE dispatch/combine → velocity bias for flow ODE), `SparseMoEReward` (NoisyTopK gating on dual latent state), `ContinuePredictor` (binary MLP).
- **`m3w/world_model.py`** — `WorldModel` orchestrator: composes all above into unified forward pass + 7-term loss (`L_pred`, `L_rec`, `L_rep`, `L_dyn`, `L_cpc`, `L_latent`, `L_flow`).
- **`m3w/planner.py`** — `MPPIPlanner`: trajectory optimization over dual latent with single-step flow sampling, policy trajectory injection, CEM elite selection.
- **`examples/train_edeline.py`** — Training entry point with `EdelineRunner`.

### MoE Design

**M3W**: SoftMoE directly outputs predicted next latent; SparseMoE predicts reward.

**EDELINE-MARL**: SoftMoE is repositioned to output a **velocity bias** that steers the flow matching ODE toward task-relevant latent regions. This separates joint distribution modeling (flow matching) from task specialization (MoE). SparseMoE reward now conditions on the full dual-latent state [z; d; h] per agent.

### Training Losses (EDELINE-MARL)

| Loss | Description | Default Weight |
|------|-------------|----------------|
| `L_pred` | Reward (TwoHot) + continue (BCE) | 1.0 |
| `L_rec` | VAE reconstruction (MSE) | 1.0 |
| `L_rep` | KL regularization (uniform categorical prior) | 0.1 |
| `L_dyn` | Linear dynamics predictor guidance (MSE, target detached) | 1.0 |
| `L_cpc` | Multi-agent AC-CPC (cross-agent InfoNCE) | 0.1 |
| `L_latent` | Barlow Twins semantic loss | 1.0 |
| `L_flow` | Flow matching velocity field (MSE) + shortcut forcing | 1.0 |

### Training Phases

1. **Warmup**: Collect random experience, pre-train world model on collected data
2. **Main loop**: MPPI planning (single-step flow sampling) → collect experience → update world model (all 7 losses) → update critic → update actor

## Wandb Logging

Project: `wm-marl`. Run names include algo, env, and env-specific name:
- M3W: `m3w_{env}_{envname}_{n}ag_{n}task_s{seed}`
- EDELINE: `edeline_{env}_{envname}_{n}ag_{n}task_s{seed}`

### Training Metrics (`train_info/`)

| Metric | Meaning | Interpretation |
|--------|---------|----------------|
| `L_total` | Weighted sum of all losses | Should decrease over training |
| `L_rec` | VAE reconstruction (MSE) | Visual encoder fidelity |
| `L_latent` | Barlow Twins loss | Semantic encoder quality, decreases toward 0 |
| `L_flow` | Flow matching + shortcut loss | World model dynamics accuracy |
| `L_reward` | TwoHot reward prediction | Reward model accuracy |
| `L_q` | Dual Q-network loss | Critic quality |
| `L_balance` | MoE load balancing | Expert utilization, negative is normal |
| `L_*_w` | Weighted versions of above | Actual contribution to total loss |
| `reward_acc` | Reward predictions within 0.05 of true | Should increase toward 1.0 |
| `reward_err` | Mean absolute reward error | Should decrease |
| `vae_temperature` | Gumbel-Softmax temperature | Anneals 1.0 → 0.1 |
| `lr` | Learning rate | Constant unless decay enabled |

### Gradient Norms (`train_info/grad_norm/`)

`encoder`, `ssm`, `flow`, `soft_moe`, `reward_head`, `critic` — monitor for exploding/vanishing gradients.

### Actor (`train_info/`)

| Metric | Meaning |
|--------|---------|
| `actor_loss_agent{i}` | Per-agent policy loss (entropy-regularized) |
| `pi_scale` | Running Q-value scale for policy gradient |

### Rollout (`rollout_info/`)

| Metric | Meaning |
|--------|---------|
| `rew_buffer` | Mean reward in replay buffer |
| `r_{task_id}` | Per-task mean episode return |

### Evaluation (`eval_info/`)

| Metric | Meaning |
|--------|---------|
| `eval_avg_rew` | Mean eval episode return |
| `eval_std_rew` | Std of eval returns |
| `eval_min_rew` / `eval_max_rew` | Min/max eval return |
| `eval_avg_len` | Mean episode length |
| `eval/video` | Rendered episode video (if env supports render) |

## Known Compatibility Notes

- **gym 0.26**: The MuJoCo wrappers in `m3w/envs/mujoco/multiagent_mujoco/mujoco_multi.py` have been patched for gym 0.26 (5-value step return, removed `.seed()`, private attribute access via `.unwrapped`).
- **Cython**: mujoco_py requires `cython<3` (Cython 3.x has incompatible callback signatures).
- **Python 3.8**: All EDELINE-MARL code uses `typing.List`/`Dict`/`Optional` (not `list[...]`/`dict[...]`).
