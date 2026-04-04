"""
EDELINE-MARL training entry point.

Uses the dual-latent WorldModel with flow matching, MoE modules, and
Transformer SSM, orchestrated by the MPPI planner.  Follows the same
overall structure as the original M3W WorldModelRunner but operates on
the new EDELINE components.
"""

import argparse
import datetime
import itertools
import json
import os
import time

import numpy as np
import torch
import setproctitle
import wandb

from m3w.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    set_seed,
    get_num_agents,
    check,
)
from m3w.utils.models_tools import init_device
from m3w.utils.configs_tools import save_config, get_task_name

from m3w.algorithms.actors.world_model_actor import WorldModelActor
from m3w.algorithms.critics.world_model_critic import WorldModelCritic
from m3w.common.buffers.world_model_buffer import WorldModelBuffer

from m3w.encoders import DualEncoder
from m3w.sequence_model import TransformerSSM_MA
from m3w.flow_predictor import FlowPredictor, FlowMatchingLoss
from m3w.moe import SoftMoEVelocityBias, SparseMoEReward, ContinuePredictor
from m3w.planner import MPPIPlanner
from m3w.interfaces import LATENT_DIM  # noqa: F401 (kept for reference; actual dims from config)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def convert_num(num):
    suffixes = ["K", "M", "G", "T"]
    for suffix in suffixes:
        num /= 1000
        if num < 1000:
            return f"{num:.1f}{suffix}"
    return f"{num:.3f}T"


class _WorldModelShim:
    """Lightweight shim that groups EDELINE sub-components under one namespace.

    Used when the full ``m3w.world_model.WorldModel`` is not yet available.
    The planner expects ``world_model.flow_predictor``, ``world_model.soft_moe``,
    ``world_model.reward_head``, and ``world_model.reward_processor``.
    """

    def __init__(self, flow_predictor, soft_moe, reward_head,
                 reward_processor, sequence_model, encoder, continue_pred):
        self.flow_predictor = flow_predictor
        self.soft_moe = soft_moe
        self.reward_head = reward_head
        self.reward_processor = reward_processor
        self.sequence_model = sequence_model
        self.encoder = encoder
        self.continue_pred = continue_pred


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EdelineRunner:
    """Training runner for EDELINE-MARL."""

    def __init__(self, args, algo_args, env_args):
        assert env_args.get("state_type", "EP") == "EP"
        assert algo_args["render"]["use_render"] is False

        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        # Shortcuts to frequently used config groups
        self.state_type = env_args.get("state_type", "EP")
        self.fixed_order = algo_args["algo"]["fixed_order"]
        self.policy_freq = algo_args["algo"].get("policy_freq", 1)
        self.horizon = algo_args["plan"]["horizon"]
        self.entropy_coef = algo_args["model"]["entropy_coef"]
        self.step_rho = algo_args["world_model"]["step_rho"]

        wm_cfg = algo_args["world_model"]
        self.num_cats = wm_cfg.get("num_cats", 32)
        self.cat_dim = wm_cfg.get("cat_dim", 32)
        self.sem_dim = wm_cfg.get("sem_dim", 512)
        self.hidden_dim = wm_cfg.get("hidden_dim", 512)

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.run_dir, self.save_dir, self.log_file, self.task_name, self.expt_name = (
            self._init_config()
        )

        if algo_args["logger"].get("wandb", False):
            env_names = "_".join(env_args.get("envs", {}).keys())
            run_name = (
                f"edeline_{args['env']}_{env_names}"
                f"_{env_args.get('n_agents', 2)}ag"
                f"_{env_args.get('n_tasks', 2)}task"
                f"_s{algo_args['seed']['seed']}"
            )
            wandb.init(
                project="wm-marl",
                group=args["algo"],
                name=run_name,
                config={
                    "main_args": args,
                    "algo_args": algo_args,
                    "env_args": env_args,
                },
            )
        setproctitle.setproctitle("edeline-marl-train")

        # ---- Environments ----
        assert algo_args["train"]["n_rollout_threads"] % env_args["n_tasks"] == 0
        num_env_per_task = algo_args["train"]["n_rollout_threads"] // env_args["n_tasks"]
        self.task_idxes = list(
            np.arange(env_args["n_tasks"]).repeat(num_env_per_task)
        )

        self.envs = make_train_env(
            args["env"],
            algo_args["seed"]["seed"],
            algo_args["train"]["n_rollout_threads"],
            env_args,
        )
        self.envs.meta_reset_task(self.task_idxes)

        if algo_args["eval"]["use_eval"]:
            self.eval_envs = make_eval_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["eval"]["n_eval_rollout_threads"],
                env_args,
            )
            n_eval = algo_args["eval"]["n_eval_rollout_threads"]
            eval_per_task = n_eval // env_args["n_tasks"]
            eval_task_idxes = list(np.arange(env_args["n_tasks"]).repeat(eval_per_task))
            self.eval_envs.meta_reset_task(eval_task_idxes)
        else:
            self.eval_envs = None

        self.num_tasks = env_args["n_tasks"]
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.agent_deaths = np.zeros(
            (algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
        )
        self.action_spaces = self.envs.action_space
        self.action_dims = []
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)
            self.action_dims.append(self.action_spaces[agent_id].shape[0])
        print("init env, done")

        # ---- EDELINE Components ----
        obs_dim = self.envs.observation_space[0].shape[0]

        self.dual_encoder = DualEncoder(
            obs_dim=obs_dim,
            num_cats=self.num_cats,
            cat_dim=self.cat_dim,
            sem_dim=self.sem_dim,
            hidden_dim=self.hidden_dim,
            tau_start=wm_cfg.get("gumbel_tau_start", 1.0),
            tau_end=wm_cfg.get("gumbel_tau_end", 0.1),
            anneal_steps=wm_cfg.get("gumbel_anneal_steps", 100000),
            bt_lambda=wm_cfg.get("barlow_twins_lambda", 0.005),
            aug_drop_p=wm_cfg.get("aug_dropout", 0.1),
        ).to(self.device)

        self.visual_latent_dim = self.num_cats * self.cat_dim
        self.latent_dim = self.visual_latent_dim + self.sem_dim
        self.grad_clip_norm = wm_cfg.get("grad_clip_norm", 20.0)

        self.sequence_model = TransformerSSM_MA(
            hidden_dim=self.hidden_dim,
            num_heads=wm_cfg.get("ssm_num_heads", 8),
            num_layers=wm_cfg.get("ssm_num_layers", 3),
            action_dim=self.action_dims[0],
            num_agents=self.num_agents,
            cross_num_heads=wm_cfg.get("ssm_cross_num_heads", 4),
            dropout=wm_cfg.get("ssm_dropout", 0.1),
            visual_dim=self.visual_latent_dim,
            sem_dim=self.sem_dim,
        ).to(self.device)

        self.flow_predictor = FlowPredictor(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            cond_dim=self.hidden_dim,
            tau_embed_dim=wm_cfg.get("flow_tau_embed_dim", 64),
        ).to(self.device)

        self.soft_moe = SoftMoEVelocityBias(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_experts=wm_cfg.get("num_experts", 8),
            num_slots=wm_cfg.get("num_slots", 1),
            expert_hidden_dim=wm_cfg.get("expert_hidden_dim", 256),
        ).to(self.device)

        per_agent_dim = self.latent_dim + self.hidden_dim
        self.reward_head = SparseMoEReward(
            n_agents=self.num_agents,
            hidden_dim=self.hidden_dim,
            num_experts=wm_cfg.get("num_reward_experts", 16),
            k=wm_cfg.get("top_k", 2),
            num_reward_bins=wm_cfg.get("num_reward_bins", 51),
            latent_dim=self.latent_dim,
            n_heads=wm_cfg.get("reward_expert_n_heads", 1),
            expert_ffn_hidden=wm_cfg.get("reward_expert_ffn_hidden", 512),
            expert_dropout=wm_cfg.get("reward_expert_dropout", 0.0),
            head_hidden=wm_cfg.get("reward_head_hidden", 256),
        ).to(self.device)

        continue_input_dim = self.num_agents * per_agent_dim
        self.continue_pred = ContinuePredictor(
            input_dim=continue_input_dim,
            hidden_dim=wm_cfg.get("continue_hidden_dim", 128),
        ).to(self.device)

        import m3w.models.world_models as wms
        self.reward_processor = wms.TwoHotProcessor(
            num_bins=wm_cfg.get("num_reward_bins", wm_cfg["num_bins"]),
            vmin=wm_cfg["reward_min"],
            vmax=wm_cfg["reward_max"],
            device=self.device,
        )

        # Build world model namespace for planner
        self.world_model = _WorldModelShim(
            flow_predictor=self.flow_predictor,
            soft_moe=self.soft_moe,
            reward_head=self.reward_head,
            reward_processor=self.reward_processor,
            sequence_model=self.sequence_model,
            encoder=self.dual_encoder,
            continue_pred=self.continue_pred,
        )

        print("init EDELINE components, done")

        # ---- Actor / Critic / Buffer ----
        from gym.spaces import Box
        latent_space = Box(
            low=-10, high=10,
            shape=(self.hidden_dim,),
            dtype=np.float32,
        )
        joint_latent_space = Box(
            low=-10, high=10,
            shape=(self.hidden_dim * self.num_agents,),
            dtype=np.float32,
        )

        self.actor = []
        for agent_id in range(self.num_agents):
            agent = WorldModelActor(
                {**algo_args["model"], **algo_args["algo"]},
                latent_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.actor.append(agent)

        self.critic = WorldModelCritic(
            args={**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
            share_obs_space=joint_latent_space,
            act_space=self.envs.action_space,
            num_agents=self.num_agents,
            state_type=self.state_type,
            device=self.device,
            wm_args=algo_args["world_model"],
        )

        self.buffer = WorldModelBuffer(
            {**algo_args["train"], **algo_args["model"], **algo_args["algo"], **env_args},
            self.envs.share_observation_space[0],
            self.num_agents,
            self.envs.observation_space,
            self.envs.action_space,
        )
        print("init actor, critic, buffer, done")

        # ---- Planner ----
        plan_cfg = {
            **algo_args["plan"],
            "gamma": algo_args["algo"]["gamma"],
            "flow_steps_infer": wm_cfg.get("flow_steps_infer", 1),
        }
        self.planner = MPPIPlanner(plan_cfg)

        # ---- Optimiser ----
        wm_params = [
            {"params": self.dual_encoder.parameters(),
             "lr": algo_args["model"]["lr"] * wm_cfg["enc_lr_scale"]},
            {"params": self.sequence_model.parameters()},
            {"params": self.flow_predictor.parameters()},
            {"params": self.soft_moe.parameters()},
            {"params": self.reward_head.parameters()},
            {"params": self.continue_pred.parameters()},
            {"params": itertools.chain(
                self.critic.critic.parameters(),
                self.critic.critic2.parameters(),
            ), "lr": algo_args["model"]["lr"]},
        ]
        self.model_optimizer = torch.optim.Adam(
            params=wm_params,
            lr=algo_args["model"]["lr"],
        )

        # ---- Training state ----
        self.total_it = 0
        self._start_time = self._check_time = time.time()
        self.train_episode_rewards = None
        self.done_episodes_rewards = None
        self.t0 = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        self._start_time = self._check_time = time.time()
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        steps = (
            self.algo_args["train"]["num_env_steps"] // n_threads
        )
        update_num = int(
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )

        self.train_episode_rewards = np.zeros(n_threads)
        self.done_episodes_rewards = [[] for _ in range(self.num_tasks)]
        self.t0 = [True] * n_threads

        # Warmup
        if self.algo_args["train"]["model_dir"] is not None:
            obs, share_obs, available_actions = self.envs.reset()
        else:
            print("start warmup")
            obs, share_obs, _ = self._warmup(
                warmup_train=self.algo_args["world_model"]["warmup_train"],
                train_steps=self.algo_args["world_model"]["wt_steps"],
            )
            print("finish warmup, start training")

        train_info = {}
        for step in range(1, steps + 1):
            # Collect with MPPI planner
            actions = self._plan_actions(obs, self.t0, add_random=True)
            (new_obs, new_share_obs, rewards, dones, infos, _) = self.envs.step(actions)

            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                None,
            )
            self._insert(data)
            obs = new_obs
            share_obs = new_share_obs

            # Train
            if step % self.algo_args["train"]["train_interval"] == 0:
                if self.algo_args["train"]["use_linear_lr_decay"]:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(step, steps)
                    self.critic.lr_decay(step, steps)

                for _ in range(update_num):
                    _info = self._train()
                    for k, v in _info.items():
                        train_info.setdefault(k, []).append(v)

            # Log
            if step % self.algo_args["train"]["log_interval"] == 0:
                for k in train_info:
                    train_info[k] = np.mean(train_info[k])
                rollout_info = {"rew_buffer": self.buffer.get_mean_rewards()}
                for task_id in range(self.num_tasks):
                    if self.done_episodes_rewards[task_id]:
                        rollout_info[f"r_{task_id}"] = np.mean(
                            self.done_episodes_rewards[task_id]
                        )
                        self.done_episodes_rewards[task_id] = []
                self._console_log(step, train_info=train_info, rollout_info=rollout_info)
                train_info = {}

            # Eval
            if step % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    print(f"\nEvaluation at step {step} / {steps}:")
                    self._eval(step)

            # Save
            if (
                step % self.algo_args["train"]["save_interval"] == 0
                and self.algo_args["logger"]["save_model"]
            ):
                self._save()
                print("\nModel has been saved\n")

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _plan_actions(self, obs, t0, add_random=True):
        """Use MPPI planner over dual-latent imagination."""
        n_threads = obs.shape[0]

        # Encode observations -> AgentLatent (h is None at this point)
        latents = []
        for i in range(self.num_agents):
            obs_i = check(obs[:, i]).to(**self.tpdv)
            latent_i = self.dual_encoder(obs_i)
            latents.append(latent_i)

        # Run sequence model to get h and messages
        # Use zero actions for initial conditioning
        dummy_actions = torch.zeros(
            n_threads, self.num_agents, self.action_dims[0], **self.tpdv
        )
        latents_with_h, messages = self.sequence_model(latents, dummy_actions)

        actions = self.planner.plan(
            world_model=self.world_model,
            latents=latents_with_h,
            messages=messages,
            actors=self.actor,
            action_dims=self.action_dims,
            t0=t0,
            eval_mode=not add_random,
            critic=self.critic,
        )
        return actions

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self):
        """One training step: world model + actor + critic."""
        if self.buffer.cur_size < self.buffer.batch_size:
            return {}
        self.total_it += 1

        t0 = torch.randperm(self.buffer.cur_size).numpy()[: self.buffer.batch_size]
        self.buffer.update_end_flag()
        indices = [t0]
        for _ in range(self.horizon - 1):
            indices.append(self.buffer.next(indices[-1]))

        data_horizon = self.buffer.sample_horizon(horizon=self.horizon, t0=t0)
        (_, sp_obs, sp_actions, _, sp_reward, _, _, sp_term, _, sp_next_obs, _, sp_gamma) = data_horizon

        sp_nstep_reward = np.zeros_like(sp_reward)
        sp_nstep_term = np.zeros_like(sp_term)
        sp_nstep_next_obs = np.zeros_like(sp_next_obs)
        sp_nstep_gamma = np.zeros_like(sp_gamma)
        for t, indice in enumerate(indices):
            data_nstep = self.buffer.sample(indice=indice)
            (_, _, _, _, nstep_reward, _, _, nstep_term, _, nstep_next_obs, _, nstep_gamma, _, _) = data_nstep
            sp_nstep_reward[t] = nstep_reward
            sp_nstep_term[t] = nstep_term
            sp_nstep_next_obs[:, t] = nstep_next_obs
            sp_nstep_gamma[t] = nstep_gamma

        # For critic target, use the last nstep sample (not the full horizon)
        last_nstep = self.buffer.sample(indice=indices[-1])
        (_, _, _, _, last_nstep_reward, _, _, last_nstep_term, _, last_nstep_next_obs, _, last_nstep_gamma, _, _) = last_nstep

        train_info = self._model_train(
            sp_obs, sp_actions, sp_reward, sp_next_obs,
            last_nstep_reward, last_nstep_term, last_nstep_next_obs, last_nstep_gamma,
        )

        if self.total_it % self.policy_freq == 0:
            actor_info = self._actor_train()
            train_info.update(actor_info)

        self.critic.soft_update()
        return train_info

    def _model_train(
        self, obs, actions, reward, next_obs,
        nstep_reward, nstep_term, nstep_next_obs, nstep_gamma,
    ):
        """Update world-model components (encoder, SSM, flow, MoE, critic)."""
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        reward = check(reward).to(**self.tpdv)
        next_obs = check(next_obs).to(**self.tpdv)
        nstep_reward = check(nstep_reward).to(**self.tpdv)
        nstep_term = check(nstep_term).to(**self.tpdv)
        nstep_next_obs = check(nstep_next_obs).to(**self.tpdv)
        nstep_gamma = check(nstep_gamma).to(**self.tpdv)

        n_agents, horizon, batch_size, _ = obs.shape

        wm_cfg = self.algo_args["world_model"]
        loss_w = wm_cfg.get("loss_weights", {})

        # -- Critic target --
        with torch.no_grad():
            next_nstep_latents = [
                self.dual_encoder(nstep_next_obs[i]) for i in range(self.num_agents)
            ]
            dummy_a = torch.zeros(batch_size, self.num_agents, self.action_dims[0], **self.tpdv)
            nstep_latents_h, _ = self.sequence_model(next_nstep_latents, dummy_a)
            next_nstep_h = [nstep_latents_h[i].h for i in range(self.num_agents)]
            next_nstep_actions = [
                self.actor[i].get_actions(next_nstep_h[i])
                for i in range(self.num_agents)
            ]

        next_q = self.critic.get_target_values(
            share_obs=torch.cat(next_nstep_h, dim=-1),
            actions=torch.cat(next_nstep_actions, dim=-1),
        )
        q_targets = nstep_reward + nstep_gamma * next_q * (1 - nstep_term)

        # -- Enable gradients --
        self._model_turn_on_grad()

        info = {"reward_acc": 0.0, "reward_err": 0.0}
        rec_loss_total = 0.0
        latent_loss_total = 0.0
        flow_loss_total = 0.0
        reward_loss_total = 0.0
        q_loss_total = 0.0
        balance_loss_total = 0.0

        for t in range(horizon):
            # Encode current obs
            latents_t = [self.dual_encoder(obs[i, t]) for i in range(self.num_agents)]
            actions_t = actions[:, t].permute(1, 0, 2)  # (na, bs, ad) -> for SSM

            # Reconstruction + Barlow Twins losses
            for i in range(self.num_agents):
                rec_loss_total += self.dual_encoder.compute_reconstruction_loss(
                    obs[i, t]
                ) * (self.step_rho ** t)
                latent_loss_total += self.dual_encoder.compute_latent_loss(
                    obs[i, t]
                ) * (self.step_rho ** t)

            # Sequence model forward
            act_for_ssm = torch.stack(
                [actions[i, t] for i in range(self.num_agents)], dim=1
            )  # (bs, na, ad)
            latents_h, msgs = self.sequence_model(latents_t, act_for_ssm)

            # Flow prediction + loss
            for i in range(self.num_agents):
                next_lat = self.dual_encoder(next_obs[i, t])
                z_flat = next_lat.z.reshape(batch_size, -1)
                x_1_target = torch.cat([z_flat, next_lat.d], dim=-1)

                bias = self.soft_moe.compute_bias(latents_h[i].h)
                self.flow_predictor.velocity_field.set_moe_bias(bias)
                flow_out = self.flow_predictor(x_1_target, latents_h[i].h, msgs[i])
                flow_losses = FlowMatchingLoss.compute(
                    v_pred=flow_out.v_field,
                    x_0=torch.randn_like(x_1_target),
                    x_1=x_1_target,
                    tau=torch.rand(batch_size, 1, device=self.device),
                    velocity_field=self.flow_predictor.velocity_field,
                    h=latents_h[i].h,
                    m=msgs[i],
                )
                flow_loss_total += (
                    flow_losses["flow_loss"] + flow_losses["shortcut_loss"]
                ) * (self.step_rho ** t)
            self.flow_predictor.velocity_field.clear_moe_bias()

            # Reward prediction
            s_t_parts = []
            for i in range(self.num_agents):
                z_flat = latents_h[i].z.reshape(batch_size, -1)
                s_t_parts.append(
                    torch.cat([z_flat, latents_h[i].d, latents_h[i].h], dim=-1)
                )
            s_t = torch.cat(s_t_parts, dim=-1)
            r_logits, aux = self.reward_head(s_t)
            reward_loss_total += self.reward_processor.dis_reg_loss(
                logits=r_logits, target=reward[t]
            ).mean() * (self.step_rho ** t)
            balance_loss_total += aux["loss_balancing"] * (self.step_rho ** t)

            with torch.no_grad():
                _r_pred = self.reward_processor.logits_decode_scalar(r_logits)
                _error = torch.abs(reward[t] - _r_pred)
                info["reward_acc"] += (_error <= 0.05).sum().item() / _error.shape[0] / horizon
                info["reward_err"] += torch.mean(_error).item() / horizon

            # Q-function loss
            joint_h = torch.cat([latents_h[i].h for i in range(self.num_agents)], dim=-1)
            joint_act_t = torch.cat([actions[i, t] for i in range(self.num_agents)], dim=-1)
            q_logits1 = self.critic.critic(joint_h, joint_act_t)
            q_logits2 = self.critic.critic2(joint_h, joint_act_t)
            # q_targets shape: (batch_size, 1) — same target for all horizon steps
            q_target_t = q_targets if q_targets.dim() == 2 else q_targets[t]
            q_loss_total += (
                (
                    self.critic.processor.dis_reg_loss(logits=q_logits1, target=q_target_t).mean()
                    + self.critic.processor.dis_reg_loss(logits=q_logits2, target=q_target_t).mean()
                )
                / 2
            ) * (self.step_rho ** t)

        # Average over horizon
        rec_loss_total /= horizon
        latent_loss_total /= horizon
        flow_loss_total /= horizon
        reward_loss_total /= horizon
        q_loss_total /= horizon
        balance_loss_total /= horizon

        # If balance loss is non-finite, replace with zero (preserves gradients for other losses)
        if not torch.isfinite(balance_loss_total):
            balance_loss_total = torch.zeros_like(balance_loss_total)

        # -- Record per-component losses --
        info["L_rec"] = float(rec_loss_total)
        info["L_latent"] = float(latent_loss_total)
        info["L_flow"] = float(flow_loss_total)
        info["L_reward"] = float(reward_loss_total)
        info["L_q"] = float(q_loss_total)
        info["L_balance"] = float(balance_loss_total)

        total_loss = (
            loss_w.get("rec", 1.0) * rec_loss_total
            + loss_w.get("latent", 1.0) * latent_loss_total
            + loss_w.get("flow", 1.0) * flow_loss_total
            + wm_cfg["reward_coef"] * reward_loss_total
            + wm_cfg["q_coef"] * q_loss_total
            + wm_cfg["balance_coef"] * balance_loss_total
        )
        info["L_total"] = float(total_loss)

        # -- Weighted contributions (for debugging loss balance) --
        info["L_rec_w"] = float(loss_w.get("rec", 1.0) * rec_loss_total)
        info["L_latent_w"] = float(loss_w.get("latent", 1.0) * latent_loss_total)
        info["L_flow_w"] = float(loss_w.get("flow", 1.0) * flow_loss_total)
        info["L_reward_w"] = float(wm_cfg["reward_coef"] * reward_loss_total)
        info["L_q_w"] = float(wm_cfg["q_coef"] * q_loss_total)

        # Skip update if loss is non-finite (prevents corrupting parameters)
        if not torch.isfinite(total_loss):
            info["L_total"] = float(total_loss)
            info["skipped_update"] = 1.0
            self._model_turn_off_grad()
            return info
        info["skipped_update"] = 0.0

        self.model_optimizer.zero_grad()
        total_loss.backward()

        # Replace NaN/Inf gradients with zero
        for group in self.model_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        # -- Gradient norms per component --
        def _grad_norm(params):
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                return 0.0
            return float(torch.nn.utils.clip_grad_norm_(grads, float("inf")))

        info["grad_norm/encoder"] = _grad_norm(self.dual_encoder.parameters())
        info["grad_norm/ssm"] = _grad_norm(self.sequence_model.parameters())
        info["grad_norm/flow"] = _grad_norm(self.flow_predictor.parameters())
        info["grad_norm/soft_moe"] = _grad_norm(self.soft_moe.parameters())
        info["grad_norm/reward_head"] = _grad_norm(self.reward_head.parameters())
        info["grad_norm/critic"] = _grad_norm(
            list(self.critic.critic.parameters()) + list(self.critic.critic2.parameters())
        )

        for group in self.model_optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], self.grad_clip_norm)
        self.model_optimizer.step()

        # -- Learning rate --
        info["lr"] = self.model_optimizer.param_groups[0]["lr"]

        # -- VAE temperature --
        info["vae_temperature"] = float(self.dual_encoder.visual_encoder.tau)

        self._model_turn_off_grad()
        return info

    def _actor_train(self):
        """Train actors using imagined rollouts."""
        info = {"actor_loss": [0.0] * self.num_agents}

        # Sample a batch from buffer for actor training
        if self.buffer.cur_size < self.buffer.batch_size:
            return info

        data = self.buffer.sample(
            indice=torch.randperm(self.buffer.cur_size).numpy()[: self.buffer.batch_size]
        )
        (_, obs_batch, _, _, _, _, _, _, _, _, _, _, _, _) = data

        obs_batch = check(obs_batch).to(**self.tpdv)  # (n_agents, batch_size, obs_dim)

        # Encode and get h
        with torch.no_grad():
            latents = [self.dual_encoder(obs_batch[i]) for i in range(self.num_agents)]
            dummy_a = torch.zeros(
                obs_batch.shape[1], self.num_agents, self.action_dims[0], **self.tpdv
            )
            latents_h, _ = self.sequence_model(latents, dummy_a)
            zs = [latents_h[i].h for i in range(self.num_agents)]

        # Get initial actions
        actions = []
        logp_actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                a, lp = self.actor[i].get_actions_with_logprobs(zs[i])
                actions.append(a)
                logp_actions.append(lp)

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(np.random.permutation(self.num_agents))

        for agent_id in agent_order:
            self.actor[agent_id].turn_on_grad()

            actions[agent_id], logp_actions[agent_id] = self.actor[
                agent_id
            ].get_actions_with_logprobs(zs[agent_id])

            value_pred = self.critic.get_values(
                torch.cat(zs, dim=-1),
                torch.cat(actions, dim=-1),
                "mean",
            )
            self.critic.scale.update(value_pred[0])
            value_pred = self.critic.scale(value_pred)

            actor_loss = (
                self.entropy_coef * logp_actions[agent_id] - value_pred
            ).mean()

            self.actor[agent_id].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor[agent_id].actor_optimizer.step()
            self.actor[agent_id].turn_off_grad()

            actions[agent_id], _ = self.actor[agent_id].get_actions_with_logprobs(
                zs[agent_id]
            )

            info["actor_loss"][agent_id] = actor_loss.item()
            info["pi_scale"] = self.critic.scale.value

        return info

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self, warmup_train=False, train_steps=10000):
        warmup_steps = (
            self.algo_args["train"]["warmup_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        obs, share_obs, _ = self.envs.reset()

        for _ in range(warmup_steps):
            actions = self._sample_random_actions()
            (new_obs, new_share_obs, rewards, dones, infos, _) = self.envs.step(actions)
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                None,
                rewards,
                dones,
                infos,
                new_share_obs.copy(),
                new_obs.copy(),
                None,
            )
            self._insert(data)
            obs = new_obs
            share_obs = new_share_obs

        if warmup_train:
            warmup_info = {}
            for i in range(1, train_steps + 1):
                _info = self._train()
                for k, v in _info.items():
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        warmup_info.setdefault(k, []).append(v)
                if i % 100 == 0:
                    for k in warmup_info:
                        warmup_info[k] = np.mean(warmup_info[k])
                    self._console_log(self.total_it, warmup_info=warmup_info)
                    warmup_info = {}

        return obs, share_obs, None

    # ------------------------------------------------------------------
    # Env helpers
    # ------------------------------------------------------------------

    def _sample_random_actions(self):
        actions = []
        for agent_id in range(self.num_agents):
            action = [
                self.action_spaces[agent_id].sample()
                for _ in range(self.algo_args["train"]["n_rollout_threads"])
            ]
            actions.append(action)
        return np.array(actions).transpose(1, 0, 2)

    def _insert(self, data):
        (
            share_obs, obs, actions, available_actions,
            rewards, dones, infos, next_share_obs, next_obs,
            next_available_actions,
        ) = data

        n_threads = self.algo_args["train"]["n_rollout_threads"]
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        valid_transitions = 1 - self.agent_deaths
        self.agent_deaths = np.expand_dims(dones, axis=-1)

        terms = np.full((n_threads, 1), False)
        for i in range(n_threads):
            if dones_env[i]:
                if not (
                    "bad_transition" in infos[i][0].keys()
                    and infos[i][0]["bad_transition"]
                ):
                    terms[i][0] = True

        for i in range(n_threads):
            if dones_env[i]:
                self.done_episodes_rewards[self.task_idxes[i]].append(
                    self.train_episode_rewards[i]
                )
                self.train_episode_rewards[i] = 0
                self.agent_deaths = np.zeros((n_threads, self.num_agents, 1))
                self.t0[i] = True
            else:
                self.t0[i] = False

        data = (
            share_obs[:, 0],
            obs,
            actions,
            available_actions,
            rewards[:, 0],
            np.expand_dims(dones_env, axis=-1),
            valid_transitions.transpose(1, 0, 2),
            terms,
            next_share_obs[:, 0],
            next_obs.transpose(1, 0, 2),
            next_available_actions,
        )
        self.buffer.insert(data)

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval(self, step):
        eval_episode_rewards = []
        one_episode_rewards = []
        n_eval = self.algo_args["eval"]["n_eval_rollout_threads"]
        for _ in range(n_eval):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        episode_lens = []
        one_episode_len = np.zeros(n_eval, dtype=np.int32)

        eval_obs, _, _ = self.eval_envs.reset()
        t0 = [True] * n_eval

        while True:
            eval_actions = self._plan_actions(eval_obs, t0, add_random=False)
            (eval_new_obs, _, eval_rewards, eval_dones, _, _) = self.eval_envs.step(
                eval_actions
            )
            eval_obs = eval_new_obs.copy()

            for i in range(n_eval):
                one_episode_rewards[i].append(eval_rewards[i])
            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)
            for i in range(n_eval):
                if eval_dones_env[i]:
                    eval_episode += 1
                    eval_episode_rewards[i].append(
                        np.sum(one_episode_rewards[i], axis=0)
                    )
                    one_episode_rewards[i] = []
                    episode_lens.append(one_episode_len[i].copy())
                    one_episode_len[i] = 0
                    t0[i] = True
                else:
                    t0[i] = False

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                eval_episode_rewards = np.concatenate(
                    [r for r in eval_episode_rewards if r]
                )
                eval_info = {
                    "eval_avg_rew": np.mean(eval_episode_rewards),
                    "eval_std_rew": np.std(eval_episode_rewards),
                    "eval_min_rew": np.min(eval_episode_rewards),
                    "eval_max_rew": np.max(eval_episode_rewards),
                    "eval_avg_len": np.mean(episode_lens),
                }
                # Per-agent reward if multi-dim
                if eval_episode_rewards.ndim > 1:
                    for ai in range(eval_episode_rewards.shape[-1]):
                        eval_info[f"eval_rew_agent{ai}"] = float(
                            np.mean(eval_episode_rewards[:, ai])
                        )
                self._console_log(step, eval_info=eval_info)

                # Log an eval video to wandb if rendering is available
                if self.algo_args["logger"].get("wandb", False):
                    try:
                        frames = []
                        vid_obs, _, _ = self.eval_envs.reset()
                        vid_t0 = [True] * n_eval
                        for _vt in range(200):
                            frame = self.eval_envs.render(mode="rgb_array")
                            if frame is not None:
                                if isinstance(frame, list):
                                    frame = frame[0]
                                frames.append(frame)
                            vid_actions = self._plan_actions(vid_obs, vid_t0, add_random=False)
                            vid_obs, _, _, vid_dones, _, _ = self.eval_envs.step(vid_actions)
                            vid_t0 = [False] * n_eval
                            vid_dones_env = np.all(vid_dones, axis=1)
                            if vid_dones_env[0]:
                                break
                        if len(frames) > 0:
                            # Stack to (T, H, W, C) then transpose to (T, C, H, W)
                            video = np.stack(frames, axis=0)
                            if video.ndim == 4:
                                video = video.transpose(0, 3, 1, 2)
                            # Add batch dim -> (1, T, C, H, W)
                            video = video[np.newaxis]
                            wandb.log(
                                {"eval/video": wandb.Video(video, fps=30)},
                                step=self.total_it,
                            )
                    except Exception:
                        pass  # Don't crash training if render fails

                break

    # ------------------------------------------------------------------
    # Logging / saving
    # ------------------------------------------------------------------

    def _init_config(self):
        task = get_task_name(self.args["env"], self.env_args)
        hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        run_dir = str(
            os.path.join(
                self.algo_args["logger"]["log_dir"],
                self.args["env"],
                task,
                self.args["algo"],
                "-".join([hms_time, f"seed-{self.algo_args['seed']['seed']:03d}"]),
            )
        )
        os.makedirs(run_dir, exist_ok=True)
        save_config(self.args, self.algo_args, self.env_args, run_dir)

        save_dir = os.path.join(run_dir, "models")
        if self.algo_args["logger"]["save_model"]:
            os.makedirs(save_dir, exist_ok=True)

        log_file = open(
            os.path.join(run_dir, "progress.txt"), "w", encoding="utf-8"
        )
        expt_name = "-".join([hms_time, f"seed-{self.algo_args['seed']['seed']:03d}"])
        return run_dir, save_dir, log_file, task, expt_name

    def _console_log(self, _step, **kwargs):
        use_wandb = self.algo_args["logger"].get("wandb", False)

        if use_wandb:
            wandb_dict = {"global_step": self.total_it}
            for group_name, metrics in kwargs.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if isinstance(v, (int, float, np.floating, np.integer)):
                            wandb_dict[f"{group_name}/{k}"] = float(v)
                        elif isinstance(v, list):
                            for idx, vi in enumerate(v):
                                if isinstance(vi, (int, float, np.floating, np.integer)):
                                    wandb_dict[f"{group_name}/{k}_agent{idx}"] = float(vi)
            wandb.log(wandb_dict, step=self.total_it)

        print_lines = [
            "",
            "******** iter: %s, steps: %s/%s, iter_time: %.1fs, total_time: "
            % (
                convert_num(self.total_it),
                convert_num(_step * self.algo_args["train"]["n_rollout_threads"]),
                convert_num(self.algo_args["train"]["num_env_steps"]),
                time.time() - self._check_time,
            )
            + str(datetime.timedelta(seconds=int(time.time() - self._start_time)))
            + " ********",
        ]
        for key, value in kwargs.items():
            if isinstance(value, dict):
                parts = []
                for k, v in value.items():
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        parts.append(f"{k}: {v:.6f}")
                    elif isinstance(v, list):
                        parts.append(f"{k}: [{', '.join(f'{vi:.4f}' for vi in v)}]")
                print_lines.append(f"{key}: {', '.join(parts)}")
        for line in print_lines:
            print(line)
            self.log_file.write(line + "\n")
        self.log_file.flush()
        self._check_time = time.time()

    def _save(self):
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(self.save_dir, agent_id)
        self.critic.save(self.save_dir)
        torch.save(
            self.dual_encoder.state_dict(),
            os.path.join(self.save_dir, "dual_encoder.pt"),
        )
        torch.save(
            self.sequence_model.state_dict(),
            os.path.join(self.save_dir, "sequence_model.pt"),
        )
        torch.save(
            self.flow_predictor.state_dict(),
            os.path.join(self.save_dir, "flow_predictor.pt"),
        )
        torch.save(
            self.soft_moe.state_dict(),
            os.path.join(self.save_dir, "soft_moe.pt"),
        )
        torch.save(
            self.reward_head.state_dict(),
            os.path.join(self.save_dir, "reward_head.pt"),
        )

    def close(self):
        self.envs.close()
        if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
            self.eval_envs.close()
        self.log_file.close()
        if self.algo_args["logger"].get("wandb", False):
            wandb.finish()

    def _model_turn_on_grad(self):
        for p in self.dual_encoder.parameters():
            p.requires_grad = True
        for p in self.sequence_model.parameters():
            p.requires_grad = True
        for p in self.flow_predictor.parameters():
            p.requires_grad = True
        for p in self.soft_moe.parameters():
            p.requires_grad = True
        for p in self.reward_head.parameters():
            p.requires_grad = True
        for p in self.continue_pred.parameters():
            p.requires_grad = True
        self.critic.turn_on_grad()

    def _model_turn_off_grad(self):
        for p in self.dual_encoder.parameters():
            p.requires_grad = False
        for p in self.sequence_model.parameters():
            p.requires_grad = False
        for p in self.flow_predictor.parameters():
            p.requires_grad = False
        for p in self.soft_moe.parameters():
            p.requires_grad = False
        for p in self.reward_head.parameters():
            p.requires_grad = False
        for p in self.continue_pred.parameters():
            p.requires_grad = False
        self.critic.turn_off_grad()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDELINE-MARL training script")
    parser.add_argument(
        "--load-config",
        type=str,
        default="configs/edeline_marl/mujoco/config.json",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.load_config, encoding="utf-8") as f:
        all_config = json.load(f)

    run_args = {
        "algo": all_config["main_args"]["algo"],
        "env": all_config["main_args"]["env"],
        "exp_name": all_config["main_args"]["exp_name"],
    }
    algo_args = all_config["algo_args"]
    env_args = all_config["env_args"]

    runner = EdelineRunner(run_args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
