"""
MPPI Planner for EDELINE-MARL.

Implements Model Predictive Path Integral (MPPI) planning over the dual latent
space, using the WorldModel's sub-components (flow predictor, MoE modules,
sequence model) for imagination rollouts.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from m3w.interfaces import AgentLatent


class MPPIPlanner:
    """MPPI planner that operates over dual latent space.

    Performs trajectory optimisation by sampling action sequences, rolling them
    out through the world model in imagination, and refining via elite
    re-weighting.

    Args:
        cfg: Dictionary with planning hyper-parameters.
            Required keys:
                horizon, num_samples, num_elites, temperature,
                num_pi_trajs, max_std, min_std
            Optional keys:
                iterations (default 6), gamma (default 0.99),
                flow_steps_infer (default 1)
    """

    def __init__(self, cfg: Dict):
        self.horizon = cfg["horizon"]
        self.num_samples = cfg["num_samples"]
        self.num_elites = cfg["num_elites"]
        self.temperature = cfg["temperature"]
        self.num_pi_trajs = cfg["num_pi_trajs"]
        self.max_std = cfg["max_std"]
        self.min_std = cfg["min_std"]
        self.iterations = cfg.get("iterations", 6)
        self.gamma = cfg.get("gamma", 0.99)
        self.flow_steps_infer = cfg.get("flow_steps_infer", 1)

        # Running mean for warm-starting across time steps (set externally).
        # List of tensors, one per agent: each (horizon, B, action_dim)
        self._running_mean: Optional[List[torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def plan(
        self,
        world_model,
        latents: List[AgentLatent],
        messages: List[torch.Tensor],
        actors: list,
        action_dims: List[int],
        t0: Optional[List[bool]] = None,
        eval_mode: bool = False,
        critic=None,
    ) -> np.ndarray:
        """Run MPPI planning and return the first action for each agent.

        Args:
            world_model: WorldModel instance (or any object exposing
                ``flow_predictor``, ``soft_moe``, ``reward_head``,
                ``reward_processor``).
            latents:  List of N AgentLatent with ``.h`` filled.
            messages: List of N message tensors, each (B, hidden_dim).
            actors:   List of N WorldModelActor instances.
            action_dims: List of per-agent action dimensions.
            t0:       Per-environment boolean flags; True = episode start
                      (resets the warm-start mean).
            eval_mode: If True, do not add final action noise.
            critic:   Optional WorldModelCritic for terminal value bootstrap.

        Returns:
            actions: np.ndarray of shape (B, N, max_action_dim).
        """
        device = latents[0].h.device
        dtype = latents[0].h.dtype
        tpdv = dict(dtype=dtype, device=device)
        B = latents[0].h.shape[0]
        N = len(latents)

        if t0 is None:
            t0 = [True] * B

        # ---- Initialise mean / std ----
        act_mean = [
            torch.zeros(self.horizon, B, action_dims[i], **tpdv)
            for i in range(N)
        ]
        act_std = [
            self.max_std * torch.ones(self.horizon, B, action_dims[i], **tpdv)
            for i in range(N)
        ]

        # Warm-start from previous planning step
        if self._running_mean is not None and len(self._running_mean) == N:
            for b in range(B):
                if not t0[b]:
                    for i in range(N):
                        act_mean[i][:-1, b] = self._running_mean[i][1:, b]

        # Pre-allocate action buffers: (horizon, B, num_samples, action_dim)
        actions = [
            torch.zeros(self.horizon, B, self.num_samples, action_dims[i], **tpdv)
            for i in range(N)
        ]

        # ---- Generate policy trajectories ----
        if self.num_pi_trajs > 0:
            pi_actions, _ = self._rollout_policy(
                world_model, latents, messages, actors, action_dims, B, N, tpdv,
            )
            for i in range(N):
                actions[i][:, :, :self.num_pi_trajs, :] = pi_actions[i]

        # ---- Output buffer ----
        out_a = [
            torch.zeros(B, action_dims[i], **tpdv)
            for i in range(N)
        ]

        # ---- MPPI iterations ----
        for it in range(self.iterations):
            # Sample Gaussian actions for the non-policy slots
            for i in range(N):
                n_gauss = self.num_samples - self.num_pi_trajs
                actions[i][:, :, self.num_pi_trajs:] = torch.normal(
                    mean=act_mean[i].unsqueeze(2).expand(-1, -1, n_gauss, -1),
                    std=act_std[i].unsqueeze(2).expand(-1, -1, n_gauss, -1),
                ).clamp(-1, 1)

            # Evaluate all trajectories
            g_returns = self._estimate_returns(
                world_model, latents, messages, actors, actions,
                action_dims, B, N, tpdv, critic=critic,
            )

            # CEM update per agent
            for i in range(N):
                # Average returns across agents for shared reward
                value = torch.mean(
                    torch.stack(g_returns, dim=0), dim=0
                ).squeeze(-1)  # (B, num_samples)

                elite_idxes = torch.topk(value, self.num_elites, dim=-1)[1]
                elite_values = torch.gather(value, dim=-1, index=elite_idxes)
                elite_actions = torch.gather(
                    actions[i], dim=2,
                    index=elite_idxes.unsqueeze(0).unsqueeze(-1).expand(
                        self.horizon, -1, -1, action_dims[i]
                    ),
                )

                max_value = elite_values.max(dim=-1, keepdim=True)[0]
                score = torch.exp(self.temperature * (elite_values - max_value))
                score = score / score.sum(dim=-1, keepdim=True)
                score_w = score.unsqueeze(0).unsqueeze(-1)  # (1, B, num_elites, 1)

                act_mean[i] = (score_w * elite_actions).sum(dim=2)
                act_std[i] = torch.sqrt(
                    (score_w * (elite_actions - act_mean[i].unsqueeze(2)) ** 2).sum(dim=2) + 1e-6
                ).clamp_(self.min_std, self.max_std)

                # On last iteration, sample the final action
                if it == self.iterations - 1:
                    score_np = score.cpu().numpy()
                    for b in range(B):
                        idx = np.random.choice(
                            np.arange(self.num_elites), p=score_np[b]
                        )
                        out_a[i][b] = elite_actions[0, b, idx]
                        if not eval_mode:
                            out_a[i][b] += (
                                torch.randn_like(out_a[i][b]) * act_std[i][0, b]
                            )
                            out_a[i][b] = out_a[i][b].clamp(-1, 1)

        # Save running mean for warm-starting
        self._running_mean = [m.clone() for m in act_mean]

        # Assemble output: (B, N, action_dim)
        out_np = [out_a[i].cpu().numpy() for i in range(N)]
        out_np = np.stack(out_np, axis=1)  # (B, N, action_dim)
        return out_np

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rollout_policy(
        self, world_model, latents, messages, actors, action_dims, B, N, tpdv,
    ):
        """Roll out policy trajectories through imagination.

        Returns:
            pi_actions: list of N tensors (horizon, B, num_pi_trajs, action_dim)
            final_latents: list of N AgentLatent at end of rollout
        """
        pi_actions = [
            torch.zeros(
                self.horizon, B, self.num_pi_trajs, action_dims[i], **tpdv
            )
            for i in range(N)
        ]

        # Expand latents and messages for num_pi_trajs
        cur_h = [
            latents[i].h.unsqueeze(1).expand(-1, self.num_pi_trajs, -1).reshape(
                B * self.num_pi_trajs, -1
            )
            for i in range(N)
        ]
        cur_m = [
            messages[i].unsqueeze(1).expand(-1, self.num_pi_trajs, -1).reshape(
                B * self.num_pi_trajs, -1
            )
            for i in range(N)
        ]

        for t in range(self.horizon):
            # Get policy actions
            for i in range(N):
                a_i = actors[i].get_actions(
                    cur_h[i], stochastic=True
                )  # (B*num_pi_trajs, action_dim)
                pi_actions[i][t] = a_i.reshape(B, self.num_pi_trajs, action_dims[i])

            # Imagination step: use flow predictor for next latent
            for i in range(N):
                # Set MoE bias
                bias = world_model.soft_moe.compute_bias(cur_h[i])
                world_model.flow_predictor.velocity_field.set_moe_bias(bias)
                # Sample next latent
                flow_out = world_model.flow_predictor.sample(
                    h=cur_h[i], m=cur_m[i], steps=self.flow_steps_infer,
                )
                # Use the predicted latent as the next hidden state
                # (simplified: project flow output back to hidden dim)
                cur_h[i] = self._project_to_hidden(
                    flow_out.x_hat, cur_h[i].shape[-1], tpdv
                )

            world_model.flow_predictor.velocity_field.clear_moe_bias()

        return pi_actions, cur_h

    @staticmethod
    def _project_to_hidden(
        x_hat: torch.Tensor, hidden_dim: int, tpdv: dict,
    ) -> torch.Tensor:
        """Project flow predictor output back to hidden dimension.

        A simple linear projection (or truncation) to bridge
        latent_dim -> hidden_dim for the next step's conditioning.
        """
        if x_hat.shape[-1] == hidden_dim:
            return x_hat
        # Simple truncation / slice (the first hidden_dim dims)
        return x_hat[:, :hidden_dim]

    def _estimate_returns(
        self,
        world_model,
        latents,
        messages,
        actors,
        actions,
        action_dims,
        B,
        N,
        tpdv,
        critic=None,
    ):
        """Estimate discounted returns for all sampled trajectories.

        Args:
            actions: list of N tensors (horizon, B, num_samples, action_dim)

        Returns:
            g_returns: list of N tensors (B, num_samples, 1)
        """
        horizon = actions[0].shape[0]
        num_samples = actions[0].shape[2]

        # Expand latent states for num_samples
        cur_h = [
            latents[i].h.unsqueeze(1).expand(-1, num_samples, -1).reshape(
                B * num_samples, -1
            )
            for i in range(N)
        ]
        cur_m = [
            messages[i].unsqueeze(1).expand(-1, num_samples, -1).reshape(
                B * num_samples, -1
            )
            for i in range(N)
        ]

        returns = [
            torch.zeros(horizon + 1, B, num_samples, 1, **tpdv)
            for _ in range(N)
        ]

        # Compute per-agent padding dim: reward head expects [z_flat; d; h] per agent
        # but planner only has h. Pad with zeros for z_flat and d.
        hidden_dim = cur_h[0].shape[-1]
        per_agent_dim = world_model.reward_head.per_agent_dim
        pad_dim = per_agent_dim - hidden_dim  # = latent_dim (z_flat + d)

        for t in range(horizon):
            # Build the concatenated state for reward prediction
            # Pad each agent's h with zeros for the latent portion
            state_parts = []
            for i in range(N):
                pad = torch.zeros(
                    cur_h[i].shape[0], pad_dim,
                    device=cur_h[i].device, dtype=cur_h[i].dtype,
                )
                state_parts.append(torch.cat([pad, cur_h[i]], dim=-1))
            s_t = torch.cat(state_parts, dim=-1)

            # Predict reward
            r_logits, _ = world_model.reward_head(s_t)  # (B*num_samples, num_bins)
            r_value = world_model.reward_processor.logits_decode_scalar(
                r_logits
            )  # (B*num_samples, 1)
            r_value = r_value.reshape(B, num_samples, -1)

            for i in range(N):
                returns[i][t + 1] = returns[i][t] + (self.gamma ** t) * r_value

            # Imagination step: flow predict next latent per agent
            for i in range(N):
                bias = world_model.soft_moe.compute_bias(cur_h[i])
                world_model.flow_predictor.velocity_field.set_moe_bias(bias)
                flow_out = world_model.flow_predictor.sample(
                    h=cur_h[i], m=cur_m[i], steps=self.flow_steps_infer,
                )
                cur_h[i] = self._project_to_hidden(
                    flow_out.x_hat, cur_h[i].shape[-1], tpdv
                )
            world_model.flow_predictor.velocity_field.clear_moe_bias()

        # Terminal value bootstrap using critic
        if critic is not None:
            joint_h = torch.cat(
                [h.reshape(B, num_samples, -1) for h in cur_h], dim=-1
            )  # (B, num_samples, N*hidden_dim)
            joint_actions = torch.cat(
                [
                    actors[i].get_actions(
                        cur_h[i], stochastic=True
                    ).reshape(B, num_samples, -1)
                    for i in range(N)
                ],
                dim=-1,
            )  # (B, num_samples, sum_action_dim)
            horizon_q = critic.get_values(
                joint_h, joint_actions, mode="mean"
            )  # (B, num_samples, 1)
            for i in range(N):
                returns[i][-1] = returns[i][-2] + (self.gamma ** horizon) * horizon_q

        g_returns = [
            returns[i][-1].nan_to_num(0)
            for i in range(N)
        ]
        return g_returns


__all__ = ["MPPIPlanner"]
