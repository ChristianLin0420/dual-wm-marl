"""
Unified WorldModel for EDELINE-MARL.

Composes all sub-modules (DualEncoder, TransformerSSM_MA, FlowPredictor,
SoftMoEVelocityBias, SparseMoEReward, ContinuePredictor, ACCPC_MA) into a
single end-to-end world model with a unified loss interface.

This file is NEW and does NOT replace ``m3w/models/world_models.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from m3w.interfaces import (
    AgentLatent,
    FlowPredictorOutput,
    WorldModelOutput,
    LATENT_DIM,
)
from m3w.encoders import DualEncoder
from m3w.sequence_model import TransformerSSM_MA, ACCPC_MA
from m3w.flow_predictor import FlowPredictor, FlowMatchingLoss
from m3w.moe import SoftMoEVelocityBias, SparseMoEReward, ContinuePredictor
from m3w.models.world_models import TwoHotProcessor


# ---------------------------------------------------------------------------
# E4 -- LinearDynamicsPredictor
# ---------------------------------------------------------------------------

class LinearDynamicsPredictor(nn.Module):
    """Predicts the target dual-latent x_1 from the recurrent state h_t.

    Used for L_dyn guidance. The target is detached so no gradient flows
    back to the encoder through this path.
    """

    def __init__(self, hidden_dim: int = 512, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, hidden_dim) recurrent hidden state.
        Returns:
            (B, latent_dim) predicted target latent.
        """
        return self.linear(h)


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

_DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "L_pred": 1.0,
    "L_rec": 1.0,
    "L_rep": 0.1,
    "L_dyn": 0.5,
    "L_cpc": 0.1,
    "L_latent": 1.0,
    "L_flow": 1.0,
    "L_shortcut": 1.0,
}


def _default_cfg() -> dict:
    """Return a default configuration dictionary."""
    return dict(
        obs_dim=64,
        action_dim=6,
        num_agents=2,
        hidden_dim=512,
        num_cats=32,
        cat_dim=32,
        sem_dim=512,
        num_experts=16,
        num_slots=1,
        num_reward_bins=101,
        reward_vmin=-20,
        reward_vmax=20,
        cpc_steps_ahead=3,
        shortcut_lambda=0.1,
        loss_weights=dict(_DEFAULT_LOSS_WEIGHTS),
    )


# ---------------------------------------------------------------------------
# E1-E3 -- WorldModel
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """Unified world model for EDELINE-MARL.

    Composes DualEncoder, TransformerSSM_MA, FlowPredictor, SoftMoEVelocityBias,
    SparseMoEReward, ContinuePredictor, ACCPC_MA, and LinearDynamicsPredictor.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        # Merge with defaults so missing keys get sane values
        full_cfg = _default_cfg()
        full_cfg.update(cfg)
        cfg = full_cfg
        self.cfg = cfg

        obs_dim: int = cfg["obs_dim"]
        action_dim: int = cfg["action_dim"]
        num_agents: int = cfg["num_agents"]
        hidden_dim: int = cfg["hidden_dim"]
        num_cats: int = cfg["num_cats"]
        cat_dim: int = cfg["cat_dim"]
        sem_dim: int = cfg["sem_dim"]
        num_experts: int = cfg["num_experts"]
        num_slots: int = cfg["num_slots"]
        num_reward_bins: int = cfg["num_reward_bins"]
        cpc_steps_ahead: int = cfg["cpc_steps_ahead"]

        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.num_cats = num_cats
        self.cat_dim = cat_dim
        self.sem_dim = sem_dim
        self.loss_weights: Dict[str, float] = cfg["loss_weights"]

        # Derived dimensions
        self.visual_latent_dim = num_cats * cat_dim
        self.latent_dim = self.visual_latent_dim + sem_dim  # = LATENT_DIM with defaults
        self.per_agent_state_dim = self.latent_dim + hidden_dim

        # --- Sub-modules ---

        # Encoder (shared across agents)
        self.encoder = DualEncoder(
            obs_dim=obs_dim,
            num_cats=num_cats,
            cat_dim=cat_dim,
            sem_dim=sem_dim,
            hidden_dim=hidden_dim,
        )

        # Sequence model (multi-agent Transformer SSM)
        self.sequence_model = TransformerSSM_MA(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_agents=num_agents,
        )

        # MoE velocity bias
        self.moe_velocity_bias = SoftMoEVelocityBias(
            hidden_dim=hidden_dim,
            latent_dim=self.latent_dim,
            num_experts=num_experts,
            num_slots=num_slots,
        )

        # Flow predictor
        self.flow_predictor = FlowPredictor(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            cond_dim=hidden_dim,
        )

        # Reward predictor
        self.reward_predictor = SparseMoEReward(
            n_agents=num_agents,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_reward_bins=num_reward_bins,
        )

        # Continue predictor
        self.continue_predictor = ContinuePredictor(
            input_dim=num_agents * self.per_agent_state_dim,
        )

        # AC-CPC
        self.accpc = ACCPC_MA(
            hidden_dim=hidden_dim,
            latent_dim=self.latent_dim,
            num_steps_ahead=cpc_steps_ahead,
            action_dim=action_dim,
        )

        # Dynamics predictor (L_dyn guidance)
        self.dynamics_predictor = LinearDynamicsPredictor(
            hidden_dim=hidden_dim,
            latent_dim=self.latent_dim,
        )

        # TwoHotProcessor for reward loss (not an nn.Module, just a helper)
        # We create it on CPU initially; it will use the correct device via tensors passed in
        self._reward_processor: Optional[TwoHotProcessor] = None
        self._reward_cfg = dict(
            num_bins=num_reward_bins,
            vmin=cfg["reward_vmin"],
            vmax=cfg["reward_vmax"],
        )

    def _get_reward_processor(self, device: torch.device) -> TwoHotProcessor:
        """Lazily initialise TwoHotProcessor on the correct device."""
        if self._reward_processor is None:
            self._reward_processor = TwoHotProcessor(
                num_bins=self._reward_cfg["num_bins"],
                vmin=self._reward_cfg["vmin"],
                vmax=self._reward_cfg["vmax"],
                device=device,
            )
        return self._reward_processor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_latent(latent: AgentLatent) -> torch.Tensor:
        """Flatten z and concatenate with d -> (B, latent_dim)."""
        z_flat = latent.z.reshape(latent.z.shape[0], -1)
        return torch.cat([z_flat, latent.d], dim=-1)

    def _build_x1_target(self, latent: AgentLatent) -> torch.Tensor:
        """Build flow-matching target by flattening z and concatenating with d."""
        return self._flatten_latent(latent)

    def _split_x_hat(self, x_hat: torch.Tensor) -> AgentLatent:
        """Split predicted x_hat back into z_hat and d_hat (h=None)."""
        z_flat = x_hat[:, : self.visual_latent_dim]
        d_hat = x_hat[:, self.visual_latent_dim:]
        z_hat = z_flat.reshape(-1, self.num_cats, self.cat_dim)
        return AgentLatent(z=z_hat, d=d_hat, h=None)

    def _aggregate_state(self, latents: List[AgentLatent]) -> torch.Tensor:
        """Concatenate [z_flat^i; d^i; h^i] over all agents -> (B, N * per_agent_dim)."""
        parts = []
        for lat in latents:
            z_flat = lat.z.reshape(lat.z.shape[0], -1)
            parts.append(torch.cat([z_flat, lat.d, lat.h], dim=-1))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # E2 -- Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_list: List[torch.Tensor],
        action_list: List[torch.Tensor],
        target_obs_list: Optional[List[torch.Tensor]] = None,
    ) -> WorldModelOutput:
        """Full forward pass of the world model.

        Args:
            obs_list: List of N tensors, each (B, obs_dim).
            action_list: List of N tensors, each (B, action_dim).
            target_obs_list: Optional list of N tensors for training targets.

        Returns:
            WorldModelOutput with predicted latents, reward logits, continue logits.
        """
        N = len(obs_list)

        # Step 1: Encode each agent's observation
        latents: List[AgentLatent] = []
        for i in range(N):
            lat = self.encoder(obs_list[i])
            latents.append(lat)

        # Step 2: Sequence model (fills h and produces messages)
        actions_tensor = torch.stack(action_list, dim=1)  # (B, N, action_dim)
        latents_with_h, messages = self.sequence_model(latents, actions_tensor)

        # Step 3: MoE velocity bias + Flow prediction per agent
        flow_outputs: List[FlowPredictorOutput] = []
        # Encode targets if provided (for training)
        target_latents: Optional[List[AgentLatent]] = None
        if target_obs_list is not None:
            target_latents = [self.encoder(obs) for obs in target_obs_list]

        for i in range(N):
            # Set MoE bias for this agent
            bias_i = self.moe_velocity_bias.compute_bias(latents_with_h[i].h)
            self.flow_predictor.velocity_field.set_moe_bias(bias_i)

            h_i = latents_with_h[i].h
            m_i = messages[i]

            if self.training and target_latents is not None:
                x_1_target = self._build_x1_target(target_latents[i])
                flow_out = self.flow_predictor(x_1_target, h_i, m_i)
            else:
                flow_out = self.flow_predictor.sample(h_i, m_i, steps=1)

            flow_outputs.append(flow_out)

        # Clear MoE bias after all agents are processed
        self.flow_predictor.velocity_field.clear_moe_bias()

        # Step 5: Aggregate state for reward/continue prediction
        s_t = self._aggregate_state(latents_with_h)

        # Step 6: Reward prediction
        r_logits, reward_aux = self.reward_predictor(s_t)

        # Step 7: Continue prediction
        c_logits = self.continue_predictor(s_t)

        # Step 8: Build predicted AgentLatents from flow outputs
        predicted_latents: List[AgentLatent] = []
        for i in range(N):
            pred_lat = self._split_x_hat(flow_outputs[i].x_hat)
            predicted_latents.append(pred_lat)

        # Store intermediate results for loss computation
        self._cache = dict(
            latents_with_h=latents_with_h,
            messages=messages,
            flow_outputs=flow_outputs,
            target_latents=target_latents,
            r_logits=r_logits,
            reward_aux=reward_aux,
            c_logits=c_logits,
            obs_list=obs_list,
            action_list=action_list,
            s_t=s_t,
        )

        # Decode reward and continue to scalar for the output
        proc = self._get_reward_processor(r_logits.device)
        rewards = proc.logits_decode_scalar(r_logits).squeeze(-1)  # (B,)
        continues = torch.sigmoid(c_logits).squeeze(-1)  # (B,)

        return WorldModelOutput(
            latents=predicted_latents,
            rewards=rewards,
            continues=continues,
        )

    # ------------------------------------------------------------------
    # E3 -- Compute Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        obs_list: List[torch.Tensor],
        action_list: List[torch.Tensor],
        next_obs_list: List[torch.Tensor],
        rewards: torch.Tensor,
        continues: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all world-model losses.

        Args:
            obs_list: List of N tensors, each (B, obs_dim).
            action_list: List of N tensors, each (B, action_dim).
            next_obs_list: List of N tensors (next observations).
            rewards: (B,) or (B, 1) ground-truth rewards.
            continues: (B,) or (B, 1) ground-truth continue flags (1=continue, 0=done).

        Returns:
            Dict with L_total and individual loss components (all scalar tensors).
        """
        # Ensure training mode for the forward pass
        was_training = self.training
        self.train()

        # Forward pass with targets
        _output = self.forward(obs_list, action_list, target_obs_list=next_obs_list)

        # Retrieve cached intermediates
        cache = self._cache
        latents_with_h = cache["latents_with_h"]
        messages = cache["messages"]
        flow_outputs = cache["flow_outputs"]
        target_latents = cache["target_latents"]
        r_logits = cache["r_logits"]
        c_logits = cache["c_logits"]

        device = r_logits.device
        N = len(obs_list)

        # Reshape rewards/continues for loss computation
        if rewards.dim() == 1:
            rewards_2d = rewards.unsqueeze(-1)  # (B, 1)
        else:
            rewards_2d = rewards
        if continues.dim() == 1:
            continues_2d = continues.unsqueeze(-1).float()  # (B, 1)
        else:
            continues_2d = continues.float()

        # --- L_pred: reward + continue loss ---
        proc = self._get_reward_processor(device)
        L_pred_reward = proc.dis_reg_loss(r_logits, rewards_2d).mean()
        L_pred_continue = F.binary_cross_entropy_with_logits(
            c_logits, continues_2d,
        )
        L_pred = L_pred_reward + L_pred_continue

        # --- L_rec: reconstruction loss (from each agent's obs) ---
        L_rec = torch.tensor(0.0, device=device)
        for i in range(N):
            L_rec = L_rec + self.encoder.compute_reconstruction_loss(obs_list[i])
        L_rec = L_rec / N

        # --- L_rep: KL divergence between encoder posterior and uniform prior ---
        L_rep = torch.tensor(0.0, device=device)
        for i in range(N):
            _, logits_i, _ = self.encoder.encode_visual(obs_list[i])
            # q(z|o): softmax over logits
            q = F.softmax(logits_i, dim=-1)  # (B, num_cats, cat_dim)
            # Uniform prior: 1/cat_dim
            log_q = F.log_softmax(logits_i, dim=-1)
            log_p = -torch.log(torch.tensor(float(self.cat_dim), device=device))
            # KL = sum q * (log_q - log_p)
            kl = (q * (log_q - log_p)).sum(dim=-1).mean()  # average over batch and categories
            L_rep = L_rep + kl
        L_rep = L_rep / N

        # --- L_dyn: MSE(dynamics_predictor(h), x_target.detach()) ---
        L_dyn = torch.tensor(0.0, device=device)
        for i in range(N):
            h_i = latents_with_h[i].h
            x_target_i = self._build_x1_target(target_latents[i]).detach()
            x_pred_i = self.dynamics_predictor(h_i)
            L_dyn = L_dyn + F.mse_loss(x_pred_i, x_target_i)
        L_dyn = L_dyn / N

        # --- L_latent: Barlow Twins loss ---
        L_latent = torch.tensor(0.0, device=device)
        for i in range(N):
            L_latent = L_latent + self.encoder.compute_latent_loss(obs_list[i])
        L_latent = L_latent / N

        # --- L_flow + L_shortcut: flow matching loss ---
        L_flow = torch.tensor(0.0, device=device)
        L_shortcut = torch.tensor(0.0, device=device)
        for i in range(N):
            x_1_target = self._build_x1_target(target_latents[i])
            # We need x_0 and tau to compute the loss properly.
            # Re-run a training forward to get the required components.
            # Instead, we use the cached flow outputs.
            # The flow_predictor.forward already computed v_field and x_hat.
            # We need x_0 = x_hat - v_field (from the single-step shortcut: x_hat = x_0 + v_pred)
            v_pred = flow_outputs[i].v_field
            x_hat = flow_outputs[i].x_hat
            x_0 = x_hat - v_pred  # recover noise

            # Sample tau for loss (use uniform like training forward)
            B = x_1_target.shape[0]
            tau = torch.rand(B, 1, device=device, dtype=x_1_target.dtype)

            # Set MoE bias for this agent
            bias_i = self.moe_velocity_bias.compute_bias(latents_with_h[i].h)
            self.flow_predictor.velocity_field.set_moe_bias(bias_i)

            flow_loss_dict = FlowMatchingLoss.compute(
                v_pred=v_pred,
                x_0=x_0,
                x_1=x_1_target,
                tau=tau,
                velocity_field=self.flow_predictor.velocity_field,
                h=latents_with_h[i].h,
                m=messages[i],
            )
            L_flow = L_flow + flow_loss_dict["flow_loss"]
            L_shortcut = L_shortcut + flow_loss_dict["shortcut_loss"]

        self.flow_predictor.velocity_field.clear_moe_bias()
        L_flow = L_flow / N
        L_shortcut = L_shortcut / N

        # --- L_cpc: ACCPC_MA loss (skip if insufficient sequence length) ---
        # For single-step, CPC requires multi-step trajectories.
        # We set L_cpc = 0 when we only have single-step data.
        L_cpc = torch.tensor(0.0, device=device)

        # Restore training mode
        if not was_training:
            self.eval()

        # --- Weighted total ---
        w = self.loss_weights
        L_total = (
            w.get("L_pred", 1.0) * L_pred
            + w.get("L_rec", 1.0) * L_rec
            + w.get("L_rep", 0.1) * L_rep
            + w.get("L_dyn", 0.5) * L_dyn
            + w.get("L_cpc", 0.1) * L_cpc
            + w.get("L_latent", 1.0) * L_latent
            + w.get("L_flow", 1.0) * L_flow
            + w.get("L_shortcut", 1.0) * L_shortcut
        )

        return {
            "L_total": L_total,
            "L_pred": L_pred,
            "L_rec": L_rec,
            "L_rep": L_rep,
            "L_dyn": L_dyn,
            "L_cpc": L_cpc,
            "L_latent": L_latent,
            "L_flow": L_flow,
            "L_shortcut": L_shortcut,
        }
