"""
Transformer State-Space Model for Multi-Agent (TSSM-MA).

Implements the sequence model components for EDELINE-MARL:
  C1: ProjectDualLatent  -- projects dual latents (z, d) into hidden space
  C2: TransformerSSM     -- causal Transformer over time for a single agent
  C3: CrossAgentAttention-- inter-agent message passing via attention
  C4: TransformerSSM_MA  -- full multi-agent sequence model
  C5: ACCPC_MA           -- multi-agent AC-CPC contrastive loss
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from m3w.interfaces import (
    AgentLatent,
    LATENT_DIM,
    VISUAL_LATENT_DIM,
    SEM_LATENT_DIM,
)


# ---------------------------------------------------------------------------
# C1 -- ProjectDualLatent
# ---------------------------------------------------------------------------

class ProjectDualLatent(nn.Module):
    """Flatten visual latent z and concatenate with semantic latent d,
    then linearly project to hidden_dim.

    Input:
        z: (B, num_cats, cat_dim)  -- e.g. (B, 32, 32) -> flat (B, 1024)
        d: (B, sem_dim)            -- e.g. (B, 512)
    Output:
        (B, hidden_dim)
    """

    def __init__(
        self,
        visual_dim: int = VISUAL_LATENT_DIM,
        sem_dim: int = SEM_LATENT_DIM,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.sem_dim = sem_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(visual_dim + sem_dim, hidden_dim)

    def forward(self, z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, num_cats, cat_dim) visual latent
            d: (B, sem_dim) semantic latent
        Returns:
            (B, hidden_dim)
        """
        z_flat = z.reshape(z.shape[0], -1)  # (B, visual_dim)
        x = torch.cat([z_flat, d], dim=-1)  # (B, visual_dim + sem_dim)
        return self.proj(x)


# ---------------------------------------------------------------------------
# C2 -- TransformerSSM
# ---------------------------------------------------------------------------

class TransformerSSM(nn.Module):
    """Causal Transformer over the time axis for a single agent.

    Input per step: proj(x_t^i) + action_emb(a_t^i)   (element-wise addition)
    Output per step: h_t^i in R^{hidden_dim}
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        action_dim: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_emb = nn.Linear(action_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    @staticmethod
    def _generate_causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = masked position)."""
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(
        self,
        projected: torch.Tensor,
        actions: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            projected: (B, T, hidden_dim)  -- already projected dual latent
            actions:   (B, T, action_dim)
            padding_mask: (B, T) bool, True for padded positions
        Returns:
            (B, T, hidden_dim)
        """
        a_emb = self.action_emb(actions)          # (B, T, hidden_dim)
        x = projected + a_emb                     # element-wise addition

        T = x.shape[1]
        causal_mask = self._generate_causal_mask(T, x.device)

        h = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return h  # (B, T, hidden_dim)


# ---------------------------------------------------------------------------
# C3 -- CrossAgentAttention
# ---------------------------------------------------------------------------

class CrossAgentAttention(nn.Module):
    """For agent i, attend over {h_t^j : j != i} using h_t^i as query.

    Returns message m_t^i -- aggregated teammate context.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        agent_hiddens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            agent_hiddens: (B, N, hidden_dim) -- h_t for all N agents
            padding_mask:  (B, N) bool, True for padded / absent agents
        Returns:
            messages: (B, N, hidden_dim) -- m_t^i for each agent i
        """
        B, N, D = agent_hiddens.shape
        messages = torch.zeros_like(agent_hiddens)

        for i in range(N):
            # Build key/value set: all agents except i
            others_idx = [j for j in range(N) if j != i]
            if len(others_idx) == 0:
                # Single agent -- no message
                continue

            query = agent_hiddens[:, i : i + 1, :]  # (B, 1, D)
            kv = agent_hiddens[:, others_idx, :]     # (B, N-1, D)

            kv_mask: Optional[torch.Tensor] = None
            if padding_mask is not None:
                kv_mask = padding_mask[:, others_idx]  # (B, N-1)

            msg, _ = self.attn(
                query, kv, kv,
                key_padding_mask=kv_mask,
                need_weights=False,
            )  # (B, 1, D)
            messages[:, i, :] = msg.squeeze(1)

        return messages


# ---------------------------------------------------------------------------
# C4 -- TransformerSSM_MA (multi-agent wrapper)
# ---------------------------------------------------------------------------

class TransformerSSM_MA(nn.Module):
    """Multi-agent Transformer SSM.

    Combines ProjectDualLatent, TransformerSSM, and CrossAgentAttention.
    Processes each agent's latent sequence, applies cross-agent attention,
    and fills AgentLatent.h.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        action_dim: Optional[int] = None,
        num_agents: int = 2,
        cross_num_heads: int = 4,
        dropout: float = 0.1,
        visual_dim: Optional[int] = None,
        sem_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if action_dim is None:
            raise ValueError("action_dim must be specified")

        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.project = ProjectDualLatent(
            visual_dim=visual_dim if visual_dim is not None else VISUAL_LATENT_DIM,
            sem_dim=sem_dim if sem_dim is not None else SEM_LATENT_DIM,
            hidden_dim=hidden_dim,
        )
        self.ssm = TransformerSSM(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            action_dim=action_dim,
            dropout=dropout,
        )
        self.cross_attn = CrossAgentAttention(
            hidden_dim=hidden_dim,
            num_heads=cross_num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        latents: List[AgentLatent],
        actions: torch.Tensor,
    ) -> Tuple[List[AgentLatent], List[torch.Tensor]]:
        """
        Args:
            latents: list of N AgentLatent (h may be None from encoder).
                     z: (B, num_cats, cat_dim), d: (B, sem_dim)
            actions: (B, N, action_dim) or list of (B, action_dim) tensors
        Returns:
            updated_latents: list of N AgentLatent with h filled
            messages: list of N tensors of shape (B, hidden_dim)
        """
        N = len(latents)

        # Handle actions as list or tensor
        if isinstance(actions, (list, tuple)):
            actions_list = actions
        else:
            # (B, N, action_dim) -> list of (B, action_dim)
            actions_list = [actions[:, i] for i in range(N)]

        # Project dual latents and run SSM for each agent
        # For single-step inference: T=1
        agent_h = []
        for i in range(N):
            z_i = latents[i].z  # (B, num_cats, cat_dim)
            d_i = latents[i].d  # (B, sem_dim)
            a_i = actions_list[i]  # (B, action_dim)

            proj_i = self.project(z_i, d_i)  # (B, hidden_dim)
            # Add time dimension: (B, 1, hidden_dim) / (B, 1, action_dim)
            proj_i = proj_i.unsqueeze(1)
            a_i = a_i.unsqueeze(1)

            h_i = self.ssm(proj_i, a_i)  # (B, 1, hidden_dim)
            h_i = h_i.squeeze(1)          # (B, hidden_dim)
            agent_h.append(h_i)

        # Stack for cross-agent attention: (B, N, hidden_dim)
        stacked_h = torch.stack(agent_h, dim=1)
        messages_stacked = self.cross_attn(stacked_h)  # (B, N, hidden_dim)

        # Build outputs
        updated_latents = []
        messages = []
        for i in range(N):
            updated = AgentLatent(
                z=latents[i].z,
                d=latents[i].d,
                h=agent_h[i],
            )
            updated_latents.append(updated)
            messages.append(messages_stacked[:, i, :])

        return updated_latents, messages


# ---------------------------------------------------------------------------
# C5 -- ACCPC_MA (Multi-Agent AC-CPC)
# ---------------------------------------------------------------------------

class ACCPC_MA(nn.Module):
    """Multi-Agent Action-Conditional Contrastive Predictive Coding.

    Each agent i predicts the future latent of agent j (j != i) using
    (s_t^i, a_{t:t+k}^i).  Loss: InfoNCE with cosine similarity.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        latent_dim: int = LATENT_DIM,
        num_steps_ahead: int = 3,
        action_dim: int = 6,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_steps_ahead = num_steps_ahead
        self.temperature = temperature

        # Project full latent (z_flat; d) into hidden space
        self.projector = nn.Linear(latent_dim, hidden_dim)

        # One predictor per step-ahead horizon
        self.predictors = nn.ModuleList()
        for k in range(1, num_steps_ahead + 1):
            self.predictors.append(
                nn.Sequential(
                    nn.Linear(hidden_dim + k * action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

    def _flatten_latent(self, latent: AgentLatent) -> torch.Tensor:
        """Flatten z and concatenate with d: (B, latent_dim)."""
        z_flat = latent.z.reshape(latent.z.shape[0], -1)
        return torch.cat([z_flat, latent.d], dim=-1)

    def compute_loss(
        self,
        agent_states: List[List[AgentLatent]],
        agent_actions: torch.Tensor,
        agent_future_latents: List[List[AgentLatent]],
    ) -> torch.Tensor:
        """
        Args:
            agent_states: agent_states[i][t] = AgentLatent for agent i at time t
                          N agents, T timesteps
            agent_actions: (B, N, T, action_dim)
            agent_future_latents: agent_future_latents[j][t+k] = AgentLatent
                                  for agent j at future time t+k.  N agents,
                                  T + num_steps_ahead timesteps.
        Returns:
            Scalar InfoNCE loss averaged over agents, timesteps, and horizons.
        """
        N = len(agent_states)
        T = len(agent_states[0])
        B = agent_states[0][0].z.shape[0]
        device = agent_states[0][0].z.device

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                for t in range(T):
                    # State of agent i at time t
                    s_i_t = self._flatten_latent(agent_states[i][t])  # (B, latent_dim)
                    s_proj = self.projector(s_i_t)  # (B, hidden_dim)

                    for k_idx, k in enumerate(range(1, self.num_steps_ahead + 1)):
                        future_t = t + k
                        if future_t >= len(agent_future_latents[j]):
                            continue

                        # Actions of agent i from t to t+k-1: (B, k * action_dim)
                        if t + k > agent_actions.shape[2]:
                            continue
                        acts = agent_actions[:, i, t : t + k, :].reshape(B, -1)

                        # Predict future latent of agent j
                        pred_input = torch.cat([s_proj, acts], dim=-1)
                        pred = self.predictors[k_idx](pred_input)  # (B, hidden_dim)

                        # Target: projected future latent of agent j
                        target_latent = self._flatten_latent(
                            agent_future_latents[j][future_t]
                        )
                        target = self.projector(target_latent)  # (B, hidden_dim)

                        # InfoNCE with cosine similarity
                        # Positive: matching pairs along batch dim
                        pred_norm = F.normalize(pred, dim=-1)
                        target_norm = F.normalize(target, dim=-1)

                        # Logits: (B, B) -- each row is one query
                        logits = (
                            torch.mm(pred_norm, target_norm.t()) / self.temperature
                        )
                        labels = torch.arange(B, device=device)
                        loss = F.cross_entropy(logits, labels)

                        total_loss = total_loss + loss
                        count += 1

        if count == 0:
            return total_loss
        return total_loss / count
