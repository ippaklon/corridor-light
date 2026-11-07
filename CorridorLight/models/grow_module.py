"""
Grow one corridor, one movement at a time.

The state is the mean embedding of the movements already on the corridor.
Each legal neighbor is scored by concatenating that state with the
neighbor's embedding. STOP is a real action, but its logit is the constant
0 rather than a learned parameter: the policy learns whether a neighbor is
worth taking by pushing that neighbor's logit above or below zero. Illegal
neighbors are masked to ``-inf`` after the heads run.

Three heads read the same concatenated pair:

* policy logit — probability of stepping onto this neighbor,
* Q value — return of that step, trained by the corridor update,
* preference in ``(0, 1)`` — how strongly the intersection agent should
  favor the lanes of this movement. Preferences are normalized later,
  across the whole corridor, into a distribution.

The preference head's last linear layer is initialized smaller than
Xavier's default so early preferences are not all stuck near 0.5.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class GrowModule(nn.Module):
    """Autoregressive neighbor policy used by ``CorridorAgent._grow_corridor``."""
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        max_neighbors: int = 4,
        max_length: int = 10
    ):
        """
        Args:
            embedding_dim: GNN embedding dimension.
            hidden_dim: Hidden dimension.
            max_neighbors: Maximum number of neighbors.
            max_length: Maximum corridor length.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_neighbors = max_neighbors
        self.max_length = max_length
        # Neighbors plus one STOP action. The mask, not this width, decides
        # which neighbors exist at a given node.
        self.action_dim = max_neighbors + 1

        # Mean-pool the prefix, then lift it to the hidden size shared with
        # the candidate embedding.
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q_value_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.preference_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # A full Xavier init puts the pre-sigmoid unit near 0, so every
        # preference starts at 0.5 and the first gradients have nothing to
        # break the tie. A smaller gain leaves room for the data.
        nn.init.xavier_uniform_(self.preference_head[2].weight, gain=0.5)
        nn.init.zeros_(self.preference_head[2].bias)
    
    def encode_state(self, corridor_prefix: List[int], node_embeddings: torch.Tensor) -> torch.Tensor:
        """Mean-pool the movements already chosen and encode that vector.

        An empty prefix has no embedding to pool. Returning zeros keeps the
        caller from special-casing the first step. The zero vector matches
        ``state_encoder``'s input width, which is ``embedding_dim``.
        """
        if len(corridor_prefix) == 0:
            return torch.zeros(self.state_encoder[0].in_features)
        
        prefix_embeddings = node_embeddings[corridor_prefix]  # [prefix_len, embedding_dim]
        
        avg_embedding = prefix_embeddings.mean(dim=0)  # [embedding_dim]
        
        state = self.state_encoder(avg_embedding)
        return state
    
    def forward(
        self,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        action_mask: torch.Tensor,
        return_q_values: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        """Score every candidate and append the STOP action.

        ``candidate_embeddings`` is ``[N, embedding_dim]`` for the legal
        neighbors only. ``action_mask`` has length ``N + 1``: one flag per
        neighbor and a final flag for STOP. Masked logits are ``-inf`` so
        ``Categorical`` cannot sample them. STOP's own logit is 0. Its
        chance of winning is therefore set by how large the neighbor logits
        are, which is what the policy learns.

        Preferences are returned only for the ``N`` neighbors. STOP has no
        lane to prefer. Q values, when requested, include STOP so the
        trainer can regress the action that was actually taken.
        """
        num_candidates = candidate_embeddings.shape[0]
        dev = state.device
        # Batch candidates in one forward pass.
        state_exp = state.unsqueeze(0).expand(num_candidates, -1)  # [N, hidden]
        combined = torch.cat([state_exp, candidate_embeddings], dim=-1)  # [N, hidden+emb]
        action_logits = self.policy_head(combined).squeeze(-1)  # [N]
        preference_values = self.preference_head(combined).squeeze(-1)  # [N]
        if return_q_values:
            q_cand = self.q_value_head(combined).squeeze(-1)  # [N]
        stop_logit = torch.tensor(0.0, device=dev, dtype=action_logits.dtype)
        action_logits = torch.cat([action_logits, stop_logit.unsqueeze(0)], dim=0)  # [N+1]
        if return_q_values:
            stop_q = torch.tensor(0.0, device=dev, dtype=q_cand.dtype)
            q_values = torch.cat([q_cand, stop_q.unsqueeze(0)], dim=0)  # [N+1]
        
        # Keep mask and logits on the same device.
        if action_mask.device != action_logits.device:
            action_mask = action_mask.to(action_logits.device)
        
        # Validate mask length.
        if action_mask.shape[0] != action_logits.shape[0]:
            raise ValueError(
                f"action_mask length ({action_mask.shape[0]}) must equal action_logits length ({action_logits.shape[0]})"
            )
        
        # Apply mask.
        action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))
        if return_q_values:
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))
        
        if return_q_values:
            return action_logits, preference_values, q_values
        return action_logits, preference_values

