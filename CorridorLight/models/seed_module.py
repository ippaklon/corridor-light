"""
Score movements so the corridor agent knows where to start.

The grow policy can extend a corridor, but it cannot choose its origin.
This module maps each GNN embedding to a scalar state value ``V(v)``.
``CorridorAgent`` sorts those values and tries seeds from best to worst,
skipping a seed whose junction is already owned by a corridor accepted
earlier in the same replan.

The head is intentionally small. The topology lives in the GNN. This
module only has to rank nodes.
"""

import torch
import torch.nn as nn


class SeedModule(nn.Module):
    """Two-layer MLP from a movement embedding to ``V(v)``."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        """
        Args:
            embedding_dim: GNN embedding dimension.
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return ``[num_nodes]`` state values. Higher means a better seed."""
        return self.value_head(embeddings).squeeze(-1)
    
    def select_top_k(self, embeddings: torch.Tensor, k: int) -> list:
        """Indices of the ``k`` highest-valued movements, best first.

        The trainer usually ranks every node itself so that a rejected seed
        can be replaced. This helper is the same ranking for callers that
        only want the shortlist.
        """
        seed_values = self.forward(embeddings)
        _, top_k_indices = torch.topk(seed_values, k)
        return top_k_indices.tolist()

