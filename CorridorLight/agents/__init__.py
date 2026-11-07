"""
The two decision makers.

``CorridorAgent`` plans a small set of junction-disjoint corridor tasks.
``IntersectionAgent`` executes one signal, mixing a local policy with a
policy conditioned on the task it was assigned.
"""

from .corridor_agent import CorridorAgent, Corridor
from .intersection_agent import IntersectionAgent

__all__ = [
    "CorridorAgent",
    "Corridor",
    "IntersectionAgent",
]

