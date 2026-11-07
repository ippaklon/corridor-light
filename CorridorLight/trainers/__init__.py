"""
Training loop.

``CorridorRLTrainer`` owns the SUMO episode, corridor replanning, and
checkpointing. ``MGDATrainer`` updates one intersection agent so that the
local objective and the cooperative objective share a backbone without
one gradient cancelling the other.
"""

from .mgda_trainer import MGDATrainer
from .corridor_trainer import CorridorRLTrainer

__all__ = [
    "MGDATrainer",
    "CorridorRLTrainer",
]

