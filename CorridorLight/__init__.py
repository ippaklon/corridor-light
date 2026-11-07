"""
CorridorLight: Cooperation as Task Negotiation via Counterfactual-Gain Gating for Traffic Signal Control.

The package is the NeurIPS 2026 implementation of a two-level controller.

* Upper level (``agents.corridor_agent``). A GNN encodes a movement graph.
  A seed module ranks movements and a grow module extends each seed into a
  corridor. Corridors that would share a controlled junction are rejected.
* Lower level (``agents.intersection_agent``). Every signal keeps a local
  policy and a cooperative policy. A counterfactual-gain gate mixes them
  from the gap between the two critics.
* Training (``trainers``). Intersection agents are updated with MGDA-PPO.
  The corridor agent is updated from the return of the lanes it claimed.

``main.py`` is the supported entry point. Importing this package also
imports the trainer, which requires ``SUMO_HOME``.
"""

from .config import (
    Config, MapConfig, AgentConfig, get_configs, get_configs_by_name, apply_runtime_logging_defaults
)
from .models.pressure_graph import PressureGraphBuilder
from .models.gnn_encoder import GNNEncoder
from .models.seed_module import SeedModule
from .models.grow_module import GrowModule
from .agents.corridor_agent import CorridorAgent, Corridor
from .agents.intersection_agent import IntersectionAgent
from .trainers.mgda_trainer import MGDATrainer
from .trainers.corridor_trainer import CorridorRLTrainer

try:
    from .utils import (
        setup_logging, MetricsTracker, compute_gae, normalize_advantages,
        set_seed, count_parameters, save_checkpoint, load_checkpoint,
        create_summary_table, format_time, get_device_info, EarlyStopping
    )
except ImportError:
    # Optional utilities.
    setup_logging = None
    MetricsTracker = None
    compute_gae = None
    normalize_advantages = None
    set_seed = None
    count_parameters = None
    save_checkpoint = None
    load_checkpoint = None
    create_summary_table = None
    format_time = None
    get_device_info = None
    EarlyStopping = None

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "Config",
    "MapConfig",
    "AgentConfig",
    "get_configs",
    "get_configs_by_name",
    "apply_runtime_logging_defaults",
    
    # Models
    "PressureGraphBuilder",
    "GNNEncoder",
    "SeedModule",
    "GrowModule",
    
    # Agents
    "CorridorAgent",
    "Corridor",
    "IntersectionAgent",
    
    # Trainers
    "MGDATrainer",
    "CorridorRLTrainer",
    
    # Utilities (if available)
    "setup_logging",
    "MetricsTracker",
    "compute_gae",
    "normalize_advantages",
    "set_seed",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "create_summary_table",
    "format_time",
    "get_device_info",
    "EarlyStopping",
]

