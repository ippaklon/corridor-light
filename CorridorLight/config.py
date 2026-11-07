"""
Configuration for a CorridorLight run.

Three dataclasses exist because a run is the merge of two files.

* ``MapConfig`` — SUMO network, demand, horizon, reward, and the gate
  friction ``lambda_friction_coeff``. Friction is map-owned: the same
  agent file is reused across networks, and the right ``ρ`` is not.
* ``AgentConfig`` — architecture, PPO, corridor budget, and logging.
* ``Config`` — the flat object the trainer actually reads. Unknown keys
  in a YAML file are ignored, so a map file can carry comments-as-keys
  without crashing, and an older checkpoint config still loads.

``get_configs`` merges an explicit pair of files. ``get_configs_by_name``
loads ``configs/agents/<agent>.yaml`` and ``configs/maps/<map>.yaml``.
After the merge, the map reward and the map friction win. The corridor
potential and the signal reward then measure the same thing, and a map
author can retune the gate without editing the agent file.

``apply_runtime_logging_defaults`` stamps ``wandb_run_name`` and
``save_dir`` with the map, the agent, a short hyperparameter suffix, and
a timestamp. The suffix is for the run list, not for reproducibility.
The YAML copies written next to each checkpoint are the reproducible record.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import datetime

_FILE_ENCODING = "utf-8"


@dataclass
class Config:
    """Flat config consumed by ``CorridorRLTrainer``.

    Defaults reproduce a small Cologne-8 run on CPU. The YAML files under
    ``configs/`` are the experiment settings and override almost all of
    these. Fields are documented in groups below. A field that also exists
    on ``MapConfig`` is taken from the map file when both are present.
    """
    
    # --- Simulator. Paths are relative to the repository root. ---
    net_file: str = "sumo_rl/nets/RESCO/cologne8/cologne8.net.xml"
    # One route file, or several separated by commas.
    route_file: str = "sumo_rl/nets/RESCO/cologne8/cologne8.rou.xml"
    # When set, net and route are read from this SUMO config instead.
    sumocfg_file: str = ""
    # Extra SUMO files (detectors, traffic-light programs), comma-separated.
    additional_files: str = ""
    # Appended verbatim to the SUMO command line built by the trainer.
    additional_sumo_cmd: str = ""
    gui: bool = False
    num_seconds: int = 8000
    min_green: int = 5
    delta_time: int = 5
    traffic_scale: float = 1.0
    
    # Training parameters
    agent_type: str = "corridor"
    max_episodes: int = 1000
    eval_interval: int = 100
    device: str = "cpu"
    num_threads: int = 4
    profile_timing: bool = False
    
    # SUMO-RL reward name: diff-waiting-time, queue, pressure, average-speed.
    reward_fn: str = "diff-waiting-time"
    # Multiplies both the signal reward and the corridor potential.
    reward_scale: float = 1.0

    # --- Corridor agent. ---
    # Local-only ablation. No gate and no upper-level update.
    disable_corridor_agent: bool = False
    # Forces the local-only ablation, because the corridor agent has nothing
    # to read without the movement encoder.
    disable_gnn: bool = False
    # Plan a corridor on the first environment step and hold it for the
    # rest of the episode, instead of replanning every corridor_delta_time.
    skip_corridor_updates: bool = False
    max_corridors: int = 3
    max_corridor_length: int = 10
    seed_top_k: int = 5
    # Discount along a corridor in CorridorAgent._compute_seed_value.
    corridor_reward_decay: float = 0.9
    # Seconds of simulated time between replans.
    corridor_delta_time: int = 30
    # When false, a corridor that covers a single lane is rejected.
    allow_single_lane_corridor: bool = True
    # Entropy bonus on the grow policy, added on top of the PPO entropy term.
    corridor_entropy_coef: float = 0.01
    # Subtracted from the upper-level return when the first grow action is STOP.
    corridor_immediate_stop_penalty: float = 0.05
    # Reduce the per-step corridor return: "sum", "mean", or "discounted_mean".
    corridor_reward_agg: str = "sum"
    
    # --- GNN. architecture is GCN, GAT, or GraphSAGE. ---
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 64
    gnn_num_layers: int = 2
    gnn_architecture: str = "GCN"

    # Hidden width of the seed-value MLP.
    seed_projection_dim: int = 64

    # --- Intersection agent and counterfactual-gain gate. ---
    hidden_dim: int = 128
    # ρ in λ = clip(Δ / ρ, 0, 1). Map YAML overrides this after the merge.
    lambda_friction_coeff: float = 1.0
    # Floor inside |V_coop| + |V_local| + eps, so the gate is defined at 0.
    lambda_delta_eps: float = 1e-6
    lane_feature_dim: int = 10
    # One shared backbone and heads for every signal. CLI: --share.
    share_intersection_parameters: bool = False

    # Optional safety rule that forces green on a saturated approach.
    # Off unless a config turns it on. The trainer reads these thresholds.
    congestion_force_green: bool = False
    congestion_metric: str = "queue"  # queue | density | queue_or_density
    congestion_queue_threshold: float = 0.9
    congestion_density_threshold: float = 0.9
    # Scale on the corridor bonus inside the cooperative return. A value of
    # 0 is rewritten to 1 at trainer startup for corridor_only and shaping,
    # so forgetting the coefficient does not silently disable the objective.
    corridor_global_reward_coef: float = 0.0
    # What the cooperative critic is trained to predict.
    #   corridor_only   — corridor-lane bonus only
    #   shaping         — local reward + coefficient * corridor bonus
    #   same_as_loc     — identical to the local reward; upper level skipped
    #   corridor_decayed — old name, rewritten to corridor_only
    reward_glob_mode: str = "corridor_only"
    # Wandb reward table controls.
    reward_log_window: int = 100
    reward_log_flush_partial: bool = True
    reward_log_max_rows: int = 50000
    
    # --- PPO and MGDA. Applied independently to each intersection agent. ---
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    ppo_lam: float = 0.95
    mgda_epsilon: float = 1e-8
    
    # Logging parameters
    wandb: bool = False
    wandb_project: str = "corridor-rl"
    wandb_run_name: str = "corridor_training"
    save_dir: str = "models"
    log_level: str = "INFO"
    # Traffic metrics collection
    collect_traffic_metrics: bool = True
    metrics_collect_interval: int = 10
    
    # --- Checkpoints. interval 0 disables them. ---
    checkpoint_interval: int = 0
    checkpoint_save_dir: str = "checkpoints"
    checkpoint_save_models: bool = True
    checkpoint_save_training_state: bool = True
    # Filled by get_configs with the YAML paths, then copied next to the
    # checkpoint so a run can be relaunched from the files that produced it.
    map_config_path: str = ""
    agent_config_path: str = ""
    
    # Evaluation parameters
    eval_episodes: int = 5
    
    # Q-value and V-value training coefficients
    q_value_coef: float = 0.5
    v_value_coef: float = 0.5
    
    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration."""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        # Ignore unknown keys.
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        converted_dict = {}
        for key, value in config_dict.items():
            # Keep only valid dataclass fields.
            if key not in valid_params:
                continue
                
            if isinstance(value, str):
                if 'e-' in value.lower() or 'e+' in value.lower():
                    try:
                        converted_dict[key] = float(value)
                    except ValueError:
                        converted_dict[key] = value
                else:
                    converted_dict[key] = value
            else:
                converted_dict[key] = value
        
        return cls(**converted_dict)
    
    def save(self, filepath: "str | Path") -> None:
        """Save configuration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        if path.suffix == '.json':
            with open(path, 'w', encoding=_FILE_ENCODING) as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w', encoding=_FILE_ENCODING) as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, filepath: "str | Path") -> 'Config':
        """Load configuration from file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        if path.suffix == '.json':
            with open(path, 'r', encoding=_FILE_ENCODING) as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r', encoding=_FILE_ENCODING) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)


@dataclass
class MapConfig:
    """Everything that changes when the city changes and the algorithm does not."""
    net_file: str = "sumo_rl/nets/RESCO/cologne8/cologne8.net.xml"
    route_file: str = "sumo_rl/nets/RESCO/cologne8/cologne8.rou.xml"
    # SUMO-RL reward function.
    reward_fn: str = "diff-waiting-time"
    # Env reward scale.
    reward_scale: float = 1.0
    sumocfg_file: str = ""
    additional_files: str = ""
    additional_sumo_cmd: str = ""
    gui: bool = False
    num_seconds: int = 8000
    min_green: int = 5
    delta_time: int = 5
    log_level: str = "INFO"
    traffic_scale: float = 1.0
    # ρ for this network. Wins over AgentConfig.lambda_friction_coeff.
    lambda_friction_coeff: float = 1.0

    @classmethod
    def from_file(cls, path: str) -> 'MapConfig':
        data = yaml.safe_load(Path(path).read_text(encoding=_FILE_ENCODING))
        # Ignore unknown keys.
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_data = {k: v for k, v in data.items() if k in valid_params}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentConfig:
    """Algorithm hyperparameters. See ``Config`` for what each field does."""
    agent_type: str = "corridor"
    device: str = "cpu"
    max_episodes: int = 1000
    eval_interval: int = 100
    num_threads: int = 4
    save_dir: str = "models"
    profile_timing: bool = False
    
    # Checkpoint parameters
    checkpoint_interval: int = 0  # 0 disables checkpoints.
    checkpoint_save_dir: str = "checkpoints"
    checkpoint_save_models: bool = True
    checkpoint_save_training_state: bool = True
    
    # Evaluation parameters
    eval_episodes: int = 5
    
    # Corridor RL specific
    reward_fn: str = "diff-waiting-time"
    reward_scale: float = 1.0
    disable_corridor_agent: bool = False  # Single-objective mode.
    disable_gnn: bool = False  # Use standard observations only.
    skip_corridor_updates: bool = False  # Skip upper-level updates.
    max_corridors: int = 3
    max_corridor_length: int = 10
    seed_top_k: int = 5
    corridor_reward_decay: float = 0.9
    corridor_delta_time: int = 20
    allow_single_lane_corridor: bool = True  # If false, require at least two lanes.
    corridor_entropy_coef: float = 0.01
    corridor_immediate_stop_penalty: float = 0.05
    # Upper-level feedback uses improvement in the selected metric.
    corridor_reward_agg: str = "sum"
    
    # GNN
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 64
    gnn_num_layers: int = 2
    gnn_architecture: str = "GCN"
    seed_projection_dim: int = 64
    
    # Intersection Agent
    hidden_dim: int = 128
    lambda_friction_coeff: float = 1.0
    lane_feature_dim: int = 10
    corridor_global_reward_coef: float = 0.0
    reward_glob_mode: str = "corridor_only"
    reward_log_window: int = 100
    reward_log_flush_partial: bool = True
    reward_log_max_rows: int = 50000
    
    # PPO/MGDA
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    ppo_lam: float = 0.95
    mgda_epsilon: float = 1e-8
    
    # Q-value and V-value training coefficients
    q_value_coef: float = 0.5
    v_value_coef: float = 0.5

    @classmethod
    def from_file(cls, path: str) -> 'AgentConfig':
        data = yaml.safe_load(Path(path).read_text(encoding=_FILE_ENCODING))
        # Ignore unknown keys.
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_data = {k: v for k, v in data.items() if k in valid_params}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_configs(
    map_config_file: Optional[str] = None,
    agent_config_file: Optional[str] = None,
    map_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge a map file and an agent file into one ``Config``.

    Either path may be omitted, in which case that side falls back to the
    dataclass defaults. ``map_name`` then overwrites ``net_file`` and
    ``route_file`` from the built-in table, which is how ``--map cologne8``
    still resolves when only one of the two files was passed explicitly.
    The map reward and the map friction overwrite the agent values.
    """
    default_all = Config.default().to_dict()
    map_defaults = {k: default_all[k] for k in ['net_file','route_file','reward_fn','reward_scale','sumocfg_file','additional_files','additional_sumo_cmd','gui','num_seconds','min_green','delta_time','log_level','traffic_scale','lambda_friction_coeff']}
    agent_defaults = {k: v for k, v in default_all.items() if k not in map_defaults}

    # Load configs and ignore unknown keys.
    map_data = yaml.safe_load(Path(map_config_file).read_text(encoding=_FILE_ENCODING)) if map_config_file else map_defaults
    agent_data = yaml.safe_load(Path(agent_config_file).read_text(encoding=_FILE_ENCODING)) if agent_config_file else agent_defaults
    
    # Filter unknown keys.
    import inspect
    map_sig = inspect.signature(MapConfig.__init__)
    agent_sig = inspect.signature(AgentConfig.__init__)
    map_valid_params = set(map_sig.parameters.keys()) - {'self'}
    agent_valid_params = set(agent_sig.parameters.keys()) - {'self'}
    
    map_filtered = {k: v for k, v in map_data.items() if k in map_valid_params}
    agent_filtered = {k: v for k, v in agent_data.items() if k in agent_valid_params}
    
    map_cfg = MapConfig(**map_filtered)
    agent_cfg = AgentConfig(**agent_filtered)

    # Map-name path overrides.
    if map_name:
        map_files = {
            "4x4-Lucas": ("sumo_rl/nets/4x4-Lucas/4x4.net.xml", "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"),
            # Folder and net file names differ here.
            "2way-single-intersection": ("sumo_rl/nets/2way-single-intersection/single-intersection.net.xml", "sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml"),
            "2x2grid": ("sumo_rl/nets/2x2grid/2x2.net.xml", "sumo_rl/nets/2x2grid/2x2.rou.xml"),
            # Actual 3x3grid file names.
            "3x3grid": ("sumo_rl/nets/3x3grid/3x3Grid2lanes.net.xml", "sumo_rl/nets/3x3grid/routes14000.rou.xml"),
            "4x4loop": ("sumo_rl/nets/4x4loop/4x4loop.net.xml", "sumo_rl/nets/4x4loop/4x4loop.rou.xml"),
            # Actual double file names.
            "double": ("sumo_rl/nets/double/network.net.xml", "sumo_rl/nets/double/flow.rou.xml"),
            # Case-sensitive directories on Linux.
            "Nguyen": ("sumo_rl/nets/Nguyen/nguyentl.net.xml", "sumo_rl/nets/Nguyen/nguyencontext.rou.xml"),
            "NguyenNoTL": ("sumo_rl/nets/Nguyen/nguyenNoTL.net.xml", "sumo_rl/nets/Nguyen/nguyencontext.rou.xml"),
            "OW": ("sumo_rl/nets/OW/OW.net.xml", "sumo_rl/nets/OW/OW-nowait.rou.xml"),
            "nanshan": ("sumo_rl/nets/nanshan/osm.net.xml", "sumo_rl/nets/nanshan/osm_pt.rou.xml"),
            "simple": ("sumo_rl/nets/simple/simple.net.xml", "sumo_rl/nets/simple/simple.rou.xml"),
            "single-intersection": ("sumo_rl/nets/single-intersection/single-intersection.net.xml", "sumo_rl/nets/single-intersection/single-intersection.rou.xml"),
            # RESCO
            "arterial4x4": ("sumo_rl/nets/RESCO/arterial4x4/arterial4x4.net.xml", "sumo_rl/nets/RESCO/arterial4x4/arterial4x4_1.rou.xml"),
            "resco_arterial4x4": ("sumo_rl/nets/RESCO/arterial4x4/arterial4x4.net.xml", "sumo_rl/nets/RESCO/arterial4x4/arterial4x4_1.rou.xml"),
            "resco_cologne1": ("sumo_rl/nets/RESCO/cologne1/cologne1.net.xml", "sumo_rl/nets/RESCO/cologne1/cologne1.rou.xml"),
            "resco_cologne3": ("sumo_rl/nets/RESCO/cologne3/cologne3.net.xml", "sumo_rl/nets/RESCO/cologne3/cologne3.rou.xml"),
            "resco_cologne8": ("sumo_rl/nets/RESCO/cologne8/cologne8.net.xml", "sumo_rl/nets/RESCO/cologne8/cologne8.rou.xml"),
            "cologne8": ("sumo_rl/nets/RESCO/cologne8/cologne8.net.xml", "sumo_rl/nets/RESCO/cologne8/cologne8.rou.xml"),
            "cologne1": ("sumo_rl/nets/RESCO/cologne1/cologne1.net.xml", "sumo_rl/nets/RESCO/cologne1/cologne1.rou.xml"),
            "cologne3": ("sumo_rl/nets/RESCO/cologne3/cologne3.net.xml", "sumo_rl/nets/RESCO/cologne3/cologne3.rou.xml"),
            "grid4x4": ("sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml", "sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml"),
            "resco_grid4x4": ("sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml", "sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml"),
            "ingolstadt1": ("sumo_rl/nets/RESCO/ingolstadt1/ingolstadt1.net.xml", "sumo_rl/nets/RESCO/ingolstadt1/ingolstadt1.rou.xml"),
            "ingolstadt7": ("sumo_rl/nets/RESCO/ingolstadt7/ingolstadt7.net.xml", "sumo_rl/nets/RESCO/ingolstadt7/ingolstadt7.rou.xml"),
            "ingolstadt21": ("sumo_rl/nets/RESCO/ingolstadt21/ingolstadt21.net.xml", "sumo_rl/nets/RESCO/ingolstadt21/ingolstadt21.rou.xml"),
        }
        if map_name in map_files:
            net_file, route_file = map_files[map_name]
            map_cfg.net_file = net_file
            map_cfg.route_file = route_file

    merged = {**map_cfg.to_dict(), **agent_cfg.to_dict()}
    # lambda_friction_coeff is map-owned: allow per-map tuning to override agent defaults.
    merged["lambda_friction_coeff"] = map_cfg.lambda_friction_coeff
    merged_config = Config.from_dict(merged)
    # Keep corridor reward aligned with the map reward.
    if getattr(map_cfg, "reward_fn", ""):
        merged_config.reward_fn = map_cfg.reward_fn
    merged_config.map_config_path = str(Path(map_config_file).resolve()) if map_config_file else ""
    merged_config.agent_config_path = str(Path(agent_config_file).resolve()) if agent_config_file else ""

    return {'map': map_cfg, 'agent': agent_cfg, 'merged': merged_config}


def get_configs_by_name(agent_name: str = "corridor", map_name: str = "cologne8") -> Dict[str, Any]:
    """Load ``configs/agents/<agent_name>.yaml`` and ``configs/maps/<map_name>.yaml``.

    A missing file falls back to the dataclass default for that side. The
    returned dict has ``map``, ``agent``, and ``merged``. Training uses
    ``merged``.
    """
    base = Path(__file__).parent / "configs"
    agent_path = base / "agents" / f"{agent_name}.yaml"
    map_path = base / "maps" / f"{map_name}.yaml"

    if map_path.exists():
        map_cfg = MapConfig.from_file(str(map_path))
    else:
        # fallback to default
        map_cfg = MapConfig()

    if agent_path.exists():
        agent_cfg = AgentConfig.from_file(str(agent_path))
    else:
        # fallback to default
        agent_cfg = AgentConfig()

    # Merge split configs.
    merged = {**map_cfg.to_dict(), **agent_cfg.to_dict()}
    # lambda_friction_coeff is map-owned: allow per-map tuning to override agent defaults.
    merged["lambda_friction_coeff"] = map_cfg.lambda_friction_coeff
    # from_dict filters unknown keys.
    merged_config = Config.from_dict(merged)
    # Keep corridor reward aligned with the map reward.
    if getattr(map_cfg, "reward_fn", ""):
        merged_config.reward_fn = map_cfg.reward_fn
    merged_config.map_config_path = str(map_path.resolve()) if map_path.exists() else ""
    merged_config.agent_config_path = str(agent_path.resolve()) if agent_path.exists() else ""
    return {'map': map_cfg, 'agent': agent_cfg, 'merged': merged_config}


def apply_runtime_logging_defaults(config: Config, map_name: Optional[str] = None, agent_name: Optional[str] = None) -> Config:
    """Fill in the run name and the ``results/`` directory.

    Called once per process from ``main.py``, and again with a suffix when
    ``--runs`` or ``--exp`` starts another condition. The directory is
    ``results/<map>/<agent>/<timestamp>``. A collision on the timestamp
    appends ``_02``, ``_03``, and so on. The default checkpoint directory
    ``checkpoints`` is redirected under that run directory so two runs
    launched from the same shell do not overwrite each other.
    """
    resolved_map = map_name or "cologne8"
    resolved_agent = agent_name or config.agent_type

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build a concise hyperparameter suffix.
    hparam_parts = []
    
    # Reward type.
    reward_fn = getattr(config, "reward_fn", "diff-waiting-time")
    hparam_parts.append(f"rwd-{reward_fn}")
    
    # PPO learning rate.
    ppo_lr = getattr(config, "ppo_lr", 3e-4)
    if abs(ppo_lr - 3e-4) < 1e-6:
        lr_str = "lr3e4"
    elif abs(ppo_lr - 1e-4) < 1e-6:
        lr_str = "lr1e4"
    elif abs(ppo_lr - 5e-4) < 1e-6:
        lr_str = "lr5e4"
    else:
        # Compact scientific notation, e.g. 3e-4 -> lr3e4.
        lr_sci = f"{ppo_lr:.0e}"
        lr_parts = lr_sci.split("e")
        if len(lr_parts) == 2:
            base = lr_parts[0].replace(".", "").replace("+", "")
            exp = lr_parts[1].replace("+", "").replace("-", "m")
            lr_str = f"lr{base}e{exp}"
        else:
            lr_str = f"lr{ppo_lr:.4f}".replace(".", "p")
    hparam_parts.append(lr_str)
    
    # Corridor reward decay.
    if not getattr(config, "disable_corridor_agent", False):
        corridor_decay = getattr(config, "corridor_reward_decay", 0.9)
        # Compact decay, e.g. 0.9 -> d9.
        decay_int = int(corridor_decay * 100)
        if decay_int == 90:
            decay_str = "d9"
        elif decay_int == 95:
            decay_str = "d95"
        else:
            decay_str = f"d{decay_int}"
        hparam_parts.append(decay_str)
    
    # GNN architecture.
    if not getattr(config, "disable_gnn", False):
        gnn_arch = getattr(config, "gnn_architecture", "GAT")
        hparam_parts.append(f"gnn-{gnn_arch}")
    
    # Final hyperparameter suffix.
    hparam_str = "_".join(hparam_parts) if hparam_parts else "default"
    
    # Full run name and output directory.
    run_name = f"{resolved_map}_{resolved_agent}_{hparam_str}_{ts}"
    run_dir = Path("results") / resolved_map / resolved_agent / ts
    if run_dir.exists():
        suffix = 2
        while True:
            candidate = Path("results") / resolved_map / resolved_agent / f"{ts}_{suffix:02d}"
            if not candidate.exists():
                run_name = f"{run_name}_{suffix:02d}"
                run_dir = candidate
                break
            suffix += 1

    config.wandb_run_name = run_name
    config.save_dir = str(run_dir)
    # Avoid sharing the default checkpoint directory across runs.
    try:
        if Path(str(config.checkpoint_save_dir)) == Path("checkpoints"):
            config.checkpoint_save_dir = str(Path(config.save_dir) / "checkpoints")
    except Exception:
        pass

    return config

