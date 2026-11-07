"""
Lower-level signal controller, one instance per SUMO traffic light.

The observation is the stock SUMO-RL vector. Graph embeddings never enter
this module. What the corridor agent sends down is a task: a boolean lane
mask and a preference per lane, packed as

    corridor_task_vec = [mask_1, ..., mask_n, pref_1, ..., pref_n]

in the signal's lane order.

Two heads read a shared backbone.

* The local head maps backbone features to phase logits. It is the policy
  the signal would run with no corridor at all.
* The cooperative head maps backbone features concatenated with the task
  vector to phase logits. It is only meaningful while this signal owns at
  least one masked lane.

The executed distribution mixes the logits with the paper's counterfactual-gain gate, not the probabilities:

    z = (1 - λ) * z_local + λ * z_coop
    λ = clip(Δ / ρ, 0, 1)
    Δ = (V_coop - V_local) / (|V_coop| + |V_local| + ε)

``ρ`` is ``lambda_friction_coeff``. Dividing by the sum of absolute values
keeps ``Δ`` on a comparable scale across signals whose critics have very
different magnitudes. ``λ`` is zero when the cooperative critic is not
ahead, when either critic is non-finite, when the corridor agent is
disabled, or when this signal has no active corridor. ``lambda_max_initial``
can cap ``λ`` below 1 while the cooperative head is still unreliable.

Two critics support the gate and the MGDA update. ``value_loc_head`` reads
the backbone. ``value_glob_head`` also reads the task vector, and optionally
a scalar grow-Q if ``include_grow_q_in_global_critic`` is set. The trainer
owns the optimizers. This class only selects and evaluates actions.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


class SharedBackbone(nn.Module):
    """Two-layer encoder shared by the local head, the cooperative head, and both critics.

    Sharing is what creates the gradient conflict MGDA is there to resolve.
    The local loss and the cooperative loss both flow into these weights.
    """
    
    def __init__(self, observation_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.backbone(observation)


class LocalHead(nn.Module):
    """Phase logits from the backbone features alone. This is ``π_local``."""
    
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class GlobalHead(nn.Module):
    """Phase logits conditioned on the corridor task. This is ``π_coop``.

    The task is concatenated with the backbone features, so a signal that
    is handed a different mask produces different logits without a second
    backbone.
    """
    
    def __init__(self, hidden_dim: int, action_dim: int, corridor_task_dim: int):
        super().__init__()
        # Input is [backbone features | lane mask | lane preferences].
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + corridor_task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        corridor_task_vec: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([features, corridor_task_vec], dim=-1)
        return self.head(combined)


class ValueHead(nn.Module):
    """Scalar critic. ``global_input_dim`` is 0 for the local critic.

    The cooperative critic sets it to the task width, plus one when the
    grow-Q feature is enabled.
    """
    
    def __init__(self, hidden_dim: int, global_input_dim: int = 0):
        super().__init__()
        input_dim = hidden_dim + global_input_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features: torch.Tensor, global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if global_features is not None:
            features = torch.cat([features, global_features], dim=-1)
        return self.head(features).squeeze(-1)


class IntersectionAgent:
    """One signal: local policy, cooperative policy, and the gate between them.

    Pass ``shared_modules`` to reuse another agent's backbone and heads.
    That is the ``--share`` ablation. Shared agents must agree on
    ``corridor_task_dim``, so the constructor pads or truncates ``lane_ids``
    until the task vector has that width. Padded lanes are named
    ``__pad_lane__*`` and never appear in a real mask.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        lane_feature_dim: int,
        config: Dict[str, Any],
        lane_ids: Optional[list[str]] = None,
        corridor_task_dim: Optional[int] = None,
        shared_modules: Optional[Dict[str, nn.Module]] = None
    ):
        """
        Args:
            observation_dim: Observation dimension.
            action_dim: Action dimension.
            lane_feature_dim: Lane feature dimension.
            config: Agent config.
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.lane_feature_dim = lane_feature_dim
        self.config = config
        self.lane_ids: list[str] = list(lane_ids) if lane_ids else []
        # Parameter sharing requires every signal to present the same task
        # width. Pad short signals and truncate long ones. The truncated
        # tail is a real limitation of sharing on irregular intersections.
        if corridor_task_dim is not None:
            target_dim = int(corridor_task_dim)
            if target_dim < 0:
                target_dim = 0
            target_lanes = int(target_dim // 2)
            if target_lanes > 0:
                if len(self.lane_ids) < target_lanes:
                    pad_n = target_lanes - len(self.lane_ids)
                    self.lane_ids = self.lane_ids + [f"__pad_lane__{i}" for i in range(pad_n)]
                elif len(self.lane_ids) > target_lanes:
                    self.lane_ids = self.lane_ids[:target_lanes]
            self.corridor_task_dim = int(2 * len(self.lane_ids))
        else:
            self.corridor_task_dim = int(2 * len(self.lane_ids))
        
        device_str = config.get("device", "cpu")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        
        hidden_dim = config.get("hidden_dim", 128)
        self.lambda_friction_coeff = config.get("lambda_friction_coeff", 1.0)
        
        # Off by default. When on, the cooperative critic also sees the
        # scalar Q that the grow policy assigned to this corridor.
        self.include_grow_q_in_global_critic = bool(config.get("include_grow_q_in_global_critic", False))
        critic_extra_dim = 1 if self.include_grow_q_in_global_critic else 0

        if shared_modules is not None:
            # Reuse shared modules.
            self.backbone = shared_modules["backbone"]
            self.local_head = shared_modules["local_head"]
            self.global_head = shared_modules["global_head"]
            self.value_loc_head = shared_modules["value_loc_head"]
            self.value_glob_head = shared_modules["value_glob_head"]
            # Follow the shared backbone device.
            try:
                self.device = next(self.backbone.parameters()).device
            except Exception:
                pass
        else:
            self.backbone = SharedBackbone(observation_dim, hidden_dim).to(self.device)
            self.local_head = LocalHead(hidden_dim, action_dim).to(self.device)
            self.global_head = GlobalHead(hidden_dim, action_dim, self.corridor_task_dim).to(self.device)
            self.value_loc_head = ValueHead(hidden_dim).to(self.device)
            self.value_glob_head = ValueHead(hidden_dim, global_input_dim=self.corridor_task_dim + critic_extra_dim).to(self.device)
        
        # Written by set_corridor_task on every replan, and read by the
        # cooperative head until the next one. An empty mask means "no
        # corridor", which forces λ = 0.
        self.preference_values = {}  # lane id -> preference in the task distribution
        self.lane_mask = {}  # lane id -> whether this corridor claims it
        self.corridor_q_value = 0.0
        
        self.disable_corridor_agent = config.get("disable_corridor_agent", False)
    
    def set_corridor_task(
        self,
        lane_mask: Dict[str, bool],
        preference_values: Dict[str, float],
        corridor_q_value: float = 0.0
    ):
        """Install the task produced by the latest corridor replan.

        Lanes absent from ``lane_mask`` are treated as unclaimed. Preferences
        absent from ``preference_values`` contribute 0 to the task vector.
        """
        self.lane_mask = lane_mask
        self.preference_values = preference_values
        try:
            self.corridor_q_value = float(corridor_q_value)
        except Exception:
            self.corridor_q_value = 0.0

    def _build_global_critic_features(self, lane_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Task vector, with the grow-Q scalar appended when that feature is on."""
        task_vec = self._build_corridor_task_vec(lane_features)
        if not getattr(self, "include_grow_q_in_global_critic", False):
            return task_vec
        v = float(getattr(self, "corridor_q_value", 0.0))
        if not np.isfinite(v):
            v = 0.0
        v_t = torch.tensor([v], dtype=task_vec.dtype, device=task_vec.device)
        return torch.cat([task_vec, v_t], dim=-1)

    def _build_corridor_task_vec(self, lane_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pack ``[mask_1..mask_n, pref_1..pref_n]`` in ``self.lane_ids`` order.

        Order is part of the contract. The cooperative head is a plain MLP,
        so a permuted lane list is a different task even when the set of
        claimed lanes is the same.
        """
        lane_ids = self.lane_ids
        if not lane_ids:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        mask_part = [1.0 if (self.lane_mask or {}).get(lid, False) else 0.0 for lid in lane_ids]
        pref_part = [float((self.preference_values or {}).get(lid, 0.0)) for lid in lane_ids]
        vec = mask_part + pref_part
        t = torch.tensor(vec, dtype=torch.float32, device=self.device)
        return t
    
    def compute_lambda(self, observation: torch.Tensor, lane_features: Dict[str, torch.Tensor]) -> float:
        """Closed-form gate. See the module docstring for the formula.

        The critics are evaluated under ``torch``'s current mode. Callers
        that only want an action should already be inside ``no_grad`` if
        they do not intend to train through ``λ``. The returned float is
        detached by ``.item()``.
        """
        backbone_features = self.backbone(observation)
        
        v_loc = self.value_loc_head(backbone_features)
        
        critic_features = self._build_global_critic_features(lane_features)
        v_glob = self.value_glob_head(backbone_features, critic_features)

        # A NaN critic would open or close the gate at random. Stay local.
        try:
            if not torch.isfinite(v_loc).all() or not torch.isfinite(v_glob).all():
                return 0.0
        except Exception:
            return 0.0
        
        # Normalize before dividing by ρ. Without this, a critic that
        # happens to live on a larger scale would pin λ at 1.
        eps = float(self.config.get("lambda_delta_eps", 1e-6))
        denom = torch.abs(v_glob) + torch.abs(v_loc) + eps
        delta = (v_glob - v_loc) / denom
        
        rho = float(getattr(self, "lambda_friction_coeff", 1.0))
        if rho <= 0.0:
            rho = 1.0
        lambda_star = torch.clamp(delta / rho, 0.0, 1.0).item()

        lambda_max_initial = self.config.get("lambda_max_initial", 1.0)
        if lambda_max_initial < 1.0:
            lambda_star = min(lambda_star, lambda_max_initial)
        
        return lambda_star
    
    def _weight_lane_features(self, lane_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean of lane features, weighted by ``(1 + preference)``.

        Lanes the current mask does not claim are left out. ``(1 + p)``
        rather than ``p`` keeps a claimed lane with preference 0 from
        disappearing. The result has width ``lane_feature_dim``. The
        training loop does not call this helper. The cooperative head reads
        the packed task vector, which keeps the mask and the preference as
        separate channels instead of folding them into the observation.
        """
        weighted_features = []
        for lane_id, features in lane_features.items():
            if self.lane_mask and not self.lane_mask.get(lane_id, False):
                continue
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            features = features.to(self.device)
            
            p_ell = self.preference_values.get(lane_id, 0.0)
            weighted = (1.0 + p_ell) * features
            weighted_features.append(weighted)
        
        if weighted_features:
            return torch.stack(weighted_features).mean(dim=0)
        else:
            return torch.zeros(self.lane_feature_dim, device=self.device)
    
    def select_action(
        self,
        observation: torch.Tensor,
        lane_features: Dict[str, torch.Tensor],
        training: bool = True,
        return_infos: bool = False,
        force_local_only: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """Sample a phase, or take the greedy phase when ``training`` is false.

        ``lane_features`` is the per-lane tensor dict used to build the task
        vector and the cooperative critic input. The local-only branch is
        taken when the corridor agent is disabled, when the caller forces
        it, or when the mask is empty. In every local-only branch the info
        dict reports ``lambda`` 0 and stores the local value in both critic
        fields, so a training buffer written from this step stays rectangular.

        With ``return_infos``, ``info`` also carries the log-probability of
        the mixed (or local) action and both critic values. Those tensors
        still require grad. The trainer detaches them when it stores the
        rollout.
        """
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.to(self.device)
        
        backbone_features = self.backbone(observation)
        
        local_logits = self.local_head(backbone_features)
        
        # Ablation, or a caller that wants the local policy on this step only.
        if self.disable_corridor_agent or force_local_only:
            dist = torch.distributions.Categorical(logits=local_logits)
            
            if training:
                action = dist.sample().item()
            else:
                action = torch.argmax(local_logits).item()
            
            value_loc = self.value_loc_head(backbone_features)
            
            info = {
                "lambda": 0.0,
                "local_logits": local_logits.detach().cpu().numpy(),
                "global_logits": None,
                "mixed_logits": local_logits.detach().cpu().numpy(),
                "mode": "single_objective" if self.disable_corridor_agent else "forced_local_only"
            }
            
            if return_infos:
                action_tensor = torch.tensor(action, device=self.device)
                log_prob = dist.log_prob(action_tensor)
                info["log_prob"] = log_prob
                info["value_loc"] = value_loc
                info["value_glob"] = value_loc
            
            return action, info
        
        # A signal the latest replan did not claim must not invent a task.
        if (not self.lane_mask) or (not any(self.lane_mask.values())):
            dist = torch.distributions.Categorical(logits=local_logits)
            if training:
                action = dist.sample().item()
            else:
                action = torch.argmax(local_logits).item()
            value_loc = self.value_loc_head(backbone_features)
            info = {
                "lambda": 0.0,
                "local_logits": local_logits.detach().cpu().numpy(),
                "global_logits": None,
                "mixed_logits": local_logits.detach().cpu().numpy(),
                "mode": "forced_local_only_no_corridor"
            }
            if return_infos:
                action_tensor = torch.tensor(action, device=self.device)
                log_prob = dist.log_prob(action_tensor)
                info["log_prob"] = log_prob
                info["value_loc"] = value_loc
                info["value_glob"] = value_loc
            return action, info
        
        lambda_star = self.compute_lambda(observation, lane_features)
        # compute_lambda already clamps, but a bad config can still hand
        # back NaN. The mixture is undefined in that case, so stay local.
        try:
            if not np.isfinite(float(lambda_star)):
                lambda_star = 0.0
        except Exception:
            lambda_star = 0.0
        lambda_star = float(max(0.0, min(1.0, float(lambda_star))))
        
        task_vec = self._build_corridor_task_vec(lane_features)
        # select_action is called on one observation. A task vector that
        # arrived with a batch axis of 1 would make the concat fail.
        try:
            if backbone_features.dim() == 1 and task_vec.dim() == 2 and task_vec.shape[0] == 1:
                task_vec = task_vec.squeeze(0)
            elif backbone_features.dim() == 2 and task_vec.dim() == 1:
                task_vec = task_vec.unsqueeze(0)
        except Exception:
            pass
        global_logits = self.global_head(backbone_features, task_vec)
        
        # Mix logits, then sanitize again. A single inf logit makes
        # Categorical produce NaN probabilities and a silent bad action.
        local_logits = torch.nan_to_num(local_logits, nan=0.0, posinf=0.0, neginf=0.0)
        global_logits = torch.nan_to_num(global_logits, nan=0.0, posinf=0.0, neginf=0.0)
        mixed_logits = (1 - lambda_star) * local_logits + lambda_star * global_logits
        mixed_logits = torch.nan_to_num(mixed_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(mixed_logits).all():
            mixed_logits = torch.zeros_like(local_logits)
        
        dist = torch.distributions.Categorical(logits=mixed_logits)
        if training:
            action = dist.sample().item()
        else:
            action = torch.argmax(mixed_logits).item()
        
        info = {
            "lambda": lambda_star,
            "local_logits": local_logits.detach().cpu().numpy(),
            "global_logits": global_logits.detach().cpu().numpy(),
            "mixed_logits": mixed_logits.detach().cpu().numpy(),
            "mode": "multi_objective"
        }
        
        if return_infos:
            action_tensor = torch.tensor(action, device=self.device)
            log_prob = dist.log_prob(action_tensor)
            value_loc = self.value_loc_head(backbone_features)
            value_glob = self.value_glob_head(backbone_features, self._build_global_critic_features(lane_features))
            info["log_prob"] = log_prob
            info["value_loc"] = value_loc
            info["value_glob"] = value_glob
        
        return action, info
    
    def evaluate_action(
        self,
        observation: torch.Tensor,
        action: int,
        lane_features: Dict[str, torch.Tensor],
        force_local_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probability and both critic values for an action already taken.

        The log-probability is computed under the same mixture that
        ``select_action`` sampled from, with ``λ`` recomputed from the
        current critics. The trainer's PPO update does not call this method.
        It rebuilds the mixture in batch form so one backward pass covers
        the whole episode. This method is the single-step equivalent, used
        when a caller needs the same numbers outside that loop.
        """
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.to(self.device)
        
        backbone_features = self.backbone(observation)
        
        local_logits = self.local_head(backbone_features)
        local_dist = torch.distributions.Categorical(logits=local_logits)
        action_tensor = torch.tensor(action, device=self.device)
        log_prob_loc = local_dist.log_prob(action_tensor)
        
        value_loc = self.value_loc_head(backbone_features)
        
        if self.disable_corridor_agent or force_local_only:
            return log_prob_loc, value_loc, value_loc
        
        # Global policy logits.
        task_vec = self._build_corridor_task_vec(lane_features)
        try:
            if backbone_features.dim() == 1 and task_vec.dim() == 2 and task_vec.shape[0] == 1:
                task_vec = task_vec.squeeze(0)
            elif backbone_features.dim() == 2 and task_vec.dim() == 1:
                task_vec = task_vec.unsqueeze(0)
        except Exception:
            pass
        global_logits = self.global_head(backbone_features, task_vec)
        global_dist = torch.distributions.Categorical(logits=global_logits)
        log_prob_glob = global_dist.log_prob(action_tensor)
        
        value_glob = self.value_glob_head(backbone_features, self._build_global_critic_features(lane_features))
        
        # Match select_action: the stored behavior policy is the mixture,
        # so the importance ratio has to be computed on the mixed logits.
        lambda_star = self.compute_lambda(observation, lane_features)
        mixed_logits = (1 - lambda_star) * local_logits + lambda_star * global_logits
        mixed_dist = torch.distributions.Categorical(logits=mixed_logits)
        log_prob_mixed = mixed_dist.log_prob(action_tensor)
        
        return log_prob_mixed, value_loc, value_glob
    
    def save(self, filepath: str) -> None:
        """Write every head, the observation sizes, and the config used to build them."""
        import torch
        checkpoint = {
            'backbone_state_dict': self.backbone.state_dict(),
            'local_head_state_dict': self.local_head.state_dict(),
            'global_head_state_dict': self.global_head.state_dict(),
            'value_loc_head_state_dict': self.value_loc_head.state_dict(),
            'value_glob_head_state_dict': self.value_glob_head.state_dict(),
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'lane_feature_dim': self.lane_feature_dim,
            'config': self.config,
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str) -> None:
        """Load a checkpoint, skipping tensors whose name or shape no longer matches.

        A task-width change between the checkpoint and this process used to
        abort the whole load. Skipping the mismatched tensors keeps the
        heads that still fit and prints how many were dropped.
        """
        import torch
        safe_np_globals = [np.core.multiarray.scalar, np.dtype, np.dtypes.Int64DType]
        with torch.serialization.safe_globals(safe_np_globals):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

        def _safe_load(module: nn.Module, saved_state: Dict[str, torch.Tensor], module_name: str) -> None:
            current_state = module.state_dict()
            filtered_state: Dict[str, torch.Tensor] = {}
            skipped_keys = []
            for key, tensor in (saved_state or {}).items():
                if key not in current_state:
                    skipped_keys.append(key)
                    continue
                if current_state[key].shape != tensor.shape:
                    skipped_keys.append(key)
                    continue
                filtered_state[key] = tensor
            missing, unexpected = module.load_state_dict(filtered_state, strict=False)
            if skipped_keys or missing or unexpected:
                print(
                    f"Warning: partial load for {module_name}. "
                    f"loaded={len(filtered_state)} skipped={len(skipped_keys)} "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )

        _safe_load(self.backbone, checkpoint.get('backbone_state_dict', {}), "backbone")
        _safe_load(self.local_head, checkpoint.get('local_head_state_dict', {}), "local_head")
        _safe_load(self.global_head, checkpoint.get('global_head_state_dict', {}), "global_head")
        _safe_load(self.value_loc_head, checkpoint.get('value_loc_head_state_dict', {}), "value_loc_head")
        _safe_load(self.value_glob_head, checkpoint.get('value_glob_head_state_dict', {}), "value_glob_head")

