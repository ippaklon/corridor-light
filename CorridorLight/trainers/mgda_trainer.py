"""
Multi-objective PPO for one intersection agent.

The local return and the cooperative return both depend on actions drawn
from the mixed policy, and both losses land on the shared backbone. Updating
the backbone with either loss alone lets one objective erase the features
the other one needs. MGDA picks the convex combination of the two backbone
gradients with minimum norm:

    w* = clip( (g_coop - g_local) · g_coop / ||g_coop - g_local||^2 , 0, 1 )
    L  = w* * L_local + (1 - w*) * L_coop

``w*`` is computed once per PPO epoch from the current backbone gradients
and then held fixed. The step itself is three updates:

1. Shared backbone, on ``L`` above.
2. Local head and cooperative head, one at a time, with the backbone
   detached. Each head is scored by the log-probability of the *mixed*
   behavior policy, with the other head's logits treated as constant, so
   a head is only credited for the part of the mixture it actually owns.
3. Local critic on every sample, cooperative critic only on samples whose
   ``in_corridor`` flag is set. A signal that was not inside a corridor
   has no cooperative target.

When the corridor agent is disabled, or the rollout contains no corridor
samples, all of the above collapses to ordinary clipped PPO on the local
head, the local critic, and the backbone.

The importance ratio always uses the log-probability of the mixture that
was sampled at action time. ``λ`` is the value stored in the rollout. If a
caller did not store it, the ratio falls back to
``clip((A_coop - A_local) / ρ, 0, 1)``, which is the same gate with
advantages in place of critic values.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from ..agents.intersection_agent import IntersectionAgent


class MGDATrainer:
    """PPO update for one ``IntersectionAgent``, with MGDA on its backbone.

    Optimizers may be injected. The trainer does that when
    ``share_intersection_parameters`` is on, so every signal steps the same
    Adam state. Otherwise this class builds one Adam optimizer per module.
    """
    
    def __init__(
        self,
        agent: IntersectionAgent,
        config: Dict[str, Any],
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None
    ):
        self.agent = agent
        self.config = config
        
        self.device = agent.device
        
        self.ppo_lr = config.get("ppo_lr", 3e-4)
        self.ppo_gamma = config.get("ppo_gamma", 0.99)
        self.ppo_clip_ratio = config.get("ppo_clip_ratio", 0.2)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.mgda_epsilon = config.get("mgda_epsilon", 1e-8)
        
        # Shared optimizers come from the trainer when several signals reuse
        # one set of modules. Each MGDATrainer still runs its own loss.
        if optimizers is not None:
            self.backbone_optimizer = optimizers["backbone"]
            self.local_head_optimizer = optimizers["local_head"]
            self.global_head_optimizer = optimizers["global_head"]
            self.value_loc_optimizer = optimizers["value_loc"]
            self.value_glob_optimizer = optimizers["value_glob"]
        else:
            self.backbone_optimizer = torch.optim.Adam(
                self.agent.backbone.parameters(),
                lr=self.ppo_lr
            )
            self.local_head_optimizer = torch.optim.Adam(
                self.agent.local_head.parameters(),
                lr=self.ppo_lr
            )
            self.global_head_optimizer = torch.optim.Adam(
                self.agent.global_head.parameters(),
                lr=self.ppo_lr
            )
            self.value_loc_optimizer = torch.optim.Adam(
                self.agent.value_loc_head.parameters(),
                lr=self.ppo_lr
            )
            self.value_glob_optimizer = torch.optim.Adam(
                self.agent.value_glob_head.parameters(),
                lr=self.ppo_lr
            )
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE(γ, λ) over one episode.

        The bootstrap value after the last step is 0. ``dones[t]`` drops the
        bootstrap, which matters on the final step and is a no-op before it
        because SUMO-RL only reports done when the episode ends. Advantages
        and returns are tensors on the agent device.
        """
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values_t = torch.tensor(values, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Bootstrap value
        values_t = torch.cat([values_t, torch.zeros(1, device=self.device)], dim=0)
        
        T = rewards_t.shape[0]
        advantages_t = torch.zeros(T, device=self.device)
        gae = torch.tensor(0.0, device=self.device)
        
        for t in reversed(range(T)):
            mask = 1.0 - dones_t[t]
            delta = rewards_t[t] + gamma * values_t[t + 1] * mask - values_t[t]
            gae = delta + gamma * lam * mask * gae
            advantages_t[t] = gae
        
        returns_t = advantages_t + values_t[:-1]
        
        return advantages_t, returns_t
    
    def compute_mgda_weight(
        self,
        grad_loc: torch.Tensor,
        grad_glob: torch.Tensor
    ) -> float:
        """Minimum-norm convex weight of the local backbone gradient.

        ``w* = 1`` steps the backbone with the local loss only. ``w* = 0``
        steps it with the cooperative loss only. A vanishing denominator
        means the two gradients are identical, and the weight falls back
        to 0.5 so neither objective is dropped on a tie.
        """
        diff = grad_glob - grad_loc
        numerator = torch.dot(diff, grad_glob)
        denominator = torch.dot(diff, diff) + self.mgda_epsilon
        
        if denominator < self.mgda_epsilon:
            w_star = 0.5
        else:
            w_star = (numerator / denominator).item()
            w_star = max(0.0, min(1.0, w_star))
        
        return w_star
    
    def update(self, rollout_data: Dict[str, Any]):
        """One PPO update from a single-agent rollout dict.

        Required keys are ``states``, ``actions``, ``rewards_loc``,
        ``rewards_glob``, ``values_loc``, ``values_glob``, ``dones``, and
        ``old_log_probs``. ``in_corridor`` masks the cooperative loss.
        ``lambdas`` is the gate used when the action was sampled.
        ``corridor_task_vec`` rebuilds the cooperative head's input.
        Missing optional keys are filled with zeros or with the local
        stream, which is the local-only update.
        """
        disable_corridor_agent = self.agent.disable_corridor_agent
        
        # Convert through NumPy for faster tensor creation.
        states_array = np.array(rollout_data['states'], dtype=np.float32)
        states = torch.tensor(states_array, dtype=torch.float32).to(self.device)
        
        actions_array = np.array(rollout_data['actions'], dtype=np.int64)
        actions = torch.tensor(actions_array, dtype=torch.long).to(self.device)
        
        rewards_loc = rollout_data['rewards_loc']
        rewards_glob = rollout_data['rewards_glob'] if not disable_corridor_agent else rewards_loc
        values_loc_old = rollout_data['values_loc']
        values_glob_old = rollout_data['values_glob'] if not disable_corridor_agent else values_loc_old
        dones = rollout_data['dones']
        
        # Convert old log-probs in one batch.
        old_log_probs_array = np.array(rollout_data.get('old_log_probs', [0.0] * len(states)), dtype=np.float32)
        old_log_probs = torch.tensor(old_log_probs_array, dtype=torch.float32).to(self.device)
        
        lambdas_list = rollout_data.get('lambdas', None)
        lambdas_t: Optional[torch.Tensor] = None
        try:
            if isinstance(lambdas_list, (list, tuple)) and len(lambdas_list) == len(states):
                lambdas_t = torch.tensor(np.array(lambdas_list, dtype=np.float32), dtype=torch.float32, device=self.device)
        except Exception:
            lambdas_t = None
        
        lane_features_list = rollout_data.get('lane_features', [{}] * len(states))
        corridor_q_observations_list = rollout_data.get('corridor_q_observations', [0.0] * len(states))
        in_corridor_list = rollout_data.get('in_corridor', [True] * len(states))
        corridor_task_vec_list = rollout_data.get('corridor_task_vec', None)
        try:
            in_corridor_mask = torch.tensor(np.array(in_corridor_list, dtype=np.float32), dtype=torch.float32, device=self.device)
        except Exception:
            in_corridor_mask = torch.ones(len(states), dtype=torch.float32, device=self.device)
        
        # An episode in which this signal was never inside a corridor has
        # nothing for the cooperative head to fit. Train the local policy.
        try:
            if (not disable_corridor_agent) and (in_corridor_mask.numel() > 0) and (not torch.any(in_corridor_mask > 0.5)):
                disable_corridor_agent = True
                rewards_glob = rewards_loc
                values_glob_old = values_loc_old
        except Exception:
            pass
        
        # Local advantages and returns.
        advantages_loc, returns_loc = self.compute_gae(
            rewards_loc, values_loc_old, dones,
            gamma=self.ppo_gamma, lam=self.config.get("ppo_lam", 0.95)
        )
        
        # Normalize advantages once before PPO epochs.
        advantages_loc_mean = advantages_loc.mean()
        # Avoid NaN std for tiny batches.
        if advantages_loc.numel() <= 1:
            advantages_loc_std = torch.zeros_like(advantages_loc_mean)
        else:
            advantages_loc_std = advantages_loc.std(unbiased=False)
        try:
            if not torch.isfinite(advantages_loc_std).all():
                advantages_loc_std = torch.zeros_like(advantages_loc_mean)
        except Exception:
            advantages_loc_std = torch.zeros_like(advantages_loc_mean)
        if advantages_loc_std > 1e-8:
            advantages_loc = (advantages_loc - advantages_loc_mean) / (advantages_loc_std + 1e-8)
        else:
            advantages_loc = advantages_loc - advantages_loc_mean
        advantages_loc = advantages_loc.to(self.device)
        # Remove non-finite values.
        advantages_loc = torch.nan_to_num(advantages_loc, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reuse tensor returns safely.
        if isinstance(returns_loc, torch.Tensor):
            returns_loc = returns_loc.detach().clone().to(self.device)
        else:
            returns_loc = torch.tensor(returns_loc, dtype=torch.float32).to(self.device)
        
        # Single-objective PPO. The cooperative head and critic are not stepped.
        if disable_corridor_agent:
            # old_log_probs must stay fixed during PPO updates.
            
            value_coef = self.config.get("ppo_value_coef", 0.5)
            entropy_coef = self.config.get("ppo_entropy_coef", 0.01)
            max_grad_norm = self.config.get("ppo_max_grad_norm", 0.5)
            
            total_policy_loss = 0.0
            total_value_loss = 0.0
            
            for epoch in range(self.ppo_epochs):
                # Batched forward pass.
                backbone_features = self.agent.backbone(states)
                
                local_logits = self.agent.local_head(backbone_features)
                local_dist = torch.distributions.Categorical(logits=local_logits)
                log_probs_loc = local_dist.log_prob(actions)
                
                values_loc_new = self.agent.value_loc_head(backbone_features)
                # Keep batch dimension stable.
                if values_loc_new.dim() > 1:
                    values_loc_new = values_loc_new.squeeze(-1)
                if values_loc_new.dim() == 0:
                    values_loc_new = values_loc_new.unsqueeze(0)
                
                # Importance ratio.
                ratio_loc = torch.exp(torch.clamp(log_probs_loc - old_log_probs, -10.0, 10.0))
                
                # PPO clipped surrogate loss.
                surr1_loc = ratio_loc * advantages_loc
                surr2_loc = torch.clamp(ratio_loc, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio) * advantages_loc
                policy_loss_loc = -torch.min(surr1_loc, surr2_loc).mean()
                
                # Skip invalid losses.
                if torch.isnan(policy_loss_loc) or torch.isinf(policy_loss_loc):
                    print(f"⚠️ Warning: policy_loss_loc is NaN/Inf, skipping update")
                    continue
                
                # Value loss.
                if values_loc_new.shape != returns_loc.shape:
                    if values_loc_new.dim() == 0:
                        values_loc_new = values_loc_new.unsqueeze(0)
                    if returns_loc.dim() == 0:
                        returns_loc = returns_loc.unsqueeze(0)
                value_loss_loc = F.mse_loss(values_loc_new, returns_loc)
                
                # Skip invalid losses.
                if torch.isnan(value_loss_loc) or torch.isinf(value_loss_loc):
                    print(f"⚠️ Warning: value_loss_loc is NaN/Inf, skipping update")
                    continue
                
                entropy_loc = local_dist.entropy().mean()
                
                # Backward policy and value losses separately.
                
                self.backbone_optimizer.zero_grad()
                self.local_head_optimizer.zero_grad()
                self.value_loc_optimizer.zero_grad()
                
                policy_loss_with_entropy = policy_loss_loc - entropy_coef * entropy_loc
                policy_loss_with_entropy.backward(retain_graph=True)
                
                value_loss_weighted = value_coef * value_loss_loc
                value_loss_weighted.backward()
                
                backbone_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.backbone.parameters(), max_grad_norm)
                local_head_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.local_head.parameters(), max_grad_norm)
                value_head_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.value_loc_head.parameters(), max_grad_norm)
                
                if torch.isnan(backbone_grad_norm) or torch.isnan(local_head_grad_norm) or torch.isnan(value_head_grad_norm):
                    print(f"⚠️ Warning: Gradient norm is NaN, skipping update")
                    continue
                
                # Warn on near-zero gradients.
                if backbone_grad_norm < 1e-8 and local_head_grad_norm < 1e-8 and value_head_grad_norm < 1e-8:
                    print(f"⚠️ Warning: All gradient norms are near zero, model may not be training")
                
                self.backbone_optimizer.step()
                self.local_head_optimizer.step()
                self.value_loc_optimizer.step()
                
                total_policy_loss += policy_loss_loc.item()
                total_value_loss += value_loss_loc.item()
                
                # Keep old_log_probs fixed across epochs.
            
            avg_policy_loss = total_policy_loss / self.ppo_epochs
            avg_value_loss = total_value_loss / self.ppo_epochs
            
            self._last_policy_loss = avg_policy_loss
            self._last_value_loss = avg_value_loss
            
            if not hasattr(self, '_last_backbone_params'):
                self._last_backbone_params = [p.clone().detach() for p in self.agent.backbone.parameters()]
            else:
                for old_p, new_p in zip(self._last_backbone_params, self.agent.backbone.parameters()):
                    if not torch.allclose(old_p, new_p, atol=1e-6):
                        break
                self._last_backbone_params = [p.clone().detach() for p in self.agent.backbone.parameters()]
            
            return
        
        # Two-objective path. Cooperative advantages are normalized on the
        # in-corridor samples only, then multiplied by the same mask so a
        # signal that was briefly outside a corridor does not contribute a
        # target of zero to the cooperative loss.
        advantages_glob, returns_glob = self.compute_gae(
            rewards_glob, values_glob_old, dones,
            gamma=self.ppo_gamma, lam=self.config.get("ppo_lam", 0.95)
        )
        
        # Normalize global advantages on corridor samples only.
        advantages_glob = advantages_glob.to(self.device)
        valid = in_corridor_mask > 0.5
        if torch.any(valid):
            adv_v = advantages_glob[valid]
            adv_mean = adv_v.mean()
            # Small-batch-safe std.
            if adv_v.numel() <= 1:
                adv_std = torch.zeros_like(adv_mean)
            else:
                adv_std = adv_v.std(unbiased=False)
            try:
                if not torch.isfinite(adv_std).all():
                    adv_std = torch.zeros_like(adv_mean)
            except Exception:
                adv_std = torch.zeros_like(adv_mean)
            if adv_std > 1e-8:
                advantages_glob = (advantages_glob - adv_mean) / (adv_std + 1e-8)
            else:
                advantages_glob = advantages_glob - adv_mean
        else:
            advantages_glob = torch.zeros_like(advantages_glob)
        # Remove non-finite values.
        advantages_glob = torch.nan_to_num(advantages_glob, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reuse tensor returns safely.
        if isinstance(returns_glob, torch.Tensor):
            returns_glob = returns_glob.detach().clone().to(self.device)
        else:
            returns_glob = torch.tensor(returns_glob, dtype=torch.float32).to(self.device)
        
        # Mask non-corridor samples for global training.
        try:
            advantages_glob = advantages_glob * in_corridor_mask
            returns_glob = returns_glob * in_corridor_mask
        except Exception:
            pass
        
        # Corridor task vector: masks + preferences.
        task_dim = int(getattr(self.agent, "corridor_task_dim", 0))
        if isinstance(corridor_task_vec_list, (list, tuple)) and len(corridor_task_vec_list) == len(states) and task_dim > 0:
            rows = []
            for v in corridor_task_vec_list:
                vv = list(v) if isinstance(v, (list, tuple)) else []
                if len(vv) < task_dim:
                    vv = vv + [0.0] * (task_dim - len(vv))
                elif len(vv) > task_dim:
                    vv = vv[:task_dim]
                rows.append(vv)
            corridor_task_vec_batch = torch.tensor(np.array(rows, dtype=np.float32), dtype=torch.float32, device=self.device)
        else:
            corridor_task_vec_batch = torch.zeros(len(states), task_dim, device=self.device) if task_dim > 0 else torch.zeros(len(states), 0, device=self.device)

        # Optional grow Q scalar for the global critic.
        if getattr(self.agent, "include_grow_q_in_global_critic", False):
            if len(corridor_q_observations_list) != len(states):
                corridor_q_observations_list = [0.0] * len(states)
            q_t = torch.tensor(corridor_q_observations_list, dtype=corridor_task_vec_batch.dtype, device=self.device).view(-1, 1)
            critic_features_batch = torch.cat([corridor_task_vec_batch, q_t], dim=-1)
        else:
            critic_features_batch = corridor_task_vec_batch
        
        # Cache config scalars.
        value_coef = self.config.get("ppo_value_coef", 0.5)
        entropy_coef = self.config.get("ppo_entropy_coef", 0.01)
        max_grad_norm = self.config.get("ppo_max_grad_norm", 0.5)
        debug_checks = self.config.get("debug_checks", True)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        backbone_params = [p for p in self.agent.backbone.parameters() if p.requires_grad]
        for epoch in range(self.ppo_epochs):
            # Batched current policy and values.
            backbone_features = self.agent.backbone(states)
            
            local_logits = self.agent.local_head(backbone_features)
            
            global_logits = self.agent.global_head(backbone_features, corridor_task_vec_batch)
            
            values_loc_new = self.agent.value_loc_head(backbone_features)
            values_glob_new = self.agent.value_glob_head(backbone_features, critic_features_batch)
            # Keep batch dimensions stable.
            if values_loc_new.dim() > 1:
                values_loc_new = values_loc_new.squeeze(-1)
            if values_glob_new.dim() > 1:
                values_glob_new = values_glob_new.squeeze(-1)
            if values_loc_new.dim() == 0:
                values_loc_new = values_loc_new.unsqueeze(0)
            if values_glob_new.dim() == 0:
                values_glob_new = values_glob_new.unsqueeze(0)

            # The behavior policy is the mixture. Using either head's
            # log-probability here would make the PPO ratio refer to a
            # policy that was not the one sampled.
            if lambdas_t is not None:
                lambda_star = torch.clamp(lambdas_t, 0.0, 1.0).detach()
            else:
                # No stored gate. Reconstruct one from the advantage gap,
                # which is the same closed form with critics replaced by
                # the returns they are being trained toward.
                rho = float(getattr(self.agent, "lambda_friction_coeff", 1.0))
                if rho <= 0.0:
                    rho = 1.0
                delta = (advantages_glob - advantages_loc).detach()
                lambda_star = torch.clamp(delta / rho, 0.0, 1.0).detach()
            lambda_max_initial = float(self.config.get("lambda_max_initial", 1.0))
            if lambda_max_initial < 1.0:
                lambda_star = torch.clamp(lambda_star, 0.0, lambda_max_initial)
            mixed_logits = (1.0 - lambda_star).unsqueeze(-1) * local_logits + lambda_star.unsqueeze(-1) * global_logits
            mixed_dist = torch.distributions.Categorical(logits=mixed_logits)
            log_probs_mixed = mixed_dist.log_prob(actions)
            
            # Importance ratio.
            ratio = torch.exp(torch.clamp(log_probs_mixed - old_log_probs, -10.0, 10.0))
            
            # PPO clipped surrogate losses.
            surr1_loc = ratio * advantages_loc
            surr2_loc = torch.clamp(ratio, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio) * advantages_loc
            policy_loss_loc = -torch.min(surr1_loc, surr2_loc).mean()
            
            surr1_glob = ratio * advantages_glob
            surr2_glob = torch.clamp(ratio, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio) * advantages_glob
            policy_loss_glob = -torch.min(surr1_glob, surr2_glob).mean()
            
            # Value losses.
            value_loss_loc = F.mse_loss(values_loc_new, returns_loc)
            # Fit the global critic only on corridor samples.
            if torch.any(in_corridor_mask > 0.5):
                m = in_corridor_mask > 0.5
                value_loss_glob = F.mse_loss(values_glob_new[m], returns_glob[m])
            else:
                value_loss_glob = torch.zeros((), device=self.device, requires_grad=True)
            
            entropy_mixed = mixed_dist.entropy().mean()
            
            # Total losses.
            loss_loc = policy_loss_loc + value_coef * value_loss_loc - entropy_coef * entropy_mixed
            loss_glob = policy_loss_glob + value_coef * value_loss_glob - entropy_coef * entropy_mixed
            
            if debug_checks:
                if (not torch.isfinite(loss_loc)) or (not torch.isfinite(loss_glob)):
                    print("⚠️ Warning: loss is NaN/Inf, skipping update")
                    continue
                if (not torch.isfinite(policy_loss_loc)) or (not torch.isfinite(policy_loss_glob)):
                    print("⚠️ Warning: policy loss is NaN/Inf, skipping update")
                    continue
                if (not torch.isfinite(value_loss_loc)) or (not torch.isfinite(value_loss_glob)):
                    print("⚠️ Warning: value loss is NaN/Inf, skipping update")
                    continue
            
            # Backbone gradients of each full loss, flattened and concatenated
            # in parameter order. Heads are not part of this inner product.
            # Their parameters are updated on their own just below.
            grad_loc_list = torch.autograd.grad(
                loss_loc,
                backbone_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            grad_loc = torch.cat([g.flatten() for g in grad_loc_list if g is not None])
            if debug_checks and (grad_loc.numel() == 0 or not torch.isfinite(grad_loc).all()):
                print("⚠️ Warning: grad_loc is empty or NaN/Inf, skipping update")
                continue
            
            grad_glob_list = torch.autograd.grad(
                loss_glob,
                backbone_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
            grad_glob = torch.cat([g.flatten() for g in grad_glob_list if g is not None])
            if debug_checks and (grad_glob.numel() == 0 or not torch.isfinite(grad_glob).all()):
                print("⚠️ Warning: grad_glob is empty or NaN/Inf, skipping update")
                continue
            
            w_star = self.compute_mgda_weight(grad_loc, grad_glob)
            
            # Recompute after parameter updates.
            backbone_features_new = self.agent.backbone(states)
            local_logits_new = self.agent.local_head(backbone_features_new)
            global_logits_new = self.agent.global_head(backbone_features_new, corridor_task_vec_batch)
            
            values_loc_new_recompute = self.agent.value_loc_head(backbone_features_new)
            values_glob_new_recompute = self.agent.value_glob_head(backbone_features_new, critic_features_batch)
            # Keep batch dimensions stable.
            if values_loc_new_recompute.dim() > 1:
                values_loc_new_recompute = values_loc_new_recompute.squeeze(-1)
            if values_glob_new_recompute.dim() > 1:
                values_glob_new_recompute = values_glob_new_recompute.squeeze(-1)
            if values_loc_new_recompute.dim() == 0:
                values_loc_new_recompute = values_loc_new_recompute.unsqueeze(0)
            if values_glob_new_recompute.dim() == 0:
                values_glob_new_recompute = values_glob_new_recompute.unsqueeze(0)

            # Recompute mixed-policy ratio.
            if lambdas_t is not None:
                lambda_star_new = torch.clamp(lambdas_t, 0.0, 1.0).detach()
            else:
                rho_new = float(getattr(self.agent, "lambda_friction_coeff", 1.0))
                if rho_new <= 0.0:
                    rho_new = 1.0
                delta = (advantages_glob - advantages_loc).detach()
                lambda_star_new = torch.clamp(delta / rho_new, 0.0, 1.0).detach()
            lambda_max_initial_new = float(self.config.get("lambda_max_initial", 1.0))
            if lambda_max_initial_new < 1.0:
                lambda_star_new = torch.clamp(lambda_star_new, 0.0, lambda_max_initial_new)
            mixed_logits_new = (1.0 - lambda_star_new).unsqueeze(-1) * local_logits_new + lambda_star_new.unsqueeze(-1) * global_logits_new
            mixed_dist_new = torch.distributions.Categorical(logits=mixed_logits_new)
            log_probs_mixed_new = mixed_dist_new.log_prob(actions)
            ratio_new = torch.exp(torch.clamp(log_probs_mixed_new - old_log_probs, -10.0, 10.0))
            surr1_loc_new = ratio_new * advantages_loc
            surr2_loc_new = torch.clamp(ratio_new, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio) * advantages_loc
            policy_loss_loc_new = -torch.min(surr1_loc_new, surr2_loc_new).mean()
            surr1_glob_new = ratio_new * advantages_glob
            surr2_glob_new = torch.clamp(ratio_new, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio) * advantages_glob
            policy_loss_glob_new = -torch.min(surr1_glob_new, surr2_glob_new).mean()
            
            value_loss_loc_new = F.mse_loss(values_loc_new_recompute, returns_loc)
            if torch.any(in_corridor_mask > 0.5):
                m = in_corridor_mask > 0.5
                value_loss_glob_new = F.mse_loss(values_glob_new_recompute[m], returns_glob[m])
            else:
                value_loss_glob_new = torch.zeros((), device=self.device, requires_grad=True)
            entropy_mixed_new = mixed_dist_new.entropy().mean()
            
            loss_loc_new = policy_loss_loc_new + value_coef * value_loss_loc_new - entropy_coef * entropy_mixed_new
            loss_glob_new = policy_loss_glob_new + value_coef * value_loss_glob_new - entropy_coef * entropy_mixed_new
            
            # The losses are recomputed after w* is known so the backward
            # that steps the backbone is a single combined graph, not the
            # graphs that were already consumed to estimate the gradients.
            self.backbone_optimizer.zero_grad()
            combined_loss = w_star * loss_loc_new + (1 - w_star) * loss_glob_new
            if debug_checks and (not torch.isfinite(combined_loss)):
                print("⚠️ Warning: combined_loss is NaN/Inf, skipping update")
                self.backbone_optimizer.zero_grad()
                continue
            combined_loss.backward()
            backbone_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.backbone.parameters(), max_grad_norm)
            if debug_checks and (not torch.isfinite(backbone_grad_norm)):
                print("⚠️ Warning: backbone grad norm is NaN/Inf, skipping update")
                self.backbone_optimizer.zero_grad()
                continue
            self.backbone_optimizer.step()
            
            # Detach backbone for per-head updates.
            backbone_features_final = self.agent.backbone(states).detach()

            # Freeze lambda for head updates.
            values_loc_final_for_lambda = self.agent.value_loc_head(backbone_features_final)
            values_glob_final_for_lambda = self.agent.value_glob_head(backbone_features_final, critic_features_batch)
            if values_loc_final_for_lambda.dim() > 1:
                values_loc_final_for_lambda = values_loc_final_for_lambda.squeeze(-1)
            if values_glob_final_for_lambda.dim() > 1:
                values_glob_final_for_lambda = values_glob_final_for_lambda.squeeze(-1)
            if values_loc_final_for_lambda.dim() == 0:
                values_loc_final_for_lambda = values_loc_final_for_lambda.unsqueeze(0)
            if values_glob_final_for_lambda.dim() == 0:
                values_glob_final_for_lambda = values_glob_final_for_lambda.unsqueeze(0)
            if lambdas_t is not None:
                lambda_star_final = torch.clamp(lambdas_t, 0.0, 1.0).detach()
            else:
                rho_final = float(getattr(self.agent, "lambda_friction_coeff", 1.0))
                if rho_final <= 0.0:
                    rho_final = 1.0
                delta = (advantages_glob - advantages_loc).detach()
                lambda_star_final = torch.clamp(delta / rho_final, 0.0, 1.0).detach()
            lambda_max_initial_final = float(self.config.get("lambda_max_initial", 1.0))
            if lambda_max_initial_final < 1.0:
                lambda_star_final = torch.clamp(lambda_star_final, 0.0, lambda_max_initial_final)
            
            # Local head: cooperative logits are a constant, so this backward
            # only moves the local head. λ stays at the behavior value.
            local_logits_final = self.agent.local_head(backbone_features_final)
            global_logits_final_const = self.agent.global_head(backbone_features_final, corridor_task_vec_batch).detach()
            mixed_logits_loc_final = (1.0 - lambda_star_final).unsqueeze(-1) * local_logits_final + lambda_star_final.unsqueeze(-1) * global_logits_final_const
            mixed_dist_loc_final = torch.distributions.Categorical(logits=mixed_logits_loc_final)
            log_probs_mixed_loc_final = mixed_dist_loc_final.log_prob(actions)
            ratio_loc_final = torch.exp(torch.clamp(log_probs_mixed_loc_final - old_log_probs, -10.0, 10.0))
            surr1_loc_final = ratio_loc_final * advantages_loc
            surr2_loc_final = torch.clamp(ratio_loc_final, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio) * advantages_loc
            policy_loss_loc_final = -torch.min(surr1_loc_final, surr2_loc_final).mean()
            
            self.local_head_optimizer.zero_grad()
            if debug_checks and (not torch.isfinite(policy_loss_loc_final)):
                print("⚠️ Warning: policy_loss_loc_final is NaN/Inf, skipping local head update")
                self.local_head_optimizer.zero_grad()
                continue
            policy_loss_loc_final.backward()
            local_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.local_head.parameters(), max_grad_norm)
            if debug_checks and (not torch.isfinite(local_grad_norm)):
                print("⚠️ Warning: local head grad norm is NaN/Inf, skipping update")
                self.local_head_optimizer.zero_grad()
                continue
            self.local_head_optimizer.step()
            
            # Cooperative head: local logits are the constant this time.
            local_logits_final_const = local_logits_final.detach()
            global_logits_final = self.agent.global_head(backbone_features_final, corridor_task_vec_batch)
            mixed_logits_glob_final = (1.0 - lambda_star_final).unsqueeze(-1) * local_logits_final_const + lambda_star_final.unsqueeze(-1) * global_logits_final
            mixed_dist_glob_final = torch.distributions.Categorical(logits=mixed_logits_glob_final)
            log_probs_mixed_glob_final = mixed_dist_glob_final.log_prob(actions)
            ratio_glob_final = torch.exp(torch.clamp(log_probs_mixed_glob_final - old_log_probs, -10.0, 10.0))
            surr1_glob_final = ratio_glob_final * advantages_glob
            surr2_glob_final = torch.clamp(ratio_glob_final, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio) * advantages_glob
            policy_loss_glob_final = -torch.min(surr1_glob_final, surr2_glob_final).mean()
            
            self.global_head_optimizer.zero_grad()
            if debug_checks and (not torch.isfinite(policy_loss_glob_final)):
                print("⚠️ Warning: policy_loss_glob_final is NaN/Inf, skipping global head update")
                self.global_head_optimizer.zero_grad()
                continue
            policy_loss_glob_final.backward()
            global_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.global_head.parameters(), max_grad_norm)
            if debug_checks and (not torch.isfinite(global_grad_norm)):
                print("⚠️ Warning: global head grad norm is NaN/Inf, skipping update")
                self.global_head_optimizer.zero_grad()
                continue
            self.global_head_optimizer.step()
            
            # Critics are fit to the returns, not to the policy gradient.
            # The cooperative critic is masked to in-corridor samples.
            values_loc_final = self.agent.value_loc_head(backbone_features_final)
            values_glob_final = self.agent.value_glob_head(backbone_features_final, critic_features_batch)
            if values_loc_final.dim() > 1:
                values_loc_final = values_loc_final.squeeze(-1)
            if values_glob_final.dim() > 1:
                values_glob_final = values_glob_final.squeeze(-1)
            if values_loc_final.dim() == 0:
                values_loc_final = values_loc_final.unsqueeze(0)
            if values_glob_final.dim() == 0:
                values_glob_final = values_glob_final.unsqueeze(0)
            
            value_loss_loc_final = F.mse_loss(values_loc_final, returns_loc)
            if torch.any(in_corridor_mask > 0.5):
                m = in_corridor_mask > 0.5
                value_loss_glob_final = F.mse_loss(values_glob_final[m], returns_glob[m])
            else:
                value_loss_glob_final = torch.zeros((), device=self.device, requires_grad=True)
            
            self.value_loc_optimizer.zero_grad()
            if debug_checks and (not torch.isfinite(value_loss_loc_final)):
                print("⚠️ Warning: value_loss_loc_final is NaN/Inf, skipping value_loc update")
                self.value_loc_optimizer.zero_grad()
                continue
            value_loss_loc_final.backward()
            value_loc_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.value_loc_head.parameters(), max_grad_norm)
            if debug_checks and (not torch.isfinite(value_loc_grad_norm)):
                print("⚠️ Warning: value_loc head grad norm is NaN/Inf, skipping update")
                self.value_loc_optimizer.zero_grad()
                continue
            self.value_loc_optimizer.step()
            
            self.value_glob_optimizer.zero_grad()
            if debug_checks and (not torch.isfinite(value_loss_glob_final)):
                print("⚠️ Warning: value_loss_glob_final is NaN/Inf, skipping value_glob update")
                self.value_glob_optimizer.zero_grad()
                continue
            value_loss_glob_final.backward()
            value_glob_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.value_glob_head.parameters(), max_grad_norm)
            if debug_checks and (not torch.isfinite(value_glob_grad_norm)):
                print("⚠️ Warning: value_glob head grad norm is NaN/Inf, skipping update")
                self.value_glob_optimizer.zero_grad()
                continue
            self.value_glob_optimizer.step()

            # Check for corrupted parameters.
            if debug_checks:
                def _params_finite(module: torch.nn.Module) -> bool:
                    for p in module.parameters():
                        if p is None:
                            continue
                        if not torch.isfinite(p).all():
                            return False
                    return True
                if not _params_finite(self.agent.backbone):
                    print("❌ ERROR: backbone parameters contain NaN/Inf after update")
                    break
                if not _params_finite(self.agent.local_head):
                    print("❌ ERROR: local_head parameters contain NaN/Inf after update")
                    break
                if not _params_finite(self.agent.global_head):
                    print("❌ ERROR: global_head parameters contain NaN/Inf after update")
                    break
                if not _params_finite(self.agent.value_loc_head):
                    print("❌ ERROR: value_loc_head parameters contain NaN/Inf after update")
                    break
                if not _params_finite(self.agent.value_glob_head):
                    print("❌ ERROR: value_glob_head parameters contain NaN/Inf after update")
                    break
            
            total_policy_loss += float(policy_loss_loc_new.item() + policy_loss_glob_new.item())
            total_value_loss += float(value_loss_loc_new.item() + value_loss_glob_new.item())
        
