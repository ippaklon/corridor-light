"""
Upper-level planner: turn movement embeddings into disjoint corridor tasks.

``generate_corridors`` is one replan.

1. Rank every movement by the seed module's state value.
2. Walk that ranking. Skip a seed whose movement is already inside an
   accepted corridor, or whose downstream junction is already claimed.
   Stop once ``max_corridors`` corridors have been accepted or
   ``seed_top_k`` seeds have been considered.
3. Grow each surviving seed with ``GrowModule``. At every step the legal
   actions are the outgoing neighbors that do not repeat a node and do
   not enter a junction owned by this corridor or by an earlier one.
   ``STOP`` ends the walk. ``STOP`` on the first step is recorded as
   ``stopped_immediately`` and penalized by the trainer.
4. Drop a corridor that has no lane the lower level can actually observe.
   Optionally also drop corridors that cover fewer than two lanes.
5. L1-normalize lane preferences. A corridor whose learned preferences
   are all zero falls back to the uniform distribution, so the
   intersection agent still receives a valid task vector.

The disjoint constraint applies only to movements that touch an observable
lane. Connector nodes with no signalled lane may be shared. That keeps a
path topologically valid on networks where two signalled approaches are
joined by an unsignalled link.

Returned objects are ``Corridor`` records. When the trainer is updating
the upper level it also asks for the log-probabilities, Q values,
preferences, and entropies of the actions that were sampled. Those tensors
stay in the autograd graph. The corridor itself stores detached floats.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
import torch
from torch_geometric.data import Data

from ..models.gnn_encoder import GNNEncoder
from ..models.seed_module import SeedModule
from ..models.grow_module import GrowModule
from ..models.pressure_graph import PressureGraphBuilder


@dataclass
class Corridor:
    """One accepted corridor task.

    ``direction_nodes`` is the movement sequence, seed first.
    ``seed_value`` is the discounted sum of movement potentials along that
    sequence (see ``_compute_seed_value``). It is a diagnostic. Seed
    *selection* uses the learned state value, not this sum.
    ``preference_values`` maps lane id to a weight that sums to 1 over the
    corridor. ``junctions`` holds the controlled junctions this corridor
    owns, and is what the next seed is checked against.
    """
    direction_nodes: List[int]
    seed_value: float
    preference_values: Dict[str, float]  # lane id -> normalized preference
    junctions: Set[str]
    # True when the grow policy sampled STOP before adding any neighbor.
    stopped_immediately: bool = False


class CorridorAgent:
    """Sample a junction-disjoint set of corridors from one graph embedding.

    The three networks are owned by the trainer and passed in. This class
    holds the constraint state: which junction each movement serves, which
    lanes it covers, and which of those lanes a signal can observe.
    """
    
    def __init__(
        self,
        gnn_encoder: GNNEncoder,
        seed_module: SeedModule,
        grow_module: GrowModule,
        config: Dict[str, Any]
    ):
        """
        Args:
            gnn_encoder: GNN encoder
            seed_module: Seed module
            grow_module: Grow module
            config: Agent config.
        """
        self.gnn_encoder = gnn_encoder
        self.seed_module = seed_module
        self.grow_module = grow_module
        self.config = config
        
        self.max_corridors = config.get("max_corridors", 3)
        self.max_corridor_length = config.get("max_corridor_length", 10)
        self.seed_top_k = config.get("seed_top_k", 5)
        self.corridor_reward_decay = config.get("corridor_reward_decay", 0.9)
        self.allow_single_lane_corridor = bool(config.get("allow_single_lane_corridor", True))
        
        # Filled by set_direction_graph. node_to_junction is the downstream
        # junction, which is the signal that serves the movement.
        self.direction_graph = None
        self.node_to_junction = {}
        self.direction_to_lanes = {}
        self.observable_lanes: Optional[Set[str]] = None
        self.last_generation_diagnostics: Dict[str, Any] = {}
    
    def set_direction_graph(self, pressure_graph: Data, graph_builder: PressureGraphBuilder):
        """Cache the current PyG graph and the lane / junction lookup tables.

        The graph's ``edge_index`` is what ``_get_neighbors`` walks. Replacing
        it every replan is safe: the topology does not change, but the node
        features do, and ``_compute_seed_value`` reads those features.
        """
        self.direction_graph = pressure_graph
        self.node_to_junction = graph_builder.direction_to_junction.copy()
        self.direction_to_lanes = graph_builder.direction_to_lanes.copy()

    def set_observable_lanes(self, lane_ids: Optional[Set[str]]):
        """Restrict the disjoint constraint to lanes a signal can observe.

        ``None`` means every lane counts. A movement with no observable lane
        is treated as a connector: it may be shared, and it does not own a
        junction.
        """
        if lane_ids is None:
            self.observable_lanes = None
            return
        self.observable_lanes = {str(lane_id) for lane_id in lane_ids}

    def generate_corridors(
        self,
        pressure_graph: Data,
        node_embeddings: torch.Tensor,
        return_log_probs: bool = False,
        return_q_values: bool = False,
        return_v_values: bool = False,
        return_pref_tensors: bool = False,
        return_entropies: bool = False
    ):
        """Plan the corridors for one replan.

        ``node_embeddings`` is ``[num_movements, embedding_dim]`` from the
        GNN. The return value is the corridor list, or a tuple whose first
        item is that list and whose remaining items follow the ``return_*``
        flags in order: log-probabilities, Q values, seed values, preference
        tensors, entropies. Flags that are false are omitted, so callers
        must unpack positionally in the same order they set the flags.

        Rejected seeds are counted in ``last_generation_diagnostics`` with
        a short reason (``grow_returned_none``, ``empty_direction_nodes``,
        ``insufficient_total_lanes``, ``no_observable_lanes``). The first
        twenty rejections also keep their node ids, which is enough to see
        why a network suddenly produces no corridors.
        """
        self.direction_graph = pressure_graph
        
        corridors = []
        used_junctions = set()
        log_probs_list = []
        q_values_list = []
        v_values_list = []
        pref_tensors_list = []
        entropies_list = []
        diagnostics = {
            "considered_seeds": 0,
            "valid_corridors": 0,
            "rejected_corridors": 0,
            "rejected_by_reason": {},
            "rejected_samples": [],
        }

        def _record_rejection(reason: str, seed_idx_value: int, corridor_obj=None, direction_nodes_value=None):
            diagnostics["rejected_corridors"] += 1
            by_reason = diagnostics["rejected_by_reason"]
            by_reason[reason] = int(by_reason.get(reason, 0)) + 1
            samples = diagnostics["rejected_samples"]
            if len(samples) >= 20:
                return
            try:
                dn = list(direction_nodes_value if direction_nodes_value is not None else getattr(corridor_obj, "direction_nodes", []) or [])
            except Exception:
                dn = []
            all_lanes = self._get_corridor_lane_ids(dn)
            observable_lanes = self._get_observable_corridor_lane_ids(dn)
            samples.append({
                "reason": str(reason),
                "seed": int(seed_idx_value),
                "direction_nodes": [int(node_id) for node_id in dn],
                "all_lane_count": int(len(all_lanes)),
                "observable_lane_count": int(len(observable_lanes)),
                "stopped_immediately": bool(getattr(corridor_obj, "stopped_immediately", False)) if corridor_obj is not None else False,
            })
        
        # Rank the whole graph. A top-k slice taken before the constraint
        # check would waste the budget on seeds whose junctions are taken.
        if return_v_values:
            seed_v_values = self.seed_module.forward(node_embeddings)  # [num_nodes]
            seed_scores = seed_v_values
        else:
            seed_scores = self.seed_module.forward(node_embeddings)
            seed_v_values = None
        ranked_seed_indices = torch.argsort(seed_scores, descending=True).tolist()
        
        # num_grown_corridors counts corridors that survived the lane checks,
        # which is what max_corridors limits. A seed that grows into an
        # empty or unobservable corridor is remembered in dropped_seeds so
        # a later pass in the same call does not retry it.
        num_grown_corridors = 0
        used_direction_nodes: Set[int] = set()
        dropped_seeds: Set[int] = set()
        considered_seeds = 0
        for seed_idx in ranked_seed_indices:
            if num_grown_corridors >= self.max_corridors:
                break
            if seed_idx in used_direction_nodes:
                continue
            if seed_idx in dropped_seeds:
                continue
            if considered_seeds >= self.seed_top_k:
                break
            considered_seeds += 1
            diagnostics["considered_seeds"] = int(considered_seeds)
            
            result = self._grow_corridor(
                seed_idx,
                node_embeddings,
                used_junctions,
                return_log_probs=return_log_probs,
                return_q_values=return_q_values,
                return_pref_tensors=return_pref_tensors,
                return_entropies=return_entropies
            )
            
            if result is None:
                _record_rejection("grow_returned_none", int(seed_idx))
                continue
            
            # Unpack according to requested return flags.
            corridor = None
            log_probs = []
            q_vals = []
            pref_tensors = []
            entropies = []

            result_parts = list(result) if isinstance(result, tuple) else [result]
            corridor = result_parts[0] if result_parts else None
            part_idx = 1
            if return_log_probs and part_idx < len(result_parts):
                log_probs = result_parts[part_idx]
                part_idx += 1
            if return_q_values and part_idx < len(result_parts):
                q_vals = result_parts[part_idx]
                part_idx += 1
            if return_pref_tensors and part_idx < len(result_parts):
                pref_tensors = result_parts[part_idx]
                part_idx += 1
            if return_entropies and part_idx < len(result_parts):
                entropies = result_parts[part_idx]

            if corridor is None:
                continue

            try:
                direction_nodes = list(getattr(corridor, "direction_nodes", []) or [])
            except Exception:
                direction_nodes = []
            if len(direction_nodes) < 1:
                dropped_seeds.add(int(seed_idx))
                _record_rejection("empty_direction_nodes", int(seed_idx), corridor, direction_nodes)
                continue
            corridor_lane_ids = self._get_corridor_lane_ids(direction_nodes)
            observable_lane_ids = self._get_observable_corridor_lane_ids(direction_nodes)
            if (not self.allow_single_lane_corridor) and len(corridor_lane_ids) < 2:
                dropped_seeds.add(int(seed_idx))
                _record_rejection("insufficient_total_lanes", int(seed_idx), corridor, direction_nodes)
                continue
            if len(observable_lane_ids) < 1:
                dropped_seeds.add(int(seed_idx))
                _record_rejection("no_observable_lanes", int(seed_idx), corridor, direction_nodes)
                continue

            num_grown_corridors += 1
            diagnostics["valid_corridors"] = int(num_grown_corridors)

            if isinstance(corridor, Corridor):
                try:
                    used_direction_nodes.update(list(corridor.direction_nodes))
                except Exception:
                    pass

            if isinstance(corridor, Corridor) and len(corridor.direction_nodes) >= 1:
                corridors.append(corridor)
                if return_log_probs:
                    log_probs_list.append(log_probs)
                if return_q_values:
                    q_values_list.append(q_vals)
                if return_v_values:
                    v_values_list.append(seed_v_values[seed_idx])
                if return_pref_tensors:
                    pref_tensors_list.append(pref_tensors)
                if return_entropies:
                    entropies_list.append(entropies)
                # corridor.junctions already excludes unobservable connectors,
                # so sharing those links does not break the constraint.
                used_junctions.update(corridor.junctions)
        
        self.last_generation_diagnostics = diagnostics
        
        result_list = [corridors]
        if return_log_probs:
            result_list.append(log_probs_list)
        if return_q_values:
            result_list.append(q_values_list)
        if return_v_values:
            result_list.append(v_values_list)
        if return_pref_tensors:
            result_list.append(pref_tensors_list)
        if return_entropies:
            result_list.append(entropies_list)
        
        if len(result_list) > 1:
            return tuple(result_list)
        return corridors
    
    def _grow_corridor(
        self,
        seed_idx: int,
        node_embeddings: torch.Tensor,
        used_junctions: Set[str],
        return_log_probs: bool = False,
        return_q_values: bool = False,
        return_pref_tensors: bool = False,
        return_entropies: bool = False
    ):
        """Extend ``seed_idx`` until STOP, a dead end, or ``max_corridor_length``.

        Returns ``None`` when the seed's own junction is already taken.
        Otherwise returns the ``Corridor`` and, in flag order, the sampled
        log-probabilities, selected Q values, selected preference tensors,
        and action entropies. The preference stored on the corridor is the
        detached float. The tensor is returned separately so the preference
        loss can still backpropagate.
        """
        corridor_prefix = [seed_idx]
        preference_values = {}
        log_probs = []
        q_values = []
        selected_pref_tensors = []
        action_entropies = []
        
        seed_junction = self.node_to_junction.get(seed_idx)
        if self._node_has_observable_lane(seed_idx) and seed_junction and seed_junction in used_junctions:
            return None
        
        used_junctions_in_corridor = set()
        if seed_junction and self._node_has_observable_lane(seed_idx):
            used_junctions_in_corridor.add(seed_junction)
        
        stopped_immediately = False
        # Grow step by step.
        for step in range(self.max_corridor_length):
            state = self.grow_module.encode_state(corridor_prefix, node_embeddings)
            
            tail_node = corridor_prefix[-1]
            neighbors = self._get_neighbors(tail_node)
            
            if len(neighbors) == 0:
                break
            
            candidate_indices = []
            candidate_embeddings = []
            action_mask = []
            
            for neighbor_idx in neighbors:
                if self._is_valid_action(corridor_prefix, neighbor_idx, used_junctions):
                    candidate_indices.append(neighbor_idx)
                    candidate_embeddings.append(node_embeddings[neighbor_idx])
                    action_mask.append(1)
            
            if len(candidate_indices) == 0:
                break

            # STOP stays legal even when every neighbor was filtered out.
            # The loop breaks earlier in that case, so this append only runs
            # when there is at least one neighbor to compare against STOP.
            action_mask.append(1)
            
            candidate_embeddings_tensor = torch.stack(candidate_embeddings)
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=node_embeddings.device)
            
            if return_q_values:
                action_logits, pref_values, q_vals = self.grow_module(
                    state,
                    candidate_embeddings_tensor,
                    action_mask_tensor,
                    return_q_values=True
                )
            else:
                action_logits, pref_values = self.grow_module(
                    state,
                    candidate_embeddings_tensor,
                    action_mask_tensor,
                    return_q_values=False
                )
            
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            if return_log_probs:
                log_probs.append(log_prob)
            if return_entropies:
                action_entropies.append(action_dist.entropy())
            
            if return_q_values:
                selected_q = q_vals[action]
                q_values.append(selected_q)
            
            action_item = action.item()
            
            # The STOP column is the one past the last neighbor.
            if action_item == len(candidate_indices):
                if step == 0:
                    stopped_immediately = True
                break
            
            selected_node = candidate_indices[action_item]
            corridor_prefix.append(selected_node)
            
            # Record preference values.
            if return_pref_tensors:
                # Keep only the selected preference tensor.
                selected_pref = pref_values[action_item]
                selected_pref_tensors.append(selected_pref)
                pref_val = float(selected_pref.detach().item())
            else:
                pref_val = float(pref_values[action_item].detach().item())

            lane_ids = self.direction_to_lanes.get(selected_node, [])
            if lane_ids:
                for lane_id in lane_ids:
                    # Keep the first value for deterministic traversal.
                    if lane_id not in preference_values:
                        preference_values[lane_id] = pref_val
            
            neighbor_junction = self.node_to_junction.get(selected_node)
            if neighbor_junction and self._node_has_observable_lane(selected_node):
                used_junctions_in_corridor.add(neighbor_junction)
        
        seed_value = self._compute_seed_value(corridor_prefix)

        # The seed itself never passed through the preference head, and a
        # movement can contain lanes that were not written above. Fill those
        # with zero. The normalization below turns an all-zero corridor into
        # a uniform task, and otherwise turns the learned scores into a
        # distribution the intersection agent can use as a lane weight.
        for node_id in corridor_prefix:
            for lane_id in self.direction_to_lanes.get(node_id, []) or []:
                if lane_id not in preference_values:
                    preference_values[lane_id] = 0.0

        # L1-normalize. Negative or non-finite scores are clipped to 0 first.
        try:
            if isinstance(preference_values, dict) and len(preference_values) > 0:
                keys = list(preference_values.keys())
                vals = []
                for k in keys:
                    try:
                        v = float(preference_values.get(k, 0.0))
                    except Exception:
                        v = 0.0
                    if not torch.isfinite(torch.tensor(v)).item():
                        v = 0.0
                    if v < 0.0:
                        v = 0.0
                    vals.append(v)
                s = float(sum(vals))
                if s <= 0.0:
                    uni = 1.0 / float(len(keys)) if len(keys) > 0 else 0.0
                    for k in keys:
                        preference_values[k] = float(uni)
                else:
                    for k, v in zip(keys, vals):
                        preference_values[k] = float(v) / s
        except Exception:
            pass
        
        corridor = Corridor(
            direction_nodes=corridor_prefix,
            seed_value=seed_value,
            preference_values=preference_values,
            junctions=used_junctions_in_corridor,
            stopped_immediately=stopped_immediately
        )
        
        result = [corridor]
        if return_log_probs:
            result.append(log_probs)
        if return_q_values:
            result.append(q_values)
        if return_pref_tensors:
            result.append(selected_pref_tensors)
        if return_entropies:
            result.append(action_entropies)
        return tuple(result) if len(result) > 1 else corridor
    
    def _get_neighbors(self, node_idx: int) -> List[int]:
        """Outgoing movement indices of ``node_idx`` in the cached edge index.

        The scan is linear in the number of edges. Movement graphs in this
        repository are a few hundred nodes, so the clarity is worth more
        than an adjacency list.
        """
        if self.direction_graph is None or self.direction_graph.edge_index is None:
            return []
        
        edge_index = self.direction_graph.edge_index
        neighbors = []
        
        for i in range(edge_index.shape[1]):
            if edge_index[0, i].item() == node_idx:
                neighbor = edge_index[1, i].item()
                neighbors.append(neighbor)
        
        return neighbors

    def _get_corridor_lane_ids(self, direction_nodes: List[int]) -> List[str]:
        """Lane ids covered by ``direction_nodes``, first occurrence kept."""
        lane_ids = []
        seen = set()
        for node_id in direction_nodes:
            for lane_id in self.direction_to_lanes.get(node_id, []) or []:
                lane_id = str(lane_id)
                if lane_id in seen:
                    continue
                seen.add(lane_id)
                lane_ids.append(lane_id)
        return lane_ids

    def _get_observable_corridor_lane_ids(self, direction_nodes: List[int]) -> List[str]:
        """Covered lanes that a signal agent can see and be rewarded for."""
        all_lane_ids = self._get_corridor_lane_ids(direction_nodes)
        if self.observable_lanes is None:
            return all_lane_ids
        return [lane_id for lane_id in all_lane_ids if lane_id in self.observable_lanes]

    def _is_valid_action(
        self,
        corridor_prefix: List[int],
        candidate_node: int,
        used_junctions: Set[str]
    ) -> bool:
        """Whether ``candidate_node`` may be appended to this corridor.

        The caller already limits candidates to outgoing neighbors. What
        remains is: do not repeat a movement (no cycles), and do not enter
        a controlled junction that this corridor or an earlier corridor
        already owns.
        """
        if candidate_node in corridor_prefix:
            return False

        candidate_junction = self.node_to_junction.get(candidate_node)
        if self._node_has_observable_lane(candidate_node) and candidate_junction and candidate_junction in used_junctions:
            return False
        
        return True

    def _node_has_observable_lane(self, node_id: int) -> bool:
        """True when this movement contains at least one observable lane.

        With no observation filter installed, every movement counts as
        observable. That is the right default for a fully signalled grid.
        """
        if self.observable_lanes is None:
            return True
        lanes = self.direction_to_lanes.get(node_id, []) or []
        return any(str(lane_id) in self.observable_lanes for lane_id in lanes)
    
    def _compute_seed_value(self, corridor_nodes: List[int]) -> float:
        """Discounted sum of movement potentials along the grown corridor.

        ``R = sum_i γ^i * x(v_i)``, with ``γ = corridor_reward_decay`` and
        ``x(v_i)`` the scalar stored on the graph. Index 0 is the seed, so
        the seed is not discounted. This number is logged on the corridor.
        It is not the ranking used to pick the seed.
        """
        if self.direction_graph is None or getattr(self.direction_graph, "x", None) is None:
            return 0.0
        
        if len(corridor_nodes) == 0:
            return 0.0
        
        pressures = self.direction_graph.x
        seed_value = 0.0
        gamma = float(self.corridor_reward_decay)
        
        for i, node_idx in enumerate(corridor_nodes):
            if node_idx >= pressures.shape[0]:
                continue
            r_pressure = float(pressures[node_idx][0].item())
            seed_value += (gamma ** i) * r_pressure
        
        return seed_value

