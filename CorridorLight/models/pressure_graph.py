"""
Build the movement graph that the corridor agent reads.

A SUMO network is a lane graph, which is too fine for corridor planning:
parallel lanes on the same approach carry the same movement. This module
collapses them. A movement node is the triple

    (upstream junction, downstream junction, lane index).

A directed edge exists when any lane of the source movement connects to any
lane of the target movement. The resulting graph is what ``GNNEncoder``
runs message passing on.

Node features are a scalar traffic potential, not a raw sensor dump. The
potential is aligned with ``reward_fn`` so the corridor agent and the
signal reward argue about the same quantity:

* ``diff-waiting-time`` — reduction in accumulated waiting time, scaled
  by 1/100 to match SUMO-RL.
* ``queue`` — negative queue length, scaled by 1/50.
* ``pressure`` — negative lane density, used as a lane-level stand-in.
* ``average-speed`` — mean speed divided by the lane speed limit.
* anything else — negative scaled waiting time.

``save_temporal_graph`` averages that potential over the lanes of each
movement and returns a PyG ``Data`` object. The edge index is topology
and does not change during an episode.
"""

import os
import sys
from typing import Any, Dict, List
from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import Data

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

import sumolib


class PressureGraphBuilder:
    """Lane-to-movement graph plus the per-lane potential used as its feature.

    Call ``build_direction_graph`` once per network. Call ``compute_pressure``
    and ``save_temporal_graph`` whenever the corridor agent replans.
    """
    
    def __init__(self, reward_fn: str = "diff-waiting-time", net_file: str = None):
        """
        Args:
            reward_fn: Reward used for lane pressure.
            net_file: SUMO network path.
        """
        self.reward_fn = reward_fn
        self.net_file = net_file
        self.direction_graph = None
        # lane id -> movement-node index
        self.lane_to_direction = {}
        # movement-node index -> lane ids that were aggregated into it
        self.direction_to_lanes = {}
        # movement-node index -> downstream junction. This is the junction
        # the disjoint constraint protects, because that is where the
        # movement is served by a signal.
        self.direction_to_junction = {}
        self.edge_index = None  # PyG edge_index, shape [2, num_edges]
        self.num_directions = 0
        self._sumo_net = None  # kept so captions can recover edge names

        # (signal id, lane id) -> previous waiting time. diff-waiting-time
        # is a delta, so the last observation has to be remembered. The
        # scale matches SUMO-RL, which divides accumulated waiting time by 100.
        self._last_lane_waiting_time: Dict[tuple, float] = {}
        # signal id -> {lane id -> column in that signal's per-lane arrays}
        self._lane_index_cache: Dict[str, Dict[str, int]] = {}
        
        if net_file:
            self.build_direction_graph(net_file)
        
    def build_direction_graph(self, net_file: str):
        """Read ``net_file`` and cache the movement topology.

        Lane connections come from ``lane.getOutgoing()``. Two movements are
        linked when at least one of those connections crosses between them.
        Self-loops are dropped: a corridor should leave the node it is on.
        """
        net = sumolib.net.readNet(net_file)
        self._sumo_net = net
        
        # Collect lanes and outgoing links.
        lanes = {}
        lane_connections = defaultdict(list)  # lane_id -> [outgoing_lane_ids]
        
        for edge in net.getEdges():
            for lane in edge.getLanes():
                lane_id = lane.getID()
                lanes[lane_id] = lane
                
                for connection in lane.getOutgoing():
                    to_lane = connection.getToLane()
                    if to_lane:
                        lane_connections[lane_id].append(to_lane.getID())
        
        # Parallel lanes of one approach share a movement. The lane index
        # keeps a left-turn lane distinct from the through lane beside it.
        direction_groups = defaultdict(list)  # (from_junction, to_junction, lane index) -> [lane ids]
        
        for lane_id, lane in lanes.items():
            edge = lane.getEdge()
            from_junction = edge.getFromNode().getID() if edge.getFromNode() else None
            to_junction = edge.getToNode().getID() if edge.getToNode() else None
            
            direction_key = (from_junction, to_junction, lane.getIndex())
            direction_groups[direction_key].append(lane_id)
        
        # Create direction nodes.
        self.lane_to_direction = {}
        self.direction_to_lanes = {}
        self.direction_to_junction = {}
        
        direction_idx = 0
        for direction_key, lane_ids in direction_groups.items():
            from_junction, to_junction, _ = direction_key
            
            self.direction_to_lanes[direction_idx] = lane_ids
            self.direction_to_junction[direction_idx] = to_junction
            
            for lane_id in lane_ids:
                self.lane_to_direction[lane_id] = direction_idx
            
            direction_idx += 1
        
        self.num_directions = direction_idx
        
        # Build direction edges.
        direction_edges = []
        for direction_idx in range(self.num_directions):
            lane_ids = self.direction_to_lanes[direction_idx]
            
            outgoing_directions = set()
            for lane_id in lane_ids:
                for out_lane_id in lane_connections.get(lane_id, []):
                    if out_lane_id in self.lane_to_direction:
                        out_direction = self.lane_to_direction[out_lane_id]
                        if out_direction != direction_idx:
                            outgoing_directions.add(out_direction)
            
            for out_direction in outgoing_directions:
                direction_edges.append([direction_idx, out_direction])
        
        # Convert to PyG edge_index.
        if direction_edges:
            self.edge_index = torch.tensor(direction_edges, dtype=torch.long).t().contiguous()
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    def compute_pressure(self, lane_id: str, traffic_signal) -> float:
        """Return this lane's scalar potential under the active reward.

        The value is signed the same way as a SUMO-RL reward: less queue
        and less waiting time are larger (less negative). Callers that want
        a congestion score for seed selection flip the sign themselves.
        Lanes that do not belong to ``traffic_signal`` contribute 0.
        """
        # Align lane ids with SUMO-RL per-lane arrays.
        ts_id = getattr(traffic_signal, "id", None)
        if ts_id is None:
            return 0.0
        ts_id = str(ts_id)

        lane_to_idx = self._lane_index_cache.get(ts_id)
        if lane_to_idx is None:
            lanes = list(getattr(traffic_signal, "lanes", []) or [])
            lane_to_idx = {lid: i for i, lid in enumerate(lanes)}
            self._lane_index_cache[ts_id] = lane_to_idx

        idx = lane_to_idx.get(lane_id, None)
        if idx is None:
            return 0.0

        # Lane-level pressure/reward.
        if self.reward_fn == "diff-waiting-time":
            # Match SUMO-RL diff waiting time scaling.
            try:
                per_lane_wait = traffic_signal.get_accumulated_waiting_time_per_lane()
                lane_wait = float(per_lane_wait[idx]) / 100.0
            except Exception:
                lane_wait = 0.0

            key = (ts_id, lane_id)
            last = float(self._last_lane_waiting_time.get(key, lane_wait))
            reward = last - lane_wait
            self._last_lane_waiting_time[key] = lane_wait
            return float(reward)

        if self.reward_fn == "queue":
            # Higher queue means lower reward.
            try:
                q = traffic_signal.get_lanes_queue()
                return -float(q[idx]) / 50.0
            except Exception:
                return 0.0

        if self.reward_fn == "pressure":
            # Use negative lane density as a lane-level proxy.
            try:
                d = traffic_signal.get_lanes_density()
                return -float(d[idx])
            except Exception:
                return 0.0

        if self.reward_fn == "average-speed":
            # Normalized lane speed.
            try:
                mean_speed = float(traffic_signal.sumo.lane.getLastStepMeanSpeed(lane_id))
                allowed_speed = float(traffic_signal.sumo.lane.getAllowedSpeed(lane_id))
                if allowed_speed <= 0.0:
                    return 0.0
                if mean_speed < 0.0:
                    mean_speed = allowed_speed
                return float(mean_speed / allowed_speed)
            except Exception:
                return 0.0

        # Default: negative scaled waiting time.
        try:
            per_lane_wait = traffic_signal.get_accumulated_waiting_time_per_lane()
            lane_wait = float(per_lane_wait[idx]) / 100.0
            return -lane_wait
        except Exception:
            return 0.0
    
    def save_temporal_graph(self, timestep: int, pressure_dict: Dict[str, float]) -> Data:
        """Average lane potentials onto movement nodes.

        ``timestep`` is accepted so callers can stamp a snapshot. The feature
        itself is only the current potential. Empty movements get 0, which
        keeps the node count identical to the topology built at startup.
        """
        node_features = []
        
        for direction_idx in range(self.num_directions):
            lane_ids = self.direction_to_lanes[direction_idx]
            
            direction_pressures = []
            for lane_id in lane_ids:
                if lane_id in pressure_dict:
                    direction_pressures.append(pressure_dict[lane_id])
            
            if direction_pressures:
                avg_pressure = np.mean(direction_pressures)
            else:
                avg_pressure = 0.0
            
            node_features.append([avg_pressure])
        
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        
        data = Data(
            x=node_features_tensor,
            edge_index=self.edge_index,
            num_nodes=self.num_directions
        )
        
        return data

    def describe_direction_node(self, direction_node_id: int) -> Dict[str, Any]:
        """Resolve a movement node back to a SUMO edge and its two junctions.

        ``direction_to_junction`` stores the downstream junction (the edge
        ``toNode``). The upstream junction is read from the cached net so
        log lines can be checked against ``sumo-gui``.
        """
        nid = int(direction_node_id)
        lane_ids = list(self.direction_to_lanes.get(nid) or [])
        junction_to_id = self.direction_to_junction.get(nid)

        junction_from_id = None
        edge_id = None
        if lane_ids and self._sumo_net is not None:
            try:
                lane0 = self._sumo_net.getLane(lane_ids[0])
                edge = lane0.getEdge()
                edge_id = edge.getID()
                junction_from_id = edge.getFromNode().getID() if edge.getFromNode() else None
                if junction_to_id is None and edge.getToNode():
                    junction_to_id = edge.getToNode().getID()
            except Exception:
                pass

        return {
            "direction_node_id": nid,
            "edge_id": edge_id or "",
            "junction_from_id": junction_from_id or "",
            "junction_to_id": str(junction_to_id) if junction_to_id else "",
            "lanes": lane_ids,
        }

    def corridor_nodes_caption_line(self, direction_nodes: List[int]) -> str:
        """One log line for a corridor, ``node:edge[from→to]`` per movement."""
        parts = []
        for dn in direction_nodes:
            info = self.describe_direction_node(int(dn))
            e = info.get("edge_id") or "?"
            jf = info.get("junction_from_id") or "?"
            jt = info.get("junction_to_id") or "?"
            parts.append(f"{dn}:{e}[{jf}→{jt}]")
        return " | ".join(parts)

