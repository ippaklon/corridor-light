# CorridorLight

Official code for **CorridorLight: Cooperation as Task Negotiation via Counterfactual-Gain Gating for Traffic Signal Control**, accepted at **NeurIPS 2026**.

CorridorLight treats cooperation as a negotiated task rather than a single shared reward. A slow **corridor agent** reads a movement graph and proposes a few junction-disjoint corridors. Each signal then runs its own **intersection agent**, which keeps a local policy and a corridor-conditioned cooperative policy and mixes them with a counterfactual-gain gate. The gate opens only when cooperating has a positive marginal gain over acting locally, so a bad corridor proposal does not override a signal that is already doing fine.

<p align="center">
  <img src="figs/framework.png" alt="CorridorLight framework: movement graph, corridor agent, and gated intersection agent" width="920">
</p>

## Method

Traffic is lifted from raw lanes to a **movement graph**. Lanes that leave the same junction, enter the same junction, and share a lane index become one node. Edges follow SUMO connections, and a GNN (GCN, GAT, or GraphSAGE) encodes each movement.

The corridor agent replans at a low frequency (`corridor_delta_time`, default 20 s):

1. **Seed.** Score every movement embedding with a state value and keep the top candidates.
2. **Grow.** Walk outward with a sequential policy. Each step chooses a neighbor or `STOP`, and writes a lane preference.
3. **Disjoint tasks.** A controlled junction may belong to only one corridor. The resulting task is a path, a lane mask, and a normalized preference vector.

<p align="center">
  <img src="figs/agents.png" alt="Corridor agent and intersection agent" width="920">
</p>

<p align="center">
  <img src="figs/seed_grow.png" alt="Seeding, growing, and the junction-disjoint constraint" width="920">
</p>

The intersection agent does not see the GNN. It uses the standard SUMO-RL observation, a shared backbone, a local head, and a cooperative head conditioned on the task vector `[mask, preference]`. The executed logits are

```text
z = (1 - λ) * z_local + λ * z_coop
λ = clip(Δ / ρ, 0, 1)
```

At decision time, `Δ` is the normalized critic gap `(V_coop - V_local) / (|V_coop| + |V_local| + ε)` and `ρ` is `lambda_friction_coeff`. A larger `ρ` makes the gate harder to open. `lambda_max_initial` can cap `λ` while the cooperative head is still learning. The shared backbone is updated with MGDA so the local and cooperative losses do not fight over the same features. See `IntersectionAgent.compute_lambda` and `MGDATrainer`.

## Learned corridors

Colored paths are corridors proposed on three RESCO networks. Each row is four successive replans. Junctions already claimed by one corridor are not reused by the next.

**Arterial 4×4**

<p align="center">
  <img src="figs/wbg_art1.png" width="220">
  <img src="figs/wbg_art2.png" width="220">
  <img src="figs/wbg_art3.png" width="220">
  <img src="figs/wbg_art4.png" width="220">
</p>

**Cologne**

<p align="center">
  <img src="figs/wbg_col1.png" width="220">
  <img src="figs/wbg_col2.png" width="220">
  <img src="figs/wbg_col3.png" width="220">
  <img src="figs/wbg_col4.png" width="220">
</p>

**Ingolstadt**

<p align="center">
  <img src="figs/wbg_ig1.png" width="220">
  <img src="figs/wbg_ig2.png" width="220">
  <img src="figs/wbg_ig3.png" width="220">
  <img src="figs/wbg_ig4.png" width="220">
</p>

## Repository layout

```text
main.py                          training entry point
CorridorLight/
  config.py                      map / agent config merge
  configs/agents/corridor.yaml   algorithm hyperparameters
  configs/maps/*.yaml            per-network SUMO files and reward
  models/                        movement graph, GNN, seed, grow
  agents/                        corridor agent and intersection agent
  trainers/                      episode loop and MGDA-PPO
sumo_rl/                         vendored SUMO-RL plus RESCO networks
figs/                            figures used above
```

| Piece | File | Role |
| --- | --- | --- |
| Movement graph | `models/pressure_graph.py` | Lane aggregation, edges, node potential |
| Encoder | `models/gnn_encoder.py` | GCN / GAT / GraphSAGE embeddings |
| Seed | `models/seed_module.py` | `V(v)` used to rank seeds |
| Grow | `models/grow_module.py` | Neighbor / `STOP` policy and preferences |
| Corridor agent | `agents/corridor_agent.py` | Top-k seeding under the disjoint constraint |
| Intersection agent | `agents/intersection_agent.py` | Counterfactual-gain gate and mixed phase policy |
| Trainer | `trainers/corridor_trainer.py` | SUMO loop, replanning, logging |
| MGDA | `trainers/mgda_trainer.py` | Two-objective PPO on the shared backbone |

## Requirements

- Linux or Windows
- Python 3.10+ (tested with 3.12)
- [SUMO](https://eclipse.dev/sumo/) with `SUMO_HOME` pointing at the installation
- PyTorch, PyTorch Geometric, NumPy, PyYAML, pandas, Gymnasium, PettingZoo

`sumo_rl` in this repository is a vendored copy of [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) and is imported from the repo root. Install PyTorch first, then a matching [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) build. The remaining packages are ordinary:

```bash
pip install numpy pyyaml pandas gymnasium pettingzoo
```

SUMO must be on the library path. On Linux:

```bash
export SUMO_HOME=/path/to/sumo
export LIBSUMO_AS_TRACI=1   # set automatically by main.py; libsumo is much faster than TraCI
```

On Windows PowerShell:

```powershell
$env:SUMO_HOME = "C:\path\to\sumo"
```

## Training

Named configs live in `CorridorLight/configs`. These four maps ship with both a YAML file and a RESCO network:

```bash
python main.py --map cologne8
python main.py --map arterial4x4 --device cuda:0
python main.py --map ingolstadt7 --wandb
python main.py --map ingolstadt21 --runs 3
```

Pass YAML paths when you want a custom pair:

```bash
python main.py \
  --map CorridorLight/configs/maps/cologne8.yaml \
  --agent CorridorLight/configs/agents/corridor.yaml
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--seed` | NumPy / PyTorch / Python seed (default 42) |
| `--device` | `cpu`, `cuda`, or `cuda:0`. Config default is `auto` |
| `--wandb` | Log to Weights & Biases. Off unless this flag is set |
| `--runs N` | Repeat training `N` times. Each run closes SUMO before the next one |
| `--max-episodes` | Override `max_episodes` |
| `--checkpoint-interval` | Save every N episodes. `0` disables checkpoints |
| `--reward` | Override the SUMO-RL reward name |
| `--scale` | Multiply demand by this factor (`traffic_scale`) |
| `--lambda-friction-coeff` | Override the gate friction `ρ` |
| `--share` | Share intersection-agent weights. Applied only when the network path contains `4x4` and every signal has the same observation and action size |
| `--exp` | Sweep `reward` in `{queue, waiting-time}` and `scale` in `{1.0, 1.5, 2.0}` |
| `--verbose` | Print the resolved config and full tracebacks |

A short smoke run:

```bash
python main.py --map cologne8 --max-episodes 2 --device cpu --checkpoint-interval 0
```

Each run writes to `results/<map>/<agent>/<timestamp>/`, including SUMO `tripinfo.xml`, episode logs, and `checkpoints/` when checkpointing is on. The run name also records the reward, learning rate, corridor decay, and GNN architecture.

## Configuration

Map files own the simulator. Agent files own the algorithm. `lambda_friction_coeff` is map-owned and overrides the agent file, because the right gate friction depends on the network. The map `reward_fn` also overrides the agent file, so the corridor potential and the signal reward stay on the same metric.

Shipped maps:

| Map | Signals | Reward | `ρ` (`lambda_friction_coeff`) | Horizon |
| --- | --- | --- | --- | --- |
| `arterial4x4` | synthetic arterial grid | `diff-waiting-time` | 0.5 | 3600 s |
| `cologne8` | Cologne subset | `queue` | 1.0 | 3600 s |
| `ingolstadt7` | Ingolstadt subset | `queue` | 1.0 | 3600 s |
| `ingolstadt21` | Ingolstadt subset | `diff-waiting-time` | 1.0 | 3600 s |

`--map <name>` loads `configs/maps/<name>.yaml`. A name without that file does not fall through to another network. Cologne1, Cologne3, and Ingolstadt1 are already under `sumo_rl/nets/RESCO/`. To train on one of them, copy `configs/maps/cologne8.yaml` and point `net_file` and `route_file` at that network.

Knobs that change the method, with the defaults in `configs/agents/corridor.yaml`:

| Key | Default | What it does |
| --- | --- | --- |
| `max_corridors` | 6 | How many disjoint corridors may be active at once |
| `max_corridor_length` | 5 | Grow steps after the seed, including the `STOP` decision |
| `seed_top_k` | 16 | How many high-value seeds are tried before planning stops |
| `corridor_delta_time` | 20 | Seconds between corridor replans |
| `corridor_reward_decay` | 0.9 | Discount along a corridor when scoring its seed value |
| `allow_single_lane_corridor` | false | Reject a corridor that covers fewer than two lanes |
| `gnn_architecture` | `GAT` | `GCN`, `GAT`, or `GraphSAGE` |
| `lambda_friction_coeff` | per map | Gate friction `ρ`. Larger values keep `λ` closer to 0 |
| `lambda_max_initial` | 0.5 | Hard cap on `λ` |
| `reward_glob_mode` | `corridor_only` | Cooperative target. See below |
| `disable_corridor_agent` | false | Local-only ablation. No gate, no upper level |
| `share_intersection_parameters` | false | One shared policy across signals. CLI: `--share` |

`reward_glob_mode`:

- `corridor_only` — the cooperative return is the corridor-lane bonus.
- `shaping` — cooperative return = local reward + coefficient × corridor bonus.
- `same_as_loc` — cooperative return equals the local reward, and the upper level is not updated.

`corridor_decayed` is accepted and rewritten to `corridor_only`.

## What one episode does

`CorridorRLTrainer.train` is the loop behind `main.py`.

1. Build the movement graph once from the SUMO net.
2. Reset SUMO-RL. Observations stay the stock per-signal vectors.
3. Every `corridor_delta_time` seconds, encode the graph, grow a disjoint corridor set, and push lane masks and preferences to the signals those corridors cover.
4. Each signal computes `λ` and samples a phase from the mixed logits. Signals with no active corridor stay on the local head (`λ = 0`).
5. After the episode, MGDA-PPO updates every intersection agent. The corridor agent is then updated from the returns of the lanes it claimed. An immediate `STOP` is penalized by `corridor_immediate_stop_penalty`.

Checkpoints store the GNN, seed module, grow module, intersection agents, and a copy of the two YAML files that produced the run.

## Citation

```bibtex
@inproceedings{corridorlight2026,
  title     = {CorridorLight: Cooperation as Task Negotiation via Counterfactual-Gain Gating for Traffic Signal Control},
  author    = {Ip, Pak Lon and Ren, Pengfei and Lin, Yuteng and Chen, Rongqin and Chan, Tsz Nam and U, Leong Hou},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

The camera-ready manuscript is not included in this repository.

## Credits and licenses

The simulator stack is [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) by Lucas N. Alegre, with [SUMO](https://eclipse.dev/sumo/). Installation details for `SUMO_HOME` and `LIBSUMO_AS_TRACI` are documented there.

RESCO networks under `sumo_rl/nets/RESCO/` keep their upstream licenses (Cologne: Creative Commons; Ingolstadt and Arterial: see the `LICENSE` files next to each net). Those networks are redistributed so this repository can be run without a separate map download.
