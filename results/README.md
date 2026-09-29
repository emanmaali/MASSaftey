# Shipped results

Every file here is either an **input** the replay scripts consume (`suffixes/`) or a **per-task outcome** that a paper table is computed from. Nothing else is included. Each outcome JSON records one (model, topology, attack, task) run. Its boolean `hd_*` / `full_*` fields say which nodes were compromised and whether the final agent called the attacker's tool. See `analysis/NOTES_*.md` for field meanings per table.

## Inputs: frozen attack suffixes (`suffixes/`)

| Folder | Contents | Used by |
|---|---|---|
| `beast_toy/<model>/` | BEAST suffixes, 50 toy tasks (qwen-0.5b, qwen-1.5b; phi-3.5-mini tasks 0–9) | `reproduce_topology.sh`, `reproduce_families.sh` (beast), `reproduce_bfcl.sh` (toy column) |
| `beast_bfcl/qwen_0.5b/` | BEAST suffixes, 100 real-BFCL tasks, original BFCL target | `reproduce_bfcl.sh`, `reproduce_frontier.sh` (toolsel) |
| `beast_bfcl_retargeted/qwen_0.5b/` | BEAST suffixes, 100 real-BFCL tasks, fixed out-of-domain target | `reproduce_bfcl.sh` |
| `cfh/`, `tamas_dpi/`, `attack_families/` | ASR0 (original payload) and ASR1 (re-optimised) suffixes for the eight reproduced attacks | `reproduce_families.sh`, `reproduce_frontier.sh` (crossfam) |
| `null_bfcl/`, `null_toy/` | Empty suffixes for the null-suffix controls | `reproduce_null_controls.sh` |

## Outcomes, by paper table

| Folder | Paper tables |
|---|---|
| `run_topology_depth_variants/`, `run_topology_fanmerge/`, `run_graph_topology/` | `tab:topo-asr`, `tab:interim-full`, `tab:rq2-generalise`, `tab:protocol-gen`, and the eight-attack tables (`tab:rq1-*`, `tab:rq2-crossfam`, `tab:toma-bounds`) |
| `run_graph_topology_bfcl/`, `run_topology_depth_variants_bfcl/`, `run_topology_fanmerge_bfcl/`, `run_beast_injected_pilot/` | `tab:realbfcl-attempt` and the BFCL prose |
| `null_suffix_check*/` | Null-suffix controls quoted in the BFCL discussion |
| `run_topology_frontier/` | `tab:frontier-toolsel`, `tab:frontier-estimator` |
| `run_topology_frontier_generic/`, `run_asr0_doublecheck/` | `tab:frontier-crossfam` |
| `run_adaptive_frontier/` | `tab:adaptive` |
| `tradingagents/` | `tab:tradingagents-leak`, `tab:tradingagents-all` |

Folder names are the ones the original drivers wrote. `run_beast_injected_pilot/` holds the real-BFCL replays with the fixed (retargeted) attack target.
