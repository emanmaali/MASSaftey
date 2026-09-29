# Known issues

This package reproduces the paper **as it was run**. Nothing in `masflow/` or `results/` has been corrected after the fact. This file lists every place where the paper, the code and the data are known to disagree, so that a reviewer re-running the pipeline is not surprised.

- **Section 1** covers the cells reported as `KNOWN` by `scripts/verify_tables.sh`.
- **Section 2** covers tables that match exactly, but only because the checker reproduces a particular choice made by the original analysis.
- **Section 3** covers implementation details that affect how the results should be read.
- **Section 4** lists what cannot be re-run.

The per-section notes in `analysis/NOTES_*.md` give the file-level evidence for each item.

---

## 1. Paper cells that do not match the shipped data

`scripts/verify_tables.sh` recomputes every numeric cell from `results/`. These 17 cells in 6 checks differ from the paper. The checker labels them `KNOWN` rather than failing on them.

| Check | Cells | Paper | Data | Explanation |
|---|---|---|---|---|
| `tab:rq1-toolsel`, `tab:rq1-full` | 2 + 2 | Star / CFH mean(interim) 18.4, error −3.6 | 20.7, −1.3 | The three Star branch rates in `results/run_topology_fanmerge/qwen_0.5b/*_fanmerge_cfhasr1_task*.json` are 13/50, 9/50 and 9/50 (26, 18, 18 %). No combination of stored fields gives 18.4. The other attacks' Star cells, computed the same way, match. |
| `tab:rq1-toolsel`, `tab:rq1-full` | 2 + 2 | Orchestrate+2 / BEAST 17.0, +7.0 | 16.7, +6.7 | Branch rates are 16/18/16 %. The paper appears to have carried the whole-percent value "17 %" from `tab:interim-full` into a one-decimal table. |
| `tab:rq2-crossfam` | 1 | CFH MAE 3.1 | 2.9 | This follows from the Star / CFH cell. With 18.4 plugged in, the MAE is 3.1. |
| `tab:tradingagents-leak` | 6 | Bear 0 %, three risk debaters 20 %, mean 22.9 %, AND 0.0 | Bear 40 %, no risk-debater fields, mean 35.0 % | The printed column equals `results/tradingagents/propagation_results.json`, not `secretleak_results.json`. `--ta-leak-source propagation` in `analysis/tables_bfcl_tradingagents.py` reproduces the paper's column exactly. |
| `tab:tradingagents-all` | 1 | Secret Leak downstream 0–20 % | 0–40 % | Same source as the row above. |
| BFCL section prose (`text:bfcl-null-suffix`) | 1 | Retargeted null-suffix baseline "0 %" | 1 % on the plain chain | One hit (task 59) in the retargeted null-suffix plain-chain run. |

## 2. Tables that match only under the original analysis choices

These tables match the paper exactly. Each one depends on an analysis choice worth knowing about. Each checker has a flag that shows the alternative.

- **`tab:frontier-estimator`: the AND/OR bounds are independence formulas.** The paper defines Fréchet/Boole bounds, OR = min(1, Σp) and AND = max(0, Σp − (k−1)). The original analysis script (not shipped) instead computed the independence formulas ∏p and 1 − ∏(1−p). `analysis/tables_frontier.py` reproduces the paper's printed values with those formulas by default. Using Fréchet/Boole changes 3 cells: GPT Centralized+2 0.4→0.0, Claude Centralized+2 0.1→0.0, Claude Star 4.9→5.0. Show them with `python3 analysis/tables_frontier.py --bound frechet`. All other tables use the paper's Fréchet/Boole definitions.
- **`tab:frontier-crossfam`: deep-tree hits are counted as zero.** The ASR1 aggregation reads `hd_final_verbatim`. Deep-tree result files store the same outcome as `hd_L3_verbatim`, so their hits count as 0 while their 50 tasks stay in the 650-task denominator. Counting them changes four cells: MASLEAK 9.8→10.8 (GPT) and 11.8→13.2 (Claude), FlowSteer 3.1→3.8 (GPT) and 3.7→4.5 (Claude).
- **`tab:interim-full`: Star+2 bounds are not valid.** The table prints OR = 38 and AND = 0 for Star+2. The appendix correctly notes that Star+2's interim measurement is already merged after the fan-in, so it cannot support a valid Fréchet bound.
- **`tab:realbfcl-attempt`: the fixed-target column has N = 92.** The caption says N = 100, but the fixed-target files cover tasks 0–91 (`results/run_beast_injected_pilot/`). File timestamps suggest the replay started while suffixes for the last 8 tasks were still being generated.
- **Rounding.** The paper rounds half away from zero and computes error columns from unrounded values (e.g. Mesh+2, qwen-1.5b: mean 6.5 → 7, error −1.5 → −2). The checkers do the same. `--rounding half_even` in `analysis/tables_topology.py` shows the one cell where Python's default rounding would differ.
- **Cell highlighting.** Several green or red highlights do not follow the rule stated in the caption: `tab:rq1-bestfit`, `tab:rq1-other`, `tab:rq1-full`, and the GPT-4o-mini Centralized cell of `tab:frontier-toolsel`. The checkers report these as information only.
- **Stale cells in the paper.** Some cells in `tab:frontier-crossfam` are marked "---" or "pending", but their data is complete. The recomputed values are printed by `python3 analysis/tables_frontier.py --table tab:frontier-crossfam`.
- **Target tools in the toy task set.** The prose says every toy task targets `auth_user`. In `masflow/tasks.py`, 20 of the 50 tasks target `auth_user`, 16 target `delete_file` and 14 target `move_file`.

## 3. Implementation notes

- **What "BEAST" is here.** `run_029_full_benchmark.run_beast` produces the frozen suffixes every replay uses. It is a beam search: each step makes uniformly random single-token swaps from the printable-ASCII vocabulary (`beast.beast_expand_beams`) and keeps the 4 lowest-loss beams. The published BEAST algorithm (Sadasivan et al.) instead builds the suffix left to right by sampling from the model's own top-k distribution.
- **The suffix optimisation loss scores only the first target token.** The target tool name is not appended to the input, so `target_slice` covers a single logit position. The loss is the cross-entropy of the first sub-token of the target tool name. The replay evaluation, by contrast, checks the actual generated tool call.
- **AgentLeak scoring.** In the topology replays, AgentLeak-F1 success is scored with the word-overlap check in `attack_protocol.py`, not with the vendored `AgentLeakTester`.
- **Frontier Mesh (+1) topology.** `topology_frontier_variants.py:74` loops `range(2)`, so the frontier Mesh (+1) variant has 3 peers, the same as Mesh. The local Mesh (+1) has 4.
- **Older driver version for the `mcp` results.** The MCP-protocol topology results were written by an earlier version of the replay drivers. Those files have no `protocol` key and extra `ft_stage*` fields. The shipped drivers produce the same `hd_*` outcomes: the A2A and raw runs made with the current code give interim and clean counts identical to the MCP runs. Our CPU replay of 10 Tree tasks also reproduces every stored outcome.
- **Suffix provenance.** The ASR1 payloads for the non-Tool-Selection families (`results/suffixes/attack_families/`) were produced by the family generators (`masflow.run_035_prompt_infection`, `run_039_agentleak_f1`, `run_040_masleak_v2`, `run_041_flowsteer`, `run_043_toma`, `run_045_evil_geniuses`), followed by a conversion step whose original script was not preserved. `scripts/convert_family_suffixes.py` reimplements that step. Run on the original generator outputs, it reproduces all 300 shipped ASR1 files exactly. ASR0 equals the payload constants in the `masflow/baseline_*` modules. `scripts/generate_family_suffixes.sh` runs the full chain. The paper says these suffixes were optimised with BEAST, but the driver docstrings say ACA v2.
- **Anchor used per attack.** BEAST uses its own suffix. CFH, TAMAS-DPI, MASLEAK v2, AgentLeak-F1 and FlowSteer use ASR1. Prompt Infection and TOMA use ASR0, because their ASR1 is 0 everywhere. This is the only mapping consistent with every cell. The paper does not state it.

## 4. What cannot be re-run

- **TradingAgents attack driver.** The script that injected the attack payloads into TradingAgents and recorded per-node leaks was not preserved. The package ships its outputs (`results/tradingagents/*.json`), the OpenAI-compatible model server it used (`masflow/local_llm_server.py`), and the upstream TradingAgents commit and container files (`third_party/tradingagents/`). The tables can be recounted from the JSONs but not regenerated. `analysis/NOTES_bfcl_tradingagents.md` describes the protocol as far as it can be inferred, clearly labelled as inference.
- **Exact launch commands.** No launcher survives for the toy-set depth-variant and graph replays, the qwen-1.5b and phi fan-merge replays, the family replays, the BFCL replays, or the frontier runs. The commands in `scripts/` are rebuilt from each driver's docstring and argparse definition and from the result-file layout. The four original launchers that the notes cite are in `scripts/original/`, kept for reference. Their paths have been anonymised and they are not meant to be run as-is.
- **BEAST tasks 10–49.** The surviving launcher and log for the toy-set BEAST suffixes cover tasks 0–9 only.
- **Bit-exact suffix optimisation.** Stage 1 (`scripts/generate_suffixes.sh`) is not bit-reproducible across GPUs and library builds. This is why the frozen suffixes are shipped and the replay scripts use them.
- **Frontier models.** `gpt-4o-mini` is an alias, not a dated snapshot. The runs date from 23–26 Sept 2026, and temperature 0 is not deterministic on these APIs. Re-runs should give close, but not identical, rates.
- **Unused Gemini runs.** `results/run_topology_frontier_generic/gemini_from_qwen_0.5b/` holds incomplete Gemini runs (stopped by a daily quota) that are not used in the paper.
