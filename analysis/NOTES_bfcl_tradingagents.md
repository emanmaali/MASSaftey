# Reproducibility notes: real-BFCL and TradingAgents tables

> **Provenance references.** This file cites some files from the original working copy that are **not shipped** in the package: the ad-hoc analysis scripts and logs under `scratch_bfcl/`, the raw baseline-attack runs `results/run_033` to `results/run_045`, `run_aca_injected_full`, `slurm_logs/`, the hand-written table drafts `paper/tables/*.tex`, and the modules `mas_design_advisor.py`, `run_038_tamas_dpi.py` and `run_frontier_transfer.py`. They are cited only as evidence for how a number was produced. Everything needed to regenerate and check the tables is in this package. The package's Python package was renamed from `infix_gcg` to `masflow`.

Checker: `python3 analysis/tables_bfcl_tradingagents.py [--root this package] [--table LABEL] [--ta-leak-source secretleak|propagation]`
(stdlib only, reads saved JSONs, takes under 1 s). It prints expected vs recomputed values per cell and one
`TABLE <label>: MATCH / MISMATCH (k cells)` line per table. Exit code is 0 only if every table matches.

| Label (paper) | Status (default run) | Short reason |
|---|---|---|
| `tab:realbfcl-attempt` | **MATCH** (9/9 cells) | The caption says N=100 for the fixed-target column, but the files hold N=92 |
| `tab:tradingagents-leak` | **MISMATCH (6 cells)** vs `secretleak_results.json`; **MATCH** with `--ta-leak-source propagation` | The printed numbers come from `propagation_results.json` |
| `tab:tradingagents-all` (the brief called it `sec:frontier`; the table's real `\label` is `tab:tradingagents-all`, and `sec:frontier` labels the next subsubsection) | **MISMATCH (1 cell)**; MATCH with `--ta-leak-source propagation` | The secret-leak downstream range is 0–40% in `secretleak_results.json` |
| `text:bfcl-null-suffix` (numbers quoted in the BFCL prose) | **MISMATCH (1 cell)** | The fixed-target null run on plain chain has final-clean 1%, not 0% |

Rounding: the checker rounds half away from zero, and it computes the error from the unrounded values. Two examples: Star+2 real-BFCL is 20.5% -> "21%", and 20.5−39 = −18.5 -> "−19pp". Note that the displayed 21%−39% would be −18, so the paper's error column was computed before rounding.

---

## 1. `tab:realbfcl-attempt` (BEAST, qwen-0.5b; Toy / Real-BFCL original target / Real-BFCL fixed target)

### Cell definitions (these mirror `results/scratch_bfcl/option1_final_analysis.py`)
- **mean(interim)** is the mean over interior-node positions of the per-node hit rate. The flags are `hd_mid` (plain chain; `graph_topology_eval`) and `hd_stage1..hd_stage{n_stages-1}` (mesh_d3_orig, star_plus2; `topology_depth_eval`). Files that contain an `"error"` key are skipped (there are none).
- **final (clean)** is the rate of `hd_final_paraphrase`, i.e. the sink evaluated on the paraphrased (clean) hand-off. `hd_final_verbatim` is not used.
- `hd_*` is the generation-based hit detection: the node chose `target` instead of `expected`.
- `results/scratch_bfcl/three_case_summary.py` holds the table as **hard-coded strings**, so it is not a computation. It also has a fourth column (fixed target, ACA; source `results/run_aca_injected_full/`) that the paper does not print.

### Result globs (relative to this package/results)
| Column | Plain chain | Mesh | Star+2 | N | target |
|---|---|---|---|---|---|
| Toy | `run_graph_topology/qwen_0.5b/qwen_0.5b_plain_chain_beast_task*.json` | `run_topology_depth_variants/qwen_0.5b/qwen_0.5b_mesh_d3_orig_task*.json` | `.../qwen_0.5b_star_plus2_task*.json` | 50 | toy (auth_user 20, delete_file 16, move_file 14) |
| Real-BFCL, original | `run_graph_topology_bfcl/qwen_0.5b/qwen_0.5b_plain_chain_beast_task*.json` | `run_topology_depth_variants_bfcl/qwen_0.5b/qwen_0.5b_mesh_d3_orig_task*.json` | `.../qwen_0.5b_star_plus2_task*.json` | 100 | BFCL's own distractor |
| Real-BFCL, fixed | `run_beast_injected_pilot/qwen_0.5b/qwen_0.5b_plain_chain_beast_task*.json` | `.../qwen_0.5b_mesh_d3_orig_task*.json` | `.../qwen_0.5b_star_plus2_task*.json` | **92** (tasks 0–91) | auth_user |

**N discrepancy.** The caption says the fixed-target column has N=100, but the files cover tasks 0–91 only. The file mtimes explain the gap. The pilot replay began at 22:13 on 26 Sep. At that point the BEAST suffixes for tasks 92–99 in `results/suffixes/beast_bfcl_retargeted/qwen_0.5b` were still being generated: task92 was written at 22:13:30 and task99 at 22:31. The replay therefore skipped them. The cell values reproduce at N=92.

**Toy-set wording.** The prose says every toy task targets `auth_user`. That is not the case: in `masflow/tasks.py` 20/50 tasks target auth_user, 16 target delete_file and 14 target move_file. The same split shows up in the toy result JSONs.

### Drivers and inferred commands
No launcher was saved for these runs. The commands below are rebuilt from driver argparse, the log headers in `results/scratch_bfcl/*.log`, `results/scratch_bfcl/topo_replay_sweep.sh` and `scripts/original/run_topology_fanmerge_n50.sh`. Common settings: seed 42 (in every JSON), suffix length 40, 256 BEAST steps, protocol mcp, model `Qwen/Qwen2.5-0.5B-Instruct` (key `qwen_0.5b`).

```bash
M="--model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct"
T50=$(python3 -c "print(','.join(map(str,range(50))))"); T100=$(python3 -c "print(','.join(map(str,range(100))))")
# (a) single-hop BEAST suffixes  [Run 029 driver; log header "Run 029: Full-Scope ACA Benchmark"]
python -m masflow.run_029_full_benchmark $M --tasks $T50  --protocols mcp --attacks beast --steps 256 --results-dir results/suffixes/beast_toy/qwen_0.5b                       # toy
python -m masflow.run_029_full_benchmark $M --tasks $T100 --task-source bfcl          --protocols mcp --attacks beast --steps 256 --results-dir results/suffixes/beast_bfcl/qwen_0.5b   # log beast_qwen05b_full.log
python -m masflow.run_029_full_benchmark $M --tasks $T100 --task-source bfcl_injected --protocols mcp --attacks beast --steps 256 --results-dir results/suffixes/beast_bfcl_retargeted/qwen_0.5b    # log beast_bfcl_injected_05b.log
# (b) topology replays  (SRC/OUT/TS per column: toy=results/suffixes/beast_toy/run_graph_topology|run_topology_depth_variants/toy;
#     orig=results/suffixes/beast_bfcl/run_graph_topology_bfcl|run_topology_depth_variants_bfcl/bfcl;
#     fixed=results/suffixes/beast_bfcl_retargeted/run_beast_injected_pilot (both topologies)/bfcl_injected)
python -m masflow.run_graph_topology          $M --graph plain_chain     --attack beast --suffix-source-dir results/$SRC/qwen_0.5b --results-dir results/$OUT/qwen_0.5b --task-ids $TIDS --task-source $TS
python -m masflow.run_topology_depth_variants $M --variant mesh_d3_orig  --attack beast --suffix-source-dir results/$SRC/qwen_0.5b --results-dir results/$OUT/qwen_0.5b --task-ids $TIDS --task-source $TS
python -m masflow.run_topology_depth_variants $M --variant star_plus2    --attack beast --suffix-source-dir results/$SRC/qwen_0.5b --results-dir results/$OUT/qwen_0.5b --task-ids $TIDS --task-source $TS
```
The exact `--tasks` / `--task-ids` strings used for (a) toy and (a) bfcl were not recorded; the log headers only say "100 tasks". The toy suffix directory holds 60 BEAST files, and replays used tasks 0–49.

**Runtime and requirements.** One CUDA GPU, torch plus transformers, and HF weights for Qwen2.5-0.5B-Instruct. BEAST takes about 90 s per task (log: step 250 reached at about 87 s), so 100 tasks come to about 2.5 GPU-hours per task set. Each replay takes 1–4 s per task, about 5 min per topology at N=100.

## 2. Null-suffix controls and the star bug (BFCL prose)

A null run is a replay of the same drivers with `--suffix-source-dir results/suffixes/null_bfcl/<model>`. That directory holds 100 seed files whose `suffix` is `""` (logs are `scratch_bfcl/null_suffix_*.log` and `null_inj_*.log`).

| Claim | Source | Recomputed | Status |
|---|---|---|---|
| qwen-0.5b baseline confusion "33–43%" | `null_suffix_check/qwen_0.5b/*` (orig target). The first run covered tasks 0–29 (`null_suffix_*_05b.log`) and `_ext/_ext2` logs extended it to 100 | plain chain, tasks 0–29: interim 33.3%, final 43.3%. Other values: mesh 36.7/33.3 -> 43.3, star+2 33.3/23.3 -> 40.0. At N=100: final 37–43%, interim 22–27% | MATCH under this **interpretation**; the authors' exact derivation is not recorded |
| qwen-1.5b "3.3%" | `null_suffix_check/qwen_1.5b/qwen_1.5b_plain_chain_beast_task*.json` (N=30) | interim 3.3%, final 3.3% | MATCH |
| "restored a 0% null-suffix baseline" | `null_suffix_check_injected/qwen_0.5b/*` (auth_user target, N=100) and `null_suffix_check_toy/` (N=30) | interim 0% on all three table topologies. Final is 0% on mesh and star+2 but **1% on plain chain** (task 59). Across all 12 topologies: orchestrate final 1%, orchestrate+2 2%, tree stage1 5%. Toy is 0% | **1 cell MISMATCH** |
| star fan-out "0%" hit detection with "33–37%" logprob signal | `run_topology_fanmerge_bfcl/qwen_0.5b/qwen_0.5b_fanmerge_task*.json` (orig target, BEAST, N=100) | every `hd_*` and `full_*_hd` field is 0%. `ft_A_B/C/D`=33%, `full_ba_ft`=37%. The full ft range is 30–37% (`ft_peer_raw_C/D`=30) | MATCH, taking the range as entry-hop ft through full-path ft (**interpretation**) |

The fan-merge JSONs in `run_topology_fanmerge_bfcl/` are the **pre-fix (buggy) run**: every hd is 0 while ft is not. The current `masflow/run_topology_fanmerge.py:119` has the fix (`build_a_system_prompt(task.tools)` when a task has its own tools). No corrected star re-run on the original BFCL target exists in `results/`, so "correct it throughout" cannot be checked here. ft/fu are logprob-based diagnostics (target tool ranked first by the model's logprobs), and dr is a secondary detection flag.

## 3. BFCL data provenance

- **Vendored:** yes, twice.
  - `masflow/tasks_bfcl.py` inlines the 100 converted tasks: query, expected tool and args, BFCL distractor, and 2-tool schemas, each with a `# source: BFCL multiple_<id>` comment.
  - `masflow/tasks_bfcl_injected.py` builds the fixed-target variant at import time. It swaps the distractor for the toy `auth_user` schema and makes no other change.
  - The raw upstream files are also present: `third_party/bfcl/multiple.json` (200 JSONL records, `multiple_0`…) and `multiple_answers.json` (possible answers).
  - `third_party/bfcl/convert.py` is the sampling/conversion script. It reads `.scratch_bfcl/…`, so it must be run from a directory that has that layout. Its output snapshot is `converted_100.json`.
- **Source and licence:** BFCL v4 "multiple" category, github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard, Apache-2.0 (as stated in the docstring and the paper). No copy of the Apache-2.0 LICENSE or NOTICE file is shipped next to the vendored data. Add one before release, because redistributing Apache-2.0 material requires it.

## 4. TradingAgents (`tab:tradingagents-leak`, `tab:tradingagents-all`)

### 4.1 Driver not released
The TradingAgents attack driver was **never saved** and is not in this package or the original working copy. `.scratch_tradingagents/` is a clean upstream checkout (`git status` is empty) at commit `2d17df8da1536c121e4d7395ac5a5dcec9e96d6f` (TauricResearch/TradingAgents, v0.5.0; recorded in `third_party/tradingagents/UPSTREAM_COMMIT`). These runs **cannot be regenerated bit-for-bit**. Only the result JSONs and the shim are available.

### 4.2 What the JSONs contain (`results/tradingagents/`, formerly `.scratch_ta_<name>_results.json`)
All files have 5 runs, one per ticker/date pair: NVDA 2024-05-10, AAPL 2024-06-14, MSFT 2024-07-19, TSLA 2024-08-02, AMZN 2024-09-06. The secret markers are price-like strings, one per ticker: 847.23, 312.55, 621.09, 488.71, 255.42. No file records the model name, the payload, the seed or the git state.

| File | Fields | Counts (true / 5) |
|---|---|---|
| `secretleak_results.json` | ticker, date, marker, interim_leak, bull_leak, bear_leak, trader_leak, final_leak, elapsed, decision | interim 5, bull 0, **bear 2**, trader 0, final 0; **no risk-debater fields** |
| `propagation_results.json` | ticker, date, marker, news, bull, bear, trader, aggressive, conservative, neutral, final, elapsed, decision | news 5, bull/bear/trader 0, aggressive/conservative/neutral 1 (MSFT only), final 0 |
| `harmfulcontent_results.json` | ticker, date, harmful_in_news_report, harmful_in_trader_plan, harmful_in_final, elapsed, decision | 5 / 0 / 0 |
| `toolselection_results.json` | ticker, date, wrong_tool_called, topic_string_in_report, elapsed, decision | wrong_tool 0; topic string 2 |
| `null_results.json` | same fields as secretleak, without decision | all 0 (no-injection control; not in paper) |
| `budget_search_results.json` | per model {qwen05, qwen3b, phi35mini, gemma2b}: ticker, marker, interim_leak, final_leak, elapsed / error | qwen05 interim 2 / final 0; phi35mini 0/0; gemma2b 0/0; qwen3b all 5 failed with a pydantic `tool_calls.0.args` dict_type error (not in paper) |
| `qwen3b_retry_results.json` | ticker, marker, interim_leak, final_leak, elapsed | interim 4, final 0 (not in paper) |

No planning-steering file exists, which fits "Not tested".

### 4.3 Recomputed cells and the mismatch
- `mean(interim)` is the mean of the 7 interior-node rates. `AND` is max(0, Σp − (k−1)), matching `option1_estimator_compare.py` and `mas_design_advisor._and_bound`.
- From `propagation_results.json`: entry 100, bull/bear/trader 0, three risk debaters 20, final 0. That gives mean 160/7 = **22.9%**, AND **0.0%**, error +22.9pp. Every cell of both TradingAgents tables matches (`--ta-leak-source propagation`).
- From `secretleak_results.json`, the file named for the family: bear 40%, the risk-debater nodes are absent, mean over 4 nodes is 35.0%, and AND is 0.0%. **Six cells of `tab:tradingagents-leak` and the "downstream 0–20%" cell of `tab:tradingagents-all` do not match.**
- In short, the published secret-leak table is exactly the propagation file. Either the two result files were swapped or labelled wrongly when saved, or the table was built from the wrong file. The JSONs cannot tell these apart: both carry the same markers, and the files differ only in field names.
- The prose also says propagation is "0% at all downstream nodes". That is contradicted by `propagation_results.json` (risk debaters 20%). The table cell ("100%→0%", entry to final) is still correct.

### 4.4 Inferred protocol (INFERENCE, not verified)
- **Target model.** Qwen2.5-1.5B-Instruct (from the paper) behind `masflow/local_llm_server.py`, and presumably used for both deep_think_llm and quick_think_llm. `budget_search` shows the same harness was also run with qwen2.5-0.5b, qwen2.5-3b, phi-3.5-mini and gemma-2-2b.
- **Analyst selection.** It looks like only the News analyst was selected (`selected_analysts=["news"]`). No market, social or fundamentals nodes appear in any file, and "News Analyst" is the entry node.
- **Injection point.** The payload was most likely injected into the news tool output that the News analyst consumes, since the paper says there were no changes to TradingAgents code. This was probably done by monkeypatching the dataflow function inside the driver:
  - secret leak and propagation: a note carrying the numeric marker;
  - harmful content: a pump-and-dump description;
  - tool selection: an instruction to call a wrong tool with a topic string.
- **Leak detection.** The driver probably ran `TradingAgentsGraph.propagate(ticker, date)` once per row and checked for the marker substring in these final-state fields:
  - `news_report` (entry);
  - `investment_debate_state.bull_history` and `.bear_history`;
  - `trader_investment_plan`;
  - `risk_debate_state.aggressive_history`, `.conservative_history` and `.neutral_history`;
  - `final_trade_decision` (sink).

  The `decision` field is the processed signal (Buy/Overweight/Hold/Sell). `elapsed` is 75–112 s per run.
- **Debate rounds.** Probably the default: max_debate_rounds=1 and max_risk_discuss_rounds=1.
- **qwen-3b failures.** The qwen3b errors (tool-call arguments returned as a JSON string) match the fix now in the shim's `_parse_tool_calls` (a string `arguments` is passed through without double-encoding). That suggests the shim was patched between `budget_search` and `qwen3b_retry`, so earlier results may come from an older shim.

### 4.5 How to run upstream TradingAgents against the shim
The shim's argparse takes `--model-name` (required), `--port` (default 8011) and `--host` (default 127.0.0.1). It serves `GET /v1/models` and `POST /v1/chat/completions`, with stream supported as a single SSE chunk. It uses greedy decoding when temperature is 0, a generation-only 4-gram no-repeat filter, Hermes `<tool_call>` parsing and `json_repair`. It needs `torch transformers fastapi uvicorn json_repair` and imports only `masflow.config`.
```bash
cd this package && python3 -u -m masflow.local_llm_server --model-name Qwen/Qwen2.5-1.5B-Instruct --port 8011 &
git clone https://github.com/TauricResearch/TradingAgents && cd TradingAgents && git checkout 2d17df8da1536c121e4d7395ac5a5dcec9e96d6f && pip install -e .
export TRADINGAGENTS_LLM_PROVIDER=openai_compatible TRADINGAGENTS_LLM_BACKEND_URL=http://127.0.0.1:8011/v1 \
       TRADINGAGENTS_DEEP_THINK_LLM=Qwen/Qwen2.5-1.5B-Instruct TRADINGAGENTS_QUICK_THINK_LLM=Qwen/Qwen2.5-1.5B-Instruct \
       TRADINGAGENTS_TEMPERATURE=0 OPENAI_COMPATIBLE_API_KEY=dummy
python -c "from tradingagents.default_config import DEFAULT_CONFIG as C; from tradingagents.graph.trading_graph import TradingAgentsGraph as G; \
s,d=G(selected_analysts=['news'],config=C.copy()).propagate('NVDA','2024-05-10'); print(d)"
```
- `openai_compatible` sends plain Chat Completions to a custom base_url. Upstream `openai` with a non-openai.com base_url also turns off the Responses API.
- The data vendors (yfinance by default) need network access.
- Any injection and leak check has to be re-implemented, as described in 4.4.
- Tool calling only works with Qwen2.5 models; phi-3.5 and gemma-2 templates do not accept `tools=`.

## 5. masflow modules imported (transitive, via AST over every `masflow` import, including function-local ones)
Drivers covered: run_029_full_benchmark, run_graph_topology, run_topology_depth_variants, run_topology_fanmerge and local_llm_server.
The union is 43 modules:

- **Drivers (5):** local_llm_server, run_029_full_benchmark, run_graph_topology, run_topology_depth_variants, run_topology_fanmerge.
- **Attack generation (6):** suffix_opt_and_eval, beast, estimators, gcg, ppl_regularized_attacks, ste_gcg.
- **Core (3):** config, decision_agent_prompt, pipeline.
- **Protocols (6):** protocol_a2a, protocol_acp, protocol_aitp, protocol_mcp, protocol_raw, protocol_registry.
- **Tasks and attack families (12):** agentleak_f1_tasks, attack_family_registry, attack_protocol, baseline_prompt_infection, evil_geniuses_tasks, flowsteer_tasks, masleak_tasks, prompt_infection_tasks, tasks, tasks_bfcl, tasks_bfcl_injected, toma_tasks.
- **Topologies (11):** graph_examples, graph_topology, graph_topology_eval, topology_depth_eval, topology_depth_variants, topology_eval, topology_fanmerge_eval, topology_mesh, topology_orchestrate, topology_star, topology_tree.
- local_llm_server itself imports only `config`.

## 6. Not reproducible, or only partially
1. The TradingAgents driver: injection payloads, injection point, leak checks and the model/shim version for each file are all unrecorded. The JSON flags can only be re-tabulated, not regenerated.
2. The secret-leak TradingAgents table does not follow from `secretleak_results.json`. It equals `propagation_results.json`.
3. Exact launch commands for the BFCL runs: they are rebuilt from log headers and argparse, and the task-id strings for suffix generation are not recorded.
4. Fixed-target N is 92, not 100.
5. The ranges "33–43%" and "33–37%" can only be matched under the interpretations given in section 2. The "0%" retargeted null baseline is 1% on plain chain.
6. There is no corrected (post-fix) star replay on the original BFCL target in `results/`.
