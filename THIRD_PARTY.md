# Third-party code and data

| Component | Where in this package | Origin | Licence |
|---|---|---|---|
| AgentLeak (detector package) | `masflow/agentleak_vendor/` | github.com/Privatris/AgentLeak | MIT (`masflow/agentleak_vendor/AGENTLEAK_LICENSE`, `VENDORED_FROM.md`) |
| TAMAS DPI instances and tool schemas | `masflow/tamas_data/` | github.com/microsoft/TAMAS (arXiv:2511.05269) | Code MIT, data CDLA-Permissive-2.0 (`masflow/tamas_data/TAMAS_LICENSE*`) |
| BFCL v4 "multiple" tasks | `third_party/bfcl/multiple*.json` (+ `convert.py`), `masflow/tasks_bfcl*.py` | Berkeley Function-Calling Leaderboard (gorilla) | Apache-2.0 (`third_party/bfcl/NOTICE`) |
| TradingAgents (not vendored) | `third_party/tradingagents/` (Dockerfile, compose file, licence, commit id) | github.com/TauricResearch/TradingAgents @ `third_party/tradingagents/UPSTREAM_COMMIT` | Apache-2.0 (`third_party/tradingagents/LICENSE`) |
| CFH real payload (one string constant) | `masflow/baseline_cfh.py` | github.com/trailofbits/pajaMAS demos | See upstream repository |
| Attack payload reimplementations: Prompt Infection, MASLEAK, TOMA, FlowSteer, Evil Geniuses | `masflow/baseline_*.py`, `masflow/*_tasks.py` | Re-implemented from the cited papers (arXiv 2410.07283, 2505.12442, 2512.04129, 2605.11514, 2311.11855) | Our code; payload text quoted from the papers |

Models are downloaded from the Hugging Face Hub under their own licences:
Qwen2.5-0.5B/1.5B-Instruct (Apache-2.0), Phi-3.5-mini-instruct (MIT),
gemma-2-2b-it (Gemma Terms of Use, gated, needs `HF_TOKEN`).
