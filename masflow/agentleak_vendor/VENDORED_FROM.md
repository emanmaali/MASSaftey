Vendored from https://github.com/Privatris/AgentLeak (MIT License, see
AGENTLEAK_LICENSE), commit as of 2026-09-16, for use as a fully local,
free detector (`AgentLeakTester(mode=DetectionMode.FAST)`, Tier1
exact/canary + Tier2 Presidio pattern matching -- no LLM-judge API calls)
in `masflow/baseline_agentleak_f1.py`. Only the `agentleak/` package
directory is vendored (not the CLI, benchmarks, or agentleak_data/ --
our own agentleak_f1_tasks.py supplies the task/scenario data instead).
