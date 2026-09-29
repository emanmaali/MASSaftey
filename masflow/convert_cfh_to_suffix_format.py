"""
convert_cfh_to_suffix_format.py -- converts CFH's ASR0 (fixed hand-crafted
payload, results/run_033/*_cfh_asr0_task*.json, no per-task suffix field)
and ASR1 (per-task ACA-optimized payload, has a "suffix" field) into the
standard {model_key}_mcp_{attack}_task{id}.json suffix-file format every
topology driver in this project already reads (--suffix-source-dir).

This lets CFH's payloads be replayed through every existing topology
(centralized/orchestrate/tree/mesh/star and depth variants, plain_chain/
plain_diamond) with ZERO changes to any topology driver -- CFH's payload
is treated as if appended to the entry query, the same injection slot
BEAST/ACA suffixes use (a deliberate adaptation: CFH's real mechanism
injects via a simulated tool-output field at the final decision point
only, which has no multi-hop journey through a topology to measure; this
conversion tests the different, well-defined question "does this text,
if injected at the entry point, survive the same way BEAST/ACA's does").

Usage:
  python3 -m masflow.convert_cfh_to_suffix_format \\
      --model-key qwen_0.5b --out-dir results/run_cfh_as_suffix/qwen_0.5b
"""
import argparse
import json
import os

from masflow.baseline_cfh import CFH_REAL_PAYLOAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="qwen_0.5b")
    ap.add_argument("--cfh-results-dir", default="results/run_033")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=50,
                     help="50 for the original toy task set (default, unchanged behavior); "
                          "100 for the real BFCL task set (results/run_cfh_bfcl).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    n_asr0 = n_asr1 = 0
    for task_id in range(args.n_tasks):
        asr0_path = os.path.join(args.cfh_results_dir, f"{args.model_key}_cfh_asr0_task{task_id}.json")
        asr1_path = os.path.join(args.cfh_results_dir, f"{args.model_key}_cfh_asr1_task{task_id}.json")

        if os.path.exists(asr0_path):
            json.dump({"task_id": task_id, "suffix": CFH_REAL_PAYLOAD, "source": "cfh_asr0_fixed_payload"},
                       open(os.path.join(args.out_dir, f"{args.model_key}_mcp_cfhasr0_task{task_id}.json"), "w"),
                       indent=2)
            n_asr0 += 1

        if os.path.exists(asr1_path):
            j = json.load(open(asr1_path))
            json.dump({"task_id": task_id, "suffix": j["suffix"], "source": "cfh_asr1_aca_optimized"},
                       open(os.path.join(args.out_dir, f"{args.model_key}_mcp_cfhasr1_task{task_id}.json"), "w"),
                       indent=2)
            n_asr1 += 1

    print(f"Converted {n_asr0} ASR0 (fixed payload) + {n_asr1} ASR1 (optimized) suffix files -> {args.out_dir}")


if __name__ == "__main__":
    main()
