"""
run_topology_frontier_generic.py -- topology-level frontier transfer
pilot for the four non-tool_selection attack families (secret_leak,
propagation, planning_steering, harmful_content), completing RQ1's
cross-family validation for frontier targets. Same suffix-replay idea as
run_topology_frontier.py, but resolves (task_list, attack_protocol) from
--attack via attack_family_registry.py (e.g. masleakasr1, tomaasr1,
flowsteerasr1, agentleakf1asr1, egasr1) instead of hardcoding
tool_selection/TASKS_BFCL.

Usage:
  python3 -u -m masflow.run_topology_frontier_generic \\
      --attack tomaasr1 \\
      --suffix-source-dir results/run_secretleak_propagation_harmful_as_suffix/qwen_0.5b \\
      --source-model-key qwen_0.5b --frontier-model gpt-4o-mini \\
      --topology mesh \\
      --results-dir results/run_topology_frontier_generic/gpt-4o-mini_from_qwen_0.5b/tomaasr1/mesh

Resumable: any per-task result already written without an "error" key is
skipped.
"""
import argparse
import json
import os
import time

from masflow.attack_family_registry import resolve, ALL_ATTACK_CHOICES
from masflow.frontier_client import FrontierClient
from masflow.topology_eval_frontier_generic import (
    run_depth_variant_eval_frontier_generic,
    run_deep_tree_eval_frontier_generic,
    run_fanmerge_eval_frontier_generic,
)

TOPOLOGY_CHOICES = [
    "centralized", "orchestrate", "tree", "mesh", "star",
    "centralized_d2", "orchestrate_d3", "mesh_d4", "deep_tree",
    "centralized_plus2", "orchestrate_plus2", "tree_plus2", "mesh_plus2",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True, choices=ALL_ATTACK_CHOICES)
    ap.add_argument("--suffix-source-dir", required=True)
    ap.add_argument("--source-model-key", default="qwen_0.5b")
    ap.add_argument("--frontier-model", required=True)
    ap.add_argument("--topology", required=True, choices=TOPOLOGY_CHOICES)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--task-ids", default="all")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    task_list, attack_protocol = resolve(args.attack)
    task_ids = [t.task_id for t in task_list] if args.task_ids == "all" else [int(x) for x in args.task_ids.split(",")]
    client = FrontierClient(args.frontier_model)

    print("=" * 72)
    print(f"  CROSS-FAMILY TOPOLOGY FRONTIER: {args.attack} ({attack_protocol.name}) | "
          f"{args.topology} | {args.frontier_model} <- {args.source_model_key} suffixes")
    print(f"  Tasks: {len(task_ids)}")
    print("=" * 72, flush=True)

    n_hit = 0
    n_done = 0
    for i, task_id in enumerate(task_ids):
        task = task_list[task_id]
        outfile = os.path.join(args.results_dir, f"task{task_id}.json")
        if os.path.exists(outfile):
            cached = json.load(open(outfile))
            if "error" not in cached:
                print(f"[{i+1}/{len(task_ids)}] task {task_id} -- SKIP (cached)", flush=True)
                n_done += 1
                n_hit += cached.get("hd_final_verbatim") or cached.get("hd_L3_verbatim") or False
                continue

        suffix_path = os.path.join(args.suffix_source_dir, f"{args.source_model_key}_mcp_{args.attack}_task{task_id}.json")
        seed_data = json.load(open(suffix_path))
        suffix_text = seed_data["suffix"]

        t0 = time.time()
        try:
            if args.topology == "star":
                res = run_fanmerge_eval_frontier_generic(client, suffix_text, task, attack_protocol)
                hit = res["hd_final_verbatim"]
            elif args.topology == "deep_tree":
                res = run_deep_tree_eval_frontier_generic(client, suffix_text, task, attack_protocol)
                hit = res["hd_L3_verbatim"]
            else:
                res = run_depth_variant_eval_frontier_generic(client, suffix_text, task, args.topology, attack_protocol)
                hit = res["hd_final_verbatim"]
            elapsed = time.time() - t0
            n_done += 1
            n_hit += hit
            print(f"[{i+1}/{len(task_ids)}] task {task_id}  HIT={hit}  ({elapsed:.1f}s)", flush=True)
            out = {"task_id": task_id, "attack": args.attack, "family": attack_protocol.name,
                   "topology": args.topology, "source_model_key": args.source_model_key,
                   "frontier_model": args.frontier_model, "elapsed": elapsed, **res}
            json.dump(out, open(outfile, "w"), indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"error": str(e), "task_id": task_id}, open(outfile, "w"), indent=2)

    print("\n" + "=" * 72)
    print(f"  CROSS-FAMILY TOPOLOGY FRONTIER COMPLETE ({args.attack}, {args.topology}): {n_hit}/{n_done} hits")
    print("=" * 72)


if __name__ == "__main__":
    main()
