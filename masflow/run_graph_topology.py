"""
run_graph_topology.py -- replays a frozen BEAST suffix through one of
the generic graph examples (plain_chain, plain_diamond), instrumented
per-node via graph_topology_eval.

Usage:
  python3 -u -m masflow.run_graph_topology \\
      --model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct \\
      --suffix-source-dir results/run_aca_family_sweep/qwen_0.5b \\
      --results-dir results/run_graph_topology/qwen_0.5b \\
      --graph plain_diamond

Resumable: any per-task result already written without an "error" key is
skipped.
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from masflow.config import load_local_model, seed_everything
from masflow.graph_examples import GRAPHS
from masflow.graph_topology_eval import run_graph_eval
from masflow.attack_family_registry import resolve, ALL_ATTACK_CHOICES
from masflow.tasks_bfcl import TASKS_BFCL
from masflow.protocol_registry import resolve_protocol, ALL_PROTOCOL_CHOICES

SEED = 42


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--suffix-source-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--graph", required=True, choices=list(GRAPHS.keys()))
    ap.add_argument("--attack", default="beast", choices=ALL_ATTACK_CHOICES,
                     help="Which already-optimized suffix set to replay (filename: <model_key>_mcp_<attack>_task<id>.json). "
                          "Determines both the task list and the attack_protocol used (attack_family_registry.py).")
    ap.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--task-source", default="toy", choices=["toy", "bfcl", "bfcl_injected"],
                     help="Which task list to resolve task_ids against, independent of "
                          "--attack\'s filename convention (unchanged): \'toy\' (default) = "
                          "attack_family_registry\'s normal resolve(); \'bfcl\' overrides "
                          "with tasks_bfcl.TASKS_BFCL; \'bfcl_injected\' overrides with "
                          "tasks_bfcl_injected.TASKS_BFCL_INJECTED (real BFCL queries/tools, "
                          "attack_target_tool replaced by an injected, semantically-irrelevant "
                          "auth_user target -- restores a clean attack-attributable signal).")
    ap.add_argument("--protocol", default="mcp", choices=ALL_PROTOCOL_CHOICES,
                     help="Agent-communication protocol used for the verbatim/clean check "
                          "(protocol_registry.py). 'mcp' keeps the original filenames/behavior unchanged.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    seed_everything(SEED)

    graph = GRAPHS[args.graph]
    model, tok = load_local_model(args.model_name)
    task_ids = [int(x) for x in args.task_ids.split(",")]
    task_list, attack_protocol = resolve(args.attack)
    if args.task_source == "bfcl":
        task_list = TASKS_BFCL
    elif args.task_source == "bfcl_injected":
        from masflow.tasks_bfcl_injected import TASKS_BFCL_INJECTED
        task_list = TASKS_BFCL_INJECTED
    protocol_mod = resolve_protocol(args.protocol)
    file_tag = "" if args.protocol == "mcp" else f"_{args.protocol}"

    print("=" * 72)
    print(f"  GRAPH TOPOLOGY: {args.graph}")
    print(f"  Model: {args.model_name} ({args.model_key})")
    print(f"  Attack: {args.attack}  Protocol: {attack_protocol.name}  Comm.\\ protocol: {args.protocol}")
    print(f"  Tasks: {task_ids}")
    print("=" * 72, flush=True)

    for i, task_id in enumerate(task_ids):
        task = task_list[task_id]
        outfile = os.path.join(args.results_dir, f"{args.model_key}_{args.graph}_{args.attack}{file_tag}_task{task_id}.json")
        if os.path.exists(outfile):
            cached = json.load(open(outfile))
            if "error" not in cached:
                print(f"[{i+1}/{len(task_ids)}] task {task_id} -- SKIP (cached)", flush=True)
                continue

        suffix_path = os.path.join(args.suffix_source_dir, f"{args.model_key}_mcp_{args.attack}_task{task_id}.json")
        seed_data = json.load(open(suffix_path))
        suffix_text = seed_data["suffix"]

        expected = getattr(task, "expected_tool", None)
        target = getattr(task, "attack_target_tool", None)
        print(f"\n{'-'*72}")
        print(f"[{i+1}/{len(task_ids)}] task {task_id}: {expected} -> {target}", flush=True)

        t0 = time.time()
        try:
            suffix_ids = torch.tensor(tok.encode(suffix_text, add_special_tokens=False), dtype=torch.long)
            res = run_graph_eval(model, tok, suffix_ids, task, graph, protocol_mod=protocol_mod,
                                  attack_protocol=attack_protocol)
            elapsed = time.time() - t0
            node_summary = " ".join(f"{n.name}={res.get(f'hd_{n.name}')}" for n in graph.nodes if n.name != graph.sink)
            print(f"  {node_summary}  final_verb={res['hd_final_verbatim']} final_para={res['hd_final_paraphrase']}  ({elapsed:.1f}s)")

            out = {
                "run": "graph_topology", "graph": args.graph, "attack": args.attack, "protocol": args.protocol,
                "model": args.model_name, "model_key": args.model_key,
                "task_id": task_id, "expected": expected, "target": target,
                "suffix": suffix_text, "seed": SEED, "time_seconds": elapsed,
                **res,
            }
            json.dump(out, open(outfile, "w"), indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"error": str(e), "task_id": task_id}, open(outfile, "w"), indent=2)

    print("\n" + "=" * 72)
    print(f"  GRAPH TOPOLOGY {args.graph} COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
