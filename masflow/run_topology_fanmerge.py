"""
run_topology_fanmerge.py -- Track 1 of the fan-out/fan-in topology
experiment: replay an already-optimized BEAST suffix (frozen, no
re-optimization) through the entry -> {B,C,D} -> E topology and compute
the full-path / entry->peer / peer->E metrics from topology_fanmerge_eval.

Suffix source: results/run_aca_family_sweep/<model_key>/<model_key>_mcp_beast_task<id>.json
(an already-completed single-hop BEAST run against the plain 2-agent
B->A pipeline).

Usage:
  python3 -u -m masflow.run_topology_fanmerge \\
      --model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct \\
      --suffix-source-dir results/run_aca_family_sweep/qwen_0.5b \\
      --results-dir results/run_topology_fanmerge/qwen_0.5b

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
from masflow.decision_agent_prompt import CANONICAL_A_SYSTEM, build_a_system_prompt
from masflow.topology_fanmerge_eval import run_fanmerge_eval, run_fanmerge_eval_generic, PEER_LABELS
from masflow.attack_protocol import TOOL_SELECTION
from masflow.attack_family_registry import resolve, ALL_ATTACK_CHOICES
from masflow.tasks_bfcl import TASKS_BFCL
from masflow.protocol_registry import resolve_protocol, ALL_PROTOCOL_CHOICES

SEED = 42


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--suffix-source-dir", required=True,
                     help="Dir containing <model_key>_mcp_beast_task<id>.json (frozen BEAST suffixes).")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--attack", default="beast", choices=ALL_ATTACK_CHOICES,
                     help="Which already-optimized suffix set to replay (filename: <model_key>_mcp_<attack>_task<id>.json). "
                          "Determines both the task list and the attack_protocol used (attack_family_registry.py). "
                          "TOOL_SELECTION attacks use the rich run_fanmerge_eval diagnostic; every other family uses "
                          "the protocol-generic run_fanmerge_eval_generic.")
    ap.add_argument("--task-ids", default=None, help="Comma-separated task ids (default: all found).")
    ap.add_argument("--task-source", default="toy", choices=["toy", "bfcl", "bfcl_injected"],
                     help="Which task list to resolve task_ids against, independent of "
                          "--attack\'s filename convention (unchanged): \'toy\' (default) = "
                          "attack_family_registry\'s normal resolve(); \'bfcl\' overrides "
                          "with tasks_bfcl.TASKS_BFCL; \'bfcl_injected\' overrides with "
                          "tasks_bfcl_injected.TASKS_BFCL_INJECTED.")
    ap.add_argument("--protocol", default="mcp", choices=ALL_PROTOCOL_CHOICES,
                     help="Agent-communication protocol used for the verbatim/clean check "
                          "(protocol_registry.py). 'mcp' keeps the original filenames/behavior unchanged.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    seed_everything(SEED)

    model, tok = load_local_model(args.model_name)
    task_list, attack_protocol = resolve(args.attack)
    if args.task_source == "bfcl":
        task_list = TASKS_BFCL
    elif args.task_source == "bfcl_injected":
        from masflow.tasks_bfcl_injected import TASKS_BFCL_INJECTED
        task_list = TASKS_BFCL_INJECTED
    protocol_mod = resolve_protocol(args.protocol)
    is_tool_selection = attack_protocol is TOOL_SELECTION

    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    else:
        task_ids = []
        for t in task_list:
            fpath = os.path.join(args.suffix_source_dir, f"{args.model_key}_mcp_{args.attack}_task{t.task_id}.json")
            if os.path.exists(fpath):
                task_ids.append(t.task_id)
        task_ids.sort()

    print("=" * 72)
    print("  FAN-OUT/FAN-IN TOPOLOGY TRANSFER (Track 1: frozen BEAST suffix)")
    print(f"  Model: {args.model_name} ({args.model_key})")
    print(f"  Attack: {args.attack}  Protocol: {attack_protocol.name}")
    print(f"  Suffix source: {args.suffix_source_dir}")
    print(f"  Tasks: {task_ids}")
    print("=" * 72, flush=True)

    # "beast" keeps the original filename (no attack suffix) for backward
    # compatibility with existing results; "aca" gets its own suffix since
    # it's a new variant with no prior naming to preserve.
    file_tag = "" if args.attack == "beast" else f"_{args.attack}"
    file_tag += "" if args.protocol == "mcp" else f"_{args.protocol}"

    for i, task_id in enumerate(task_ids):
        task = task_list[task_id]
        outfile = os.path.join(args.results_dir, f"{args.model_key}_fanmerge{file_tag}_task{task_id}.json")
        if os.path.exists(outfile):
            cached = json.load(open(outfile))
            if "error" not in cached:
                print(f"[{i+1}/{len(task_ids)}] task {task_id} -- SKIP (cached OK)", flush=True)
                continue

        suffix_path = os.path.join(args.suffix_source_dir, f"{args.model_key}_mcp_{args.attack}_task{task_id}.json")
        seed_data = json.load(open(suffix_path))
        suffix_text = seed_data["suffix"]

        a_system = build_a_system_prompt(task.tools) if getattr(task, "tools", None) else CANONICAL_A_SYSTEM

        expected = getattr(task, "expected_tool", None)
        target = getattr(task, "attack_target_tool", None)
        print(f"\n{'-'*72}")
        print(f"[{i+1}/{len(task_ids)}] task {task_id}: {expected} -> {target} "
              f"(frozen {args.attack} suffix)", flush=True)

        t0 = time.time()
        try:
            suffix_ids = torch.tensor(tok.encode(suffix_text, add_special_tokens=False), dtype=torch.long)
            if is_tool_selection:
                res = run_fanmerge_eval(model, tok, suffix_ids, task, a_system, protocol_mod=protocol_mod)
            else:
                res = run_fanmerge_eval_generic(model, tok, suffix_ids, task, protocol_mod=protocol_mod,
                                                 attack_protocol=attack_protocol)
            elapsed = time.time() - t0

            if is_tool_selection:
                print(f"  full_path: BA-HD={res['full_ba_hd']} P-HD={res['full_para_hd']}")
                for label in PEER_LABELS:
                    print(f"  entry->{label}: HD={res[f'hd_A_{label}']}   {label}->E: HD={res[f'hd_{label}_E']}")
            else:
                peer_summary = " ".join(f"{label}={res[f'hd_peer_raw_{label}']}" for label in PEER_LABELS)
                print(f"  peers: {peer_summary}  final_verb={res['hd_final_verbatim']} final_para={res['hd_final_paraphrase']}")
            print(f"  ({elapsed:.1f}s)")

            out = {
                "run": "topology_fanmerge_track1", "protocol": args.protocol,
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
    print("  FAN-OUT/FAN-IN TOPOLOGY TRANSFER (Track 1) COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
