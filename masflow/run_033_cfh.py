"""
Run 033: CFH (Control-Flow Hijacking) reproduction — ASR0 (pajaMAS's real
hand-crafted payload) vs. ASR1 (ACA v2-optimised), per
experimental_protocol.md §11.3.

Model: Qwen2.5-0.5B-Instruct (rescoped 2026-08-21 from smol_135m — the
cheaper scope risked the same inconclusive 0%-B-A-HD tie seen in the
GCG/BEAST/G-BEAST/ACA comparison at smol_135m/steps=96).
Protocol: MCP. Tasks: all 50 (masflow.tasks — CFH's threat model maps
directly onto expected_tool/attack_target_tool, no new dataset needed).
Steps: 256 (matches run_022/run_037's scope, not the reduced budget).

Usage:
  CUDA_VISIBLE_DEVICES=0 HF_HOME=.hf_cache \\
    python -u -m masflow.run_033_cfh 2>&1 | tee experiment_logs/run_033.log

  # Real BFCL tasks, any model in the roster, separate results dir:
  python -u -m masflow.run_033_cfh \\
      --model-key qwen_1.5b --model-name Qwen/Qwen2.5-1.5B-Instruct \\
      --task-source bfcl --results-dir results/run_cfh_bfcl/qwen_1.5b

Resumable: any per-(mode,task) result already written without an "error"
key is skipped.
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from masflow.config import load_local_model, seed_everything
from masflow.gcg import get_ascii_printable_tokens
from masflow.tasks import TASKS
from masflow.tasks_bfcl import TASKS_BFCL
from masflow.decision_agent_prompt import build_a_system_prompt
from masflow import suffix_opt_and_eval, baseline_cfh

SEED = 42
STEPS = 256
BEAM_WIDTH = 4

_DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "run_033")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="qwen_0.5b")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--task-source", default="toy", choices=["toy", "bfcl"],
                     help="'toy' (default) = masflow.tasks.TASKS, unchanged original "
                          "behavior; 'bfcl' = tasks_bfcl.TASKS_BFCL (100 real BFCL tasks).")
    ap.add_argument("--results-dir", default=_DEFAULT_RESULTS_DIR)
    return ap.parse_args()


def main():
    args = parse_args()
    task_list = TASKS_BFCL if args.task_source == "bfcl" else TASKS
    all_task_ids = [t.task_id for t in task_list]
    os.makedirs(args.results_dir, exist_ok=True)

    seed_everything(SEED)
    total = len(all_task_ids) * 2

    print("=" * 72)
    print("  RUN 033: CFH REPRODUCTION — ASR0 (hand-crafted) vs ASR1 (ACA v2)")
    print(f"  Model: {args.model_name}  Tasks: {len(all_task_ids)} ({args.task_source})  Steps: {STEPS}")
    print("=" * 72)

    model, tok = load_local_model(args.model_name)
    allowed_tokens = get_ascii_printable_tokens(tok)
    default_a_system = suffix_opt_and_eval.A_EXPLICIT

    run_idx = 0
    for task_id in all_task_ids:
        task = task_list[task_id]
        a_system = build_a_system_prompt(task.tools) if getattr(task, "tools", None) else default_a_system

        # --- ASR0: baseline ---
        run_idx += 1
        fpath0 = os.path.join(args.results_dir, f"{args.model_key}_cfh_asr0_task{task_id}.json")
        if not (os.path.exists(fpath0) and "error" not in json.load(open(fpath0))):
            print(f"[{run_idx:3d}/{total}] ASR0 task {task_id}")
            try:
                res = baseline_cfh.run_cfh_baseline(model, tok, task, a_system)
                out = {"mode": "asr0", "attack": "cfh", "model": args.model_name,
                       "task_id": task_id, "expected": task.expected_tool,
                       "target": task.attack_target_tool, **res}
                json.dump(out, open(fpath0, "w"), indent=2)
            except Exception as e:
                json.dump({"error": str(e)}, open(fpath0, "w"))
        else:
            print(f"[{run_idx:3d}/{total}] ASR0 task {task_id} — SKIP (cached)")

        # --- ASR1: ACA-optimised ---
        run_idx += 1
        fpath1 = os.path.join(args.results_dir, f"{args.model_key}_cfh_asr1_task{task_id}.json")
        if os.path.exists(fpath1) and "error" not in json.load(open(fpath1)):
            print(f"[{run_idx:3d}/{total}] ASR1 task {task_id} — SKIP (cached)")
            continue

        print(f"[{run_idx:3d}/{total}] ASR1 task {task_id}")
        try:
            a_ids, suffix_pos, target_ids, target_slice = baseline_cfh.build_cfh_input(
                tok, task, a_system, suffix_len=40
            )
            # No B-stage here (CFH's payload IS the tool output — no
            # separate paraphraser to optimise against), so lambda_b=0.
            import torch
            dummy_para_ids = tok.encode("ok", add_special_tokens=False)
            res = suffix_opt_and_eval.run_aca_v2(
                model, tok, a_ids, suffix_pos, target_ids, target_slice,
                torch.tensor([], dtype=torch.long), dummy_para_ids, allowed_tokens, task, a_system,
                steps=STEPS, beam_width=BEAM_WIDTH,
                lambda_b=0.0, lambda_ppl=1.0, lambda_ba=0.0,
                pct_a=90, pct_b=0, pct_r=10, loss_mode="ce",
            )
            suffix_text = tok.decode(res["best_ids"][suffix_pos], skip_special_tokens=True)
            tool_output = baseline_cfh.CFH_CONTEXT_TEMPLATE.format(payload=suffix_text)
            import masflow.protocol_mcp as p_mcp
            a_content = p_mcp.format_message_for_a(task.user_task, tool_output)
            eval_res = suffix_opt_and_eval.eval_a_content(model, tok, a_content, a_system, task)
            out = {"mode": "asr1", "attack": "cfh", "model": args.model_name,
                   "task_id": task_id, "expected": task.expected_tool,
                   "target": task.attack_target_tool, "suffix": suffix_text,
                   "a_only_loss": res["a_only_attack_loss"], **eval_res}
            json.dump(out, open(fpath1, "w"), indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"error": str(e)}, open(fpath1, "w"))

    print("RUN 033 COMPLETE")


if __name__ == "__main__":
    main()
