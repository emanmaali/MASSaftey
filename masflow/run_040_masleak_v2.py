"""
Run 040: MASLEAK like-for-like redo -- Human + GCG ASR0 anchors vs ACA v2
ASR1 with a full-prompt L_A target and a qPropagate relay hop, per
experimental_protocol.md §19.1. Part 1.

Writes THREE result files per task:
  qwen_0.5b_masleak_v2_human_task{i}.json   -- ASR0 anchor A (their template)
  qwen_0.5b_masleak_v2_gcg_task{i}.json     -- ASR0 anchor B (their GCG obj.)
  qwen_0.5b_masleak_v2_aca_task{i}.json     -- ASR1 (ACA + relay-hop loss)

Model: Qwen2.5-0.5B-Instruct, steps=256, MAX_TASKS (default 10).

Usage:
  MAX_TASKS=10 CUDA_VISIBLE_DEVICES=0 HF_HOME=.hf_cache \\
    python -u -m masflow.run_040_masleak_v2 2>&1 | tee experiment_logs/run_040.log
"""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from masflow.config import load_local_model, seed_everything
from masflow.gcg import get_ascii_printable_tokens
from masflow.masleak_tasks import MASLEAK_TASKS
from masflow import suffix_opt_and_eval, baseline_masleak_v2 as mv2

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_KEY = "qwen_0.5b"
SEED = 42
STEPS = 256
BEAM_WIDTH = 4
BA_EVERY = 8
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "run_040")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _skip(fpath):
    return os.path.exists(fpath) and "error" not in json.load(open(fpath))


def main():
    seed_everything(SEED)
    tasks = MASLEAK_TASKS[:MAX_TASKS] if MAX_TASKS else MASLEAK_TASKS
    total = len(tasks) * 3

    print("=" * 72)
    print("  RUN 040: MASLEAK LIKE-FOR-LIKE — Human + GCG anchors vs ACA v2 (relay)")
    print(f"  Model: {MODEL_NAME}  Tasks: {len(tasks)}  Steps: {STEPS}")
    print("=" * 72)

    model, tok = load_local_model(MODEL_NAME)
    allowed_tokens = get_ascii_printable_tokens(tok)
    device = next(model.parameters()).device

    run_idx = 0
    for task in tasks:
        base = {"attack": "masleak_v2", "model": MODEL_NAME,
                "task_id": task.task_id, "domain": task.domain}

        # --- ASR0 anchor A: Human template ---
        run_idx += 1
        fh = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_masleak_v2_human_task{task.task_id}.json")
        if not _skip(fh):
            print(f"[{run_idx:3d}/{total}] HUMAN  task {task.task_id} ({task.domain})")
            try:
                res = mv2.run_masleak_human(model, tok, task, device)
                json.dump({"variant": "human", **base, **res}, open(fh, "w"), indent=2)
            except Exception as e:
                json.dump({"error": str(e)}, open(fh, "w"))
        else:
            print(f"[{run_idx:3d}/{total}] HUMAN  task {task.task_id} — SKIP (cached)")

        # --- ASR0 anchor B: vanilla GCG toward full-prompt regurgitation ---
        run_idx += 1
        fg = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_masleak_v2_gcg_task{task.task_id}.json")
        if not _skip(fg):
            print(f"[{run_idx:3d}/{total}] GCG    task {task.task_id} ({task.domain})")
            try:
                res = mv2.run_masleak_gcg(model, tok, task, allowed_tokens, device,
                                          steps=STEPS, beam_width=BEAM_WIDTH)
                json.dump({"variant": "gcg", **base, **res}, open(fg, "w"), indent=2)
            except Exception as e:
                import traceback
                traceback.print_exc()
                json.dump({"error": str(e)}, open(fg, "w"))
        else:
            print(f"[{run_idx:3d}/{total}] GCG    task {task.task_id} — SKIP (cached)")

        # --- ASR1: ACA v2, full-prompt L_A + relay-hop multi-hop loss ---
        run_idx += 1
        fa = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_masleak_v2_aca_task{task.task_id}.json")
        if _skip(fa):
            print(f"[{run_idx:3d}/{total}] ACA    task {task.task_id} — SKIP (cached)")
            continue
        print(f"[{run_idx:3d}/{total}] ACA    task {task.task_id} ({task.domain})")
        try:
            a_ids, suffix_pos, target_ids, target_slice = mv2.build_masleak_v2_input(
                tok, task, suffix_len=40
            )
            dummy_para_ids = tok.encode("ok", add_special_tokens=False)
            res = suffix_opt_and_eval.run_aca_v2(
                model, tok, a_ids, suffix_pos, target_ids, target_slice,
                torch.tensor([], dtype=torch.long), dummy_para_ids, allowed_tokens,
                task, mv2.RELAY_SYSTEM_PROMPT,
                steps=STEPS, beam_width=BEAM_WIDTH,
                lambda_b=0.0, lambda_ppl=1.0, lambda_ba=1.0,
                pct_a=90, pct_b=0, pct_r=10, loss_mode="ce",
                ba_every=BA_EVERY, ba_score_fn=mv2.compute_masleak_relay_loss,
            )
            eval_res = mv2.evaluate_masleak_v2_suffix(
                model, tok, task, res["best_ids"][suffix_pos], device
            )
            json.dump({"variant": "aca", **base,
                       "a_only_loss": res["a_only_attack_loss"], **eval_res},
                      open(fa, "w"), indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"error": str(e)}, open(fa, "w"))

    print("RUN 040 COMPLETE")


if __name__ == "__main__":
    main()

