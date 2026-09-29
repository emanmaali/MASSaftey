"""
Run 043: TOMA / "Tipping the Dominos" reproduction -- ASR0 (hand-crafted
hierarchical payload) vs ASR1 (ACA v2, multi-hop loss), per
experimental_protocol.md §19.2. Part 2 -- the second multi-hop
regression data point alongside Prompt Infection.

Model: Qwen2.5-0.5B-Instruct, steps=256, MAX_TASKS (default 10).
Structurally mirrors run_035 (Prompt Infection): a linear trust chain,
hop-0 injection, multi-hop `ba_score_fn`.

Usage:
  MAX_TASKS=10 CUDA_VISIBLE_DEVICES=0 HF_HOME=.hf_cache \\
    python -u -m masflow.run_043_toma 2>&1 | tee experiment_logs/run_043.log
"""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from masflow.config import load_local_model, seed_everything
from masflow.gcg import get_ascii_printable_tokens
from masflow.toma_tasks import TOMA_TASKS, ROLE_SYSTEM_PROMPTS
from masflow import suffix_opt_and_eval, baseline_toma as bt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_KEY = "qwen_0.5b"
SEED = 42
STEPS = 256
BEAM_WIDTH = 4
BA_EVERY = 8
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "run_043")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    seed_everything(SEED)
    tasks = TOMA_TASKS[:MAX_TASKS] if MAX_TASKS else TOMA_TASKS
    total = len(tasks) * 2

    print("=" * 72)
    print("  RUN 043: TOMA REPRODUCTION — ASR0 (hand-crafted) vs ASR1 (multi-hop ACA)")
    print(f"  Model: {MODEL_NAME}  Tasks: {len(tasks)}  Steps: {STEPS}  BA_EVERY: {BA_EVERY}")
    print("=" * 72)

    model, tok = load_local_model(MODEL_NAME)
    allowed_tokens = get_ascii_printable_tokens(tok)
    device = next(model.parameters()).device
    a_system = ROLE_SYSTEM_PROMPTS["web_reader"]

    run_idx = 0
    for task in tasks:
        run_idx += 1
        fpath0 = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_toma_asr0_task{task.task_id}.json")
        if not (os.path.exists(fpath0) and "error" not in json.load(open(fpath0))):
            print(f"[{run_idx:3d}/{total}] ASR0 task {task.task_id} ({task.domain})")
            try:
                res = bt.run_toma_baseline(model, tok, task, device)
                json.dump({"mode": "asr0", "attack": "toma", "model": MODEL_NAME,
                           "task_id": task.task_id, "domain": task.domain, **res},
                          open(fpath0, "w"), indent=2)
            except Exception as e:
                json.dump({"error": str(e)}, open(fpath0, "w"))
        else:
            print(f"[{run_idx:3d}/{total}] ASR0 task {task.task_id} — SKIP (cached)")

        run_idx += 1
        fpath1 = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_toma_asr1_task{task.task_id}.json")
        if os.path.exists(fpath1) and "error" not in json.load(open(fpath1)):
            print(f"[{run_idx:3d}/{total}] ASR1 task {task.task_id} — SKIP (cached)")
            continue

        print(f"[{run_idx:3d}/{total}] ASR1 task {task.task_id} ({task.domain})")
        try:
            a_ids, suffix_pos, target_ids, target_slice = bt.build_toma_input(
                tok, task, suffix_len=40
            )
            dummy_para_ids = tok.encode("ok", add_special_tokens=False)
            res = suffix_opt_and_eval.run_aca_v2(
                model, tok, a_ids, suffix_pos, target_ids, target_slice,
                torch.tensor([], dtype=torch.long), dummy_para_ids, allowed_tokens, task, a_system,
                steps=STEPS, beam_width=BEAM_WIDTH,
                lambda_b=0.0, lambda_ppl=1.0, lambda_ba=1.0,
                pct_a=90, pct_b=0, pct_r=10, loss_mode="ce",
                ba_every=BA_EVERY, ba_score_fn=bt.compute_toma_loss,
            )
            eval_res = bt.evaluate_toma_suffix(model, tok, task, res["best_ids"][suffix_pos], device)
            json.dump({"mode": "asr1", "attack": "toma", "model": MODEL_NAME,
                       "task_id": task.task_id, "domain": task.domain,
                       "a_only_loss": res["a_only_attack_loss"], **eval_res},
                      open(fpath1, "w"), indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"error": str(e)}, open(fpath1, "w"))

    print("RUN 043 COMPLETE")


if __name__ == "__main__":
    main()

