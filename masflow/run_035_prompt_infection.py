"""
Run 035: Prompt Infection reproduction — ASR0 (reconstructed hand-crafted
self-replication template) vs. ASR1 (ACA v2, multi-hop loss), per
experimental_protocol.md §11.3.

Model: Qwen2.5-0.5B-Instruct (rescoped 2026-08-21 from smol_135m, same
rationale as run_033/CFH — matches run_022/run_037's known-good scope
rather than the cheaper scope that risks a weak/inconclusive signal).
Tasks: prompt_infection_tasks.INFECTION_TASKS (10 document-reader chain
instances). Steps: 256 (was 96) — each candidate evaluation here replays
a full 5-hop generate() chain, much more expensive per step than
CFH/MASLEAK; see run comment on BA_EVERY below.

Usage:
  CUDA_VISIBLE_DEVICES=0 HF_HOME=.hf_cache \\
    python -u -m masflow.run_035_prompt_infection 2>&1 | tee experiment_logs/run_035.log
"""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from masflow.config import load_local_model, seed_everything
from masflow.gcg import get_ascii_printable_tokens
from masflow.prompt_infection_tasks import INFECTION_TASKS
from masflow import suffix_opt_and_eval, baseline_prompt_infection as bpi

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_KEY = "qwen_0.5b"
SEED = 42
STEPS = 256
BEAM_WIDTH = 4
# Multi-hop replay is expensive (5 generate() calls per top candidate) —
# score less often than CFH/MASLEAK's default BA_EVERY=4 to keep this
# tractable; still frequent enough to meaningfully guide the beam.
BA_EVERY = 8

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "run_035")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    seed_everything(SEED)
    total = len(INFECTION_TASKS) * 2

    print("=" * 72)
    print("  RUN 035: PROMPT INFECTION REPRODUCTION — ASR0 vs ASR1 (multi-hop ACA)")
    print(f"  Model: {MODEL_NAME}  Tasks: {len(INFECTION_TASKS)}  Steps: {STEPS}  BA_EVERY: {BA_EVERY}")
    print("=" * 72)

    model, tok = load_local_model(MODEL_NAME)
    allowed_tokens = get_ascii_printable_tokens(tok)
    device = next(model.parameters()).device
    a_system = bpi.ROLE_SYSTEM_PROMPTS["reader"]

    run_idx = 0
    for task in INFECTION_TASKS:
        run_idx += 1
        fpath0 = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_infection_asr0_task{task.task_id}.json")
        if not (os.path.exists(fpath0) and "error" not in json.load(open(fpath0))):
            print(f"[{run_idx:3d}/{total}] ASR0 task {task.task_id} ({task.doc_type})")
            try:
                res = bpi.run_infection_baseline(model, tok, task, device)
                json.dump({"mode": "asr0", "attack": "prompt_infection", "model": MODEL_NAME,
                           "task_id": task.task_id, "doc_type": task.doc_type, **res},
                          open(fpath0, "w"), indent=2)
            except Exception as e:
                json.dump({"error": str(e)}, open(fpath0, "w"))
        else:
            print(f"[{run_idx:3d}/{total}] ASR0 task {task.task_id} — SKIP (cached)")

        run_idx += 1
        fpath1 = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_infection_asr1_task{task.task_id}.json")
        if os.path.exists(fpath1) and "error" not in json.load(open(fpath1)):
            print(f"[{run_idx:3d}/{total}] ASR1 task {task.task_id} — SKIP (cached)")
            continue

        print(f"[{run_idx:3d}/{total}] ASR1 task {task.task_id} ({task.doc_type})")
        try:
            a_ids, suffix_pos, target_ids, target_slice = bpi.build_infection_input(
                tok, task, suffix_len=40
            )
            dummy_para_ids = tok.encode("ok", add_special_tokens=False)
            res = suffix_opt_and_eval.run_aca_v2(
                model, tok, a_ids, suffix_pos, target_ids, target_slice,
                torch.tensor([], dtype=torch.long), dummy_para_ids, allowed_tokens, task, a_system,
                steps=STEPS, beam_width=BEAM_WIDTH,
                lambda_b=0.0, lambda_ppl=1.0, lambda_ba=1.0,
                pct_a=90, pct_b=0, pct_r=10, loss_mode="ce",
                ba_every=BA_EVERY, ba_score_fn=bpi.compute_infection_loss,
            )
            eval_res = bpi.evaluate_infection_suffix(model, tok, task, res["best_ids"][suffix_pos], device)
            json.dump({"mode": "asr1", "attack": "prompt_infection", "model": MODEL_NAME,
                       "task_id": task.task_id, "doc_type": task.doc_type,
                       "a_only_loss": res["a_only_attack_loss"], **eval_res},
                      open(fpath1, "w"), indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"error": str(e)}, open(fpath1, "w"))

    print("RUN 035 COMPLETE")


if __name__ == "__main__":
    main()
