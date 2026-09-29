"""
Run 039: AgentLeak F1.1 (Direct Prompt Injection) reproduction — ASR0
(their real hand-crafted template) vs. ASR1 (ACA v2-optimised suffix),
scored by AgentLeak's own vendored `AgentLeakTester(mode=FAST)`, per
experimental_protocol.md §12.3/§14.

Model: Qwen2.5-0.5B-Instruct, steps=256 — matches run_033/034/035/038's
established scope. Tasks: all 50 (agentleak_f1_tasks.AGENTLEAK_F1_TASKS
— AgentLeak's real first-50-of-100 scenario set, vendored verbatim).

Usage:
  CUDA_VISIBLE_DEVICES=0 HF_HOME=.hf_cache \\
    python -u -m masflow.run_039_agentleak_f1 2>&1 | tee experiment_logs/run_039.log

Resumable: any per-(mode,task) result already written without an
"error" key is skipped.
"""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from masflow.config import load_local_model, seed_everything
from masflow.gcg import get_ascii_printable_tokens
from masflow.agentleak_f1_tasks import AGENTLEAK_F1_TASKS
from masflow import suffix_opt_and_eval, baseline_agentleak_f1

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_KEY = "qwen_0.5b"
SEED = 42
STEPS = 256
BEAM_WIDTH = 4

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "run_039")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    seed_everything(SEED)
    total = len(AGENTLEAK_F1_TASKS) * 2

    print("=" * 72)
    print("  RUN 039: AGENTLEAK F1.1 REPRODUCTION — ASR0 vs ASR1 (ACA v2)")
    print(f"  Model: {MODEL_NAME}  Tasks: {len(AGENTLEAK_F1_TASKS)}  Steps: {STEPS}")
    print("=" * 72)

    model, tok = load_local_model(MODEL_NAME)
    allowed_tokens = get_ascii_printable_tokens(tok)

    run_idx = 0
    for task in AGENTLEAK_F1_TASKS:
        run_idx += 1
        fpath0 = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_agentleak_f1_asr0_task{task.task_id}.json")
        if not (os.path.exists(fpath0) and "error" not in json.load(open(fpath0))):
            print(f"[{run_idx:3d}/{total}] ASR0 task {task.task_id} ({task.vertical}/{task.target_field})")
            try:
                res = baseline_agentleak_f1.run_agentleak_f1_baseline(model, tok, task)
                out = {"mode": "asr0", "attack": "agentleak_f1", "model": MODEL_NAME,
                       "task_id": task.task_id, "vertical": task.vertical,
                       "target_field": task.target_field, **res}
                json.dump(out, open(fpath0, "w"), indent=2)
            except Exception as e:
                json.dump({"error": str(e)}, open(fpath0, "w"))
        else:
            print(f"[{run_idx:3d}/{total}] ASR0 task {task.task_id} — SKIP (cached)")

        run_idx += 1
        fpath1 = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_agentleak_f1_asr1_task{task.task_id}.json")
        if os.path.exists(fpath1) and "error" not in json.load(open(fpath1)):
            print(f"[{run_idx:3d}/{total}] ASR1 task {task.task_id} — SKIP (cached)")
            continue

        print(f"[{run_idx:3d}/{total}] ASR1 task {task.task_id} ({task.vertical}/{task.target_field})")
        try:
            a_ids, suffix_pos, target_ids, target_slice, a_system = baseline_agentleak_f1.build_agentleak_f1_input(
                tok, task, suffix_len=40
            )
            dummy_para_ids = tok.encode("ok", add_special_tokens=False)
            res = suffix_opt_and_eval.run_aca_v2(
                model, tok, a_ids, suffix_pos, target_ids, target_slice,
                torch.tensor([], dtype=torch.long), dummy_para_ids, allowed_tokens, task, a_system,
                steps=STEPS, beam_width=BEAM_WIDTH,
                lambda_b=0.0, lambda_ppl=1.0, lambda_ba=0.0,
                pct_a=90, pct_b=0, pct_r=10, loss_mode="ce",
            )
            device = next(model.parameters()).device
            eval_res = baseline_agentleak_f1.evaluate_agentleak_f1_suffix(
                model, tok, task, res["best_ids"][suffix_pos], a_system, device
            )
            out = {"mode": "asr1", "attack": "agentleak_f1", "model": MODEL_NAME,
                   "task_id": task.task_id, "vertical": task.vertical,
                   "target_field": task.target_field,
                   "a_only_loss": res["a_only_attack_loss"], **eval_res}
            json.dump(out, open(fpath1, "w"), indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"error": str(e)}, open(fpath1, "w"))

    print("RUN 039 COMPLETE")


if __name__ == "__main__":
    main()

