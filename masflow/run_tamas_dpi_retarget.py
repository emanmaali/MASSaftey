"""
run_tamas_dpi_retarget.py -- generates ASR0 (TAMAS's real injection
clause, verbatim, expected near-0% here for the same reason CFH's ASR0
was 0% -- it names TAMAS's own target tool, not ours) and ASR1 (freshly
ACA-optimized against OUR 50 tasks) TAMAS-DPI suffixes, then converts
them into the standard {model_key}_mcp_{attack}_task{id}.json format
every topology driver reads -- same purpose as convert_cfh_to_suffix_format.py,
but TAMAS-DPI's ASR1 needs a fresh optimization run first (unlike CFH's,
which was already optimized against our task set from the start).

Usage:
  python3 -u -m masflow.run_tamas_dpi_retarget \\
      --model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct \\
      --out-dir results/run_tamas_dpi_as_suffix/qwen_0.5b \\
      --task-ids 0,1,...,49
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from dataclasses import dataclass

from masflow.config import load_local_model, seed_everything
from masflow.gcg import get_ascii_printable_tokens
from masflow.tasks import TASKS
from masflow.decision_agent_prompt import TOOL_DESC
from masflow import suffix_opt_and_eval, baseline_tamas_dpi as btd
from masflow.tamas_dpi_tasks import TAMAS_DPI_TASKS

SEED = 42
STEPS = 256
BEAM_WIDTH = 4
# a representative real TAMAS injection clause, reused verbatim as ASR0
# across all our tasks -- expected near-0% here for the same reason CFH's
# ASR0 was 0%: it names TAMAS's own target tool, not ours. Documented, not
# silently glossed over.
REAL_INJECTION_SAMPLE = TAMAS_DPI_TASKS[0].real_injection


@dataclass
class RetargetedTamasTask:
    task_id: int
    target_agent: str
    target_tool: str
    benign_query: str
    real_injection: str
    tool_schema_text: str


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="qwen_0.5b")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(SEED)
    task_ids = [int(x) for x in args.task_ids.split(",")]

    print("Loading model:", args.model_name)
    model, tok = load_local_model(args.model_name)
    allowed_tokens = get_ascii_printable_tokens(tok)

    for i, task_id in enumerate(task_ids):
        task = TASKS[task_id]
        rtask = RetargetedTamasTask(
            task_id=task_id, target_agent="Decision Agent",
            target_tool=task.attack_target_tool, benign_query=task.user_task,
            real_injection=REAL_INJECTION_SAMPLE, tool_schema_text=TOOL_DESC,
        )

        # ASR0: real clause, verbatim, no search
        asr0_path = os.path.join(args.out_dir, f"{args.model_key}_mcp_tamasasr0_task{task_id}.json")
        if not os.path.exists(asr0_path):
            json.dump({"task_id": task_id, "suffix": REAL_INJECTION_SAMPLE,
                       "source": "tamas_dpi_asr0_real_clause"}, open(asr0_path, "w"), indent=2)

        # ASR1: fresh ACA v2 optimization against OUR target tool
        asr1_path = os.path.join(args.out_dir, f"{args.model_key}_mcp_tamasasr1_task{task_id}.json")
        if os.path.exists(asr1_path):
            cached = json.load(open(asr1_path))
            if "error" not in cached:
                print(f"[{i+1}/{len(task_ids)}] task {task_id} ASR1 -- SKIP (cached)", flush=True)
                continue

        print(f"[{i+1}/{len(task_ids)}] task {task_id} ASR1: optimizing...", flush=True)
        try:
            a_ids, suffix_pos, target_ids, target_slice, a_system = btd.build_tamas_dpi_input(
                tok, rtask, suffix_len=40
            )
            dummy_para_ids = tok.encode("ok", add_special_tokens=False)
            res = suffix_opt_and_eval.run_aca_v2(
                model, tok, a_ids, suffix_pos, target_ids, target_slice,
                torch.tensor([], dtype=torch.long), dummy_para_ids, allowed_tokens, task, a_system,
                steps=STEPS, beam_width=BEAM_WIDTH,
                lambda_b=0.0, lambda_ppl=1.0, lambda_ba=0.0,
                pct_a=90, pct_b=0, pct_r=10, loss_mode="ce",
            )
            suffix_text = tok.decode(res["best_ids"][suffix_pos], skip_special_tokens=True)
            device = next(model.parameters()).device
            eval_res = btd.evaluate_tamas_dpi_suffix(model, tok, rtask, res["best_ids"][suffix_pos], a_system, device)
            print(f"    hd={eval_res['hd']}  suffix={suffix_text[:60]!r}", flush=True)
            json.dump({"task_id": task_id, "suffix": suffix_text,
                       "source": "tamas_dpi_asr1_retargeted", "hd": eval_res["hd"]},
                      open(asr1_path, "w"), indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"error": str(e), "task_id": task_id}, open(asr1_path, "w"))

    print("\n=== TAMAS-DPI RETARGET COMPLETE ===")


if __name__ == "__main__":
    main()
