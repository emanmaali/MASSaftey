#!/usr/bin/env bash
# ACA vs BEAST/G-BEAST family-size sweep driver — one model per invocation.
# Question this answers: at run_030's "base" ACA config (suffix=40, CE loss,
# lambda_B=1.0, lambda_ppl=1.0, lambda_BA=0.5, pool 80/10/10), MCP protocol
# only, 10 tasks, steps=256 — at what model size (within the Qwen family)
# does ACA v2 start beating BEAST and G-BEAST on B-A HD?
#
# Reuses run_029_full_benchmark.py unchanged (all 4 attacks: GCG/BEAST/
# G-BEAST/ACA) — every one of "base"'s hyperparameters is already that
# module's own default (SUFFIX_LEN=40, LAMBDA_PARA=1.0, LAMBDA_PPL=1.0,
# STEPS_DEFAULT=256, default aca-lambda-ba=0.5, default pool 80/10/10); the
# --aca-pool-* flags below are passed explicitly anyway, for a reproducible
# record of exactly what was run rather than relying on silent defaults.
#
# Model + results dir are parameterized via env vars so this one script
# serves the whole Qwen-family sweep (0.5B -> Qwen3-0.6B -> 1.5B -> 3B)
# without duplicating a file per size.
#
# Usage:
#   MODEL_KEY=qwen_0.5b MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
#     bash scripts/run_aca_family_sweep.sh

set -u
cd "$(dirname "$0")/.."

# Load secrets (e.g. HF_TOKEN, for gated models like google/gemma-2-2b-it)
# from .env if present. Safe to omit -- variables simply stay unset if
# .env doesn't exist or doesn't define them.
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Activate the project venv so `python3` resolves to the environment
# with torch/transformers/etc. installed, not the system Python.
source "${VENV}/bin/activate"

MODEL_KEY="${MODEL_KEY:?set MODEL_KEY, e.g. qwen_0.5b}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME, e.g. Qwen/Qwen2.5-0.5B-Instruct}"
RESULTS_DIR="${RESULTS_DIR:-results/run_aca_family_sweep/${MODEL_KEY}}"
LOG_FILE="${LOG_FILE:-experiment_logs/run_aca_family_sweep_${MODEL_KEY}.log}"
STATUS_LOG="${RESULTS_DIR}/STATUS.log"
mkdir -p "${RESULTS_DIR}" experiment_logs

log_status() {
    echo "[$(date -Iseconds)] $1" | tee -a "${STATUS_LOG}"
}

log_status "=== ACA family sweep: ${MODEL_KEY} (${MODEL_NAME}) driver started (PID $$) ==="

python3 -u -m infix_gcg.run_029_full_benchmark \
    --model-key "${MODEL_KEY}" --model-name "${MODEL_NAME}" \
    --tasks 0,1,2,3,4,5,6,7,8,9 --protocols mcp --attacks all --steps 256 \
    --aca-lambda-ba 0.5 --aca-pool-a 80 --aca-pool-b 10 --aca-pool-r 10 \
    --results-dir "${RESULTS_DIR}" \
    >> "${LOG_FILE}" 2>&1
exit_code=$?

if [ $exit_code -eq 0 ]; then
    n_results=$(find "${RESULTS_DIR}" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
    log_status "DONE ${MODEL_KEY} (exit=0, ${n_results}/40 result files) — see ${LOG_FILE}"
else
    log_status "FAIL ${MODEL_KEY} (exit=${exit_code}) — see ${LOG_FILE} for traceback"
fi

log_status "=== ACA family sweep: ${MODEL_KEY} driver finished — checkpoint, no auto-chain ==="
exit $exit_code
