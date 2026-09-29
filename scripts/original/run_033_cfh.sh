#!/usr/bin/env bash
# Run 033 driver — CFH reproduction (ASR0 vs ASR1). First of the
# checkpoint-gated 033->034->035 chain (experimental_protocol.md §11).
# Chains to run_034 (MASLEAK) on completion; the chain stops after
# run_035 (Prompt Infection) — no further auto-continue into run_030/etc.

set -u
cd "$(dirname "$0")/.."

# Activate the project venv so `python3` resolves to the environment
# with torch/transformers/etc. installed, not the system Python.
source "${VENV}/bin/activate"

RESULTS_DIR="results/run_033"
STATUS_LOG="${RESULTS_DIR}/STATUS.log"
mkdir -p "${RESULTS_DIR}" experiment_logs

log_status() {
    echo "[$(date -Iseconds)] $1" | tee -a "${STATUS_LOG}"
}

log_status "=== Run 033 (CFH) driver started (PID $$) ==="

python3 -u -m infix_gcg.run_033_cfh >> experiment_logs/run_033.log 2>&1
exit_code=$?

if [ $exit_code -eq 0 ]; then
    n_results=$(find "${RESULTS_DIR}" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
    log_status "DONE run_033 (exit=0, ${n_results} result files)"
else
    log_status "FAIL run_033 (exit=${exit_code}) — see experiment_logs/run_033.log for traceback"
fi

log_status "Chaining to run_034 (MASLEAK)..."
setsid nohup bash "$(dirname "$0")/run_034_masleak.sh" < /dev/null > /dev/null 2>&1 &
disown
log_status "run_034 launched (PID $!) — see results/run_034/STATUS.log"
log_status "=== Run 033 driver finished ==="
