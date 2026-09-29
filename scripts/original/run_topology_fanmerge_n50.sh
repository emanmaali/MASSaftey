#!/usr/bin/env bash
# Extends the fan-merge topology experiment (Track 1: frozen BEAST replay)
# from N=10 to N=50 on qwen_0.5b. Two stages:
#   1. Generate the missing single-hop BEAST suffixes for tasks 10-49
#      (only 0-9 exist in results/run_aca_family_sweep/qwen_0.5b).
#   2. Run the fan-merge eval (topology_fanmerge_eval) across all 50 tasks
#      (tasks 0-9 are already cached and will be skipped).
set -u
cd "$(dirname "$0")/.."

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

source "${VENV}/bin/activate"
export TMPDIR=/tmp
mkdir -p /tmp

BEAST_DIR="results/run_aca_family_sweep/qwen_0.5b"
FANMERGE_DIR="results/run_topology_fanmerge/qwen_0.5b"
LOG_DIR="experiment_logs"
mkdir -p "${BEAST_DIR}" "${FANMERGE_DIR}" "${LOG_DIR}"

STATUS_LOG="${FANMERGE_DIR}/STATUS_n50.log"
log_status() {
    echo "[$(date -Iseconds)] $1" | tee -a "${STATUS_LOG}"
}

log_status "=== fanmerge N=50 driver started (PID $$) ==="

MISSING_TASKS=$(python3 -c "
import os
missing = [str(i) for i in range(10, 50) if not os.path.exists('${BEAST_DIR}/qwen_0.5b_mcp_beast_task'+str(i)+'.json')]
print(','.join(missing))
")

if [ -n "${MISSING_TASKS}" ]; then
    log_status "--- Stage 1: BEAST suffixes for tasks [${MISSING_TASKS}] ---"
    python3 -u -m infix_gcg.run_029_full_benchmark \
        --model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct \
        --tasks "${MISSING_TASKS}" --protocols mcp --attacks beast --steps 256 \
        --results-dir "${BEAST_DIR}" \
        >> "${LOG_DIR}/fanmerge_n50_stage1_beast.log" 2>&1
    exit1=$?
    n1=$(find "${BEAST_DIR}" -maxdepth 1 -name "qwen_0.5b_mcp_beast_task*.json" 2>/dev/null | wc -l)
    log_status "Stage 1 done (exit=${exit1}, ${n1}/50 BEAST suffix files total)"
else
    log_status "Stage 1: all 50 BEAST suffixes already present, skipping"
fi

log_status "--- Stage 2: fan-merge eval, all 50 tasks ---"
python3 -u -m infix_gcg.run_topology_fanmerge \
    --model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --suffix-source-dir "${BEAST_DIR}" \
    --results-dir "${FANMERGE_DIR}" \
    >> "${LOG_DIR}/fanmerge_n50_stage2_eval.log" 2>&1
exit2=$?
n2=$(find "${FANMERGE_DIR}" -maxdepth 1 -name "qwen_0.5b_fanmerge_task*.json" 2>/dev/null | wc -l)
log_status "Stage 2 done (exit=${exit2}, ${n2}/50 fanmerge result files)"

log_status "=== fanmerge N=50 driver finished ==="
