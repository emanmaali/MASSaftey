#!/usr/bin/env bash
# Stage 1 (optional, expensive): re-optimise the BEAST adversarial suffixes
# that every replay script consumes. The paper's suffixes are already shipped
# in results/suffixes/beast_*/, so reviewers normally skip this stage.
# BEAST is not bit-reproducible across GPUs and library builds, so a fresh run
# gives different suffixes and slightly different downstream rates.
# About 2.2 GPU-hours (qwen_0.5b), 5.3 (qwen_1.5b) and 3.9 (phi35_mini, tasks 0-9).
#   MODEL=qwen_0.5b (default) | qwen_1.5b | phi35_mini     TASK_SOURCE=toy (default) | bfcl | bfcl_injected
#   STEPS=256 (paper). Use e.g. STEPS=5 TASKS=0 only to check the pipeline runs.
source "$(dirname "$0")/_common.sh"
MODEL="${MODEL:-qwen_0.5b}"; TS="${TASK_SOURCE:-toy}"
case "$MODEL" in
  qwen_0.5b) MN=Qwen/Qwen2.5-0.5B-Instruct ;; qwen_1.5b) MN=Qwen/Qwen2.5-1.5B-Instruct ;;
  phi35_mini) MN=microsoft/Phi-3.5-mini-instruct ;; *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac
case "$TS" in toy) TIDS=$T50; D=results/suffixes/beast_toy ;; bfcl) TIDS=$T100; D=results/suffixes/beast_bfcl ;;
  bfcl_injected) TIDS=$T100; D=results/suffixes/beast_bfcl_retargeted ;; *) echo "unknown TASK_SOURCE=$TS"; exit 1 ;; esac
[ "$MODEL" = phi35_mini ] && TIDS="${TASKS:-0,1,2,3,4,5,6,7,8,9}"   # paper: tasks 0-9
log "BEAST suffixes: $MODEL, $TS tasks"
$PY -u -m masflow.run_029_full_benchmark --model-key "$MODEL" --model-name "$MN" \
  --tasks "$TIDS" --task-source "$TS" --protocols mcp --attacks beast --steps "${STEPS:-256}" \
  --results-dir "$OUT/$D/$MODEL"
log "Done. To replay with your new suffixes, point --suffix-source-dir at $OUT/$D/$MODEL"
