#!/usr/bin/env bash
# End-to-end check of the real attack pipeline on a small slice of the paper:
# replays the paper's frozen BEAST suffixes for 3 tasks through the Tree
# topology with Qwen2.5-0.5B-Instruct, then compares every attack outcome
# with the shipped results.
#   GPU: about 1 minute.  CPU only: about 10-15 minutes.
source "$(dirname "$0")/_common.sh"
TASKS="${TASKS:-0,1,2}"
log "Replaying tasks $TASKS through tree_d2_orig (Qwen2.5-0.5B-Instruct)"
$PY -u -m masflow.run_topology_depth_variants \
  --model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --suffix-source-dir results/suffixes/beast_toy/qwen_0.5b \
  --results-dir "$OUT/results/run_topology_depth_variants/qwen_0.5b" \
  --variant tree_d2_orig --attack beast --task-ids "$TASKS"
log "Comparing with shipped results"
$PY analysis/compare_replay.py --reproduced "$OUT" --show-diffs
