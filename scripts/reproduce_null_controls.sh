#!/usr/bin/env bash
# Null-suffix controls quoted in the real-BFCL discussion: the same replay
# drivers with an EMPTY adversarial suffix (seed files whose "suffix" is ""),
# so any "hit" is the model's own confusion rather than the attack.
#   original BFCL target, qwen-0.5b (N=100) and qwen-1.5b (plain chain, N=30)
#   retargeted BFCL (fixed out-of-domain target), qwen-0.5b (N=100)
#   toy task set, qwen-0.5b (plain chain, N=30)
# One GPU; well under 1 GPU-hour in total.
source "$(dirname "$0")/_common.sh"
T30="${TASKS:-$($PY -c "print(','.join(map(str,range(30))))")}"
run3() {   # model-key model-name seeds task-source out-dir tasks
  local mk=$1 mn=$2 seeds=$3 ts=$4 out=$5 tids=$6
  local M=(--model-key "$mk" --model-name "$mn" --attack beast --suffix-source-dir "$seeds" --task-source "$ts" --task-ids "$tids")
  log "$out: plain_chain ($mk, $ts)"
  $PY -u -m masflow.run_graph_topology "${M[@]}" --graph plain_chain --results-dir "$OUT/results/$out/$mk"
  for V in mesh_d3_orig star_plus2; do
    log "$out: $V ($mk, $ts)"
    $PY -u -m masflow.run_topology_depth_variants "${M[@]}" --variant "$V" --results-dir "$OUT/results/$out/$mk"
  done
}
run3 qwen_0.5b Qwen/Qwen2.5-0.5B-Instruct results/suffixes/null_bfcl/qwen_0.5b bfcl          null_suffix_check          "$T100"
run3 qwen_0.5b Qwen/Qwen2.5-0.5B-Instruct results/suffixes/null_bfcl/qwen_0.5b bfcl_injected null_suffix_check_injected "$T100"
log "null_suffix_check: plain_chain (qwen_1.5b, bfcl)"
$PY -u -m masflow.run_graph_topology --model-key qwen_1.5b --model-name Qwen/Qwen2.5-1.5B-Instruct --attack beast \
  --suffix-source-dir results/suffixes/null_bfcl/qwen_1.5b --task-source bfcl --task-ids "$T30" --graph plain_chain \
  --results-dir "$OUT/results/null_suffix_check/qwen_1.5b"
log "null_suffix_check_toy: plain_chain (qwen_0.5b, toy)"
$PY -u -m masflow.run_graph_topology --model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct --attack beast \
  --suffix-source-dir results/suffixes/null_toy/qwen_0.5b --task-source toy --task-ids "$T30" --graph plain_chain \
  --results-dir "$OUT/results/null_suffix_check_toy/qwen_0.5b"
log "Done. Compare with: python3 analysis/compare_replay.py --reproduced $OUT"
