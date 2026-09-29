#!/usr/bin/env bash
# Stage 2 (replay) for tab:realbfcl-attempt: the three topologies of the
# real-BFCL migration attempt, Qwen2.5-0.5B-Instruct, for each of the three
# columns (toy / real-BFCL original target / real-BFCL fixed target). One GPU.
source "$(dirname "$0")/_common.sh"
M=(--model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct --attack beast)
#        column  suffix source              output dir (graph)       output dir (depth variants)       task source  tasks
run_col() {
  local src=$1 og=$2 od=$3 ts=$4 tids=$5
  log "column: $ts"
  $PY -u -m masflow.run_graph_topology "${M[@]}" --graph plain_chain --task-source "$ts" --task-ids "$tids" \
    --suffix-source-dir "$src/qwen_0.5b" --results-dir "$OUT/results/$og/qwen_0.5b"
  for V in mesh_d3_orig star_plus2; do
    $PY -u -m masflow.run_topology_depth_variants "${M[@]}" --variant "$V" --task-source "$ts" --task-ids "$tids" \
      --suffix-source-dir "$src/qwen_0.5b" --results-dir "$OUT/results/$od/qwen_0.5b"
  done
}
run_col results/suffixes/beast_toy      run_graph_topology        run_topology_depth_variants      toy           "$T50"
run_col results/suffixes/beast_bfcl run_graph_topology_bfcl   run_topology_depth_variants_bfcl bfcl          "$T100"
run_col results/suffixes/beast_bfcl_retargeted   run_beast_injected_pilot  run_beast_injected_pilot         bfcl_injected "$T100"
log "Done. Compare with: python3 analysis/compare_replay.py --reproduced $OUT"
