#!/usr/bin/env bash
# Stage 2 (replay) for tab:topo-asr, tab:interim-full, tab:rq2-generalise and
# tab:protocol-gen: replays the shipped frozen BEAST suffixes through all 12
# topologies. One GPU. The original runs recorded about 2-3 s/task per
# topology (fan-merge 7-14 s), about 1 GPU-hour per qwen model in total.
#   MODEL=qwen_0.5b (default) | qwen_1.5b | phi35_mini
source "$(dirname "$0")/_common.sh"
MODEL="${MODEL:-qwen_0.5b}"
case "$MODEL" in
  qwen_0.5b)  MN=Qwen/Qwen2.5-0.5B-Instruct; TIDS=(--task-ids "$T50") ;;
  qwen_1.5b)  MN=Qwen/Qwen2.5-1.5B-Instruct; TIDS=(--task-ids "$T50") ;;
  phi35_mini) MN=microsoft/Phi-3.5-mini-instruct; TIDS=(--task-ids "${TASKS:-0,1,2,3,4,5,6,7,8,9}") ;;   # paper used tasks 0-9
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac
SRC=results/suffixes/beast_toy/$MODEL
M=(--model-key "$MODEL" --model-name "$MN" --suffix-source-dir "$SRC" --attack beast)
VARIANTS="centralized_d1_orig orchestrate_d2_orig tree_d2_orig mesh_d3_orig
          centralized_plus2 orchestrate_plus2 tree_plus2 mesh_plus2 star_plus2"
# phi35_mini appears only in tab:topo-asr (the four base shapes plus Star)
[ "$MODEL" = phi35_mini ] && VARIANTS="centralized_d1_orig orchestrate_d2_orig tree_d2_orig mesh_d3_orig"
for V in $VARIANTS; do
  log "$MODEL depth variant $V"
  $PY -u -m masflow.run_topology_depth_variants "${M[@]}" --variant "$V" "${TIDS[@]}" \
    --results-dir "$OUT/results/run_topology_depth_variants/$MODEL"
done
log "$MODEL star (fan-merge)"
$PY -u -m masflow.run_topology_fanmerge "${M[@]}" "${TIDS[@]}" \
  --results-dir "$OUT/results/run_topology_fanmerge/$MODEL"
[ "$MODEL" = phi35_mini ] || for G in plain_chain plain_diamond; do
  log "$MODEL graph $G"
  $PY -u -m masflow.run_graph_topology "${M[@]}" --graph "$G" "${TIDS[@]}" \
    --results-dir "$OUT/results/run_graph_topology/$MODEL"
done
if [ "$MODEL" = qwen_0.5b ]; then        # tab:protocol-gen
  for P in a2a raw; do for V in orchestrate_d2_orig tree_plus2; do
    log "protocol $P, $V"
    $PY -u -m masflow.run_topology_depth_variants "${M[@]}" --variant "$V" --protocol "$P" \
      "${TIDS[@]}" --results-dir "$OUT/results/run_topology_depth_variants/$MODEL"
  done; done
fi
log "Done. Compare with: python3 analysis/compare_replay.py --reproduced $OUT"
