#!/usr/bin/env bash
# Stage 2 (replay) for tab:rq1-toolsel, tab:rq1-other, tab:rq1-bestfit,
# tab:rq1-full, tab:rq2-crossfam and tab:toma-bounds: replays each of the
# eight reproduced attacks' shipped payloads/suffixes through every topology,
# Qwen2.5-0.5B-Instruct, N=50. One GPU; the original runs recorded about 5 GPU-hours.
#   ATTACKS="beast cfhasr1 ..." to run a subset.
source "$(dirname "$0")/_common.sh"
ATTACKS="${ATTACKS:-beast cfhasr1 tamasasr1 masleakasr1 agentleakf1asr1 flowsteerasr1 infectionasr0 tomaasr0}"
for A in $ATTACKS; do
  case "$A" in
    beast)     SRC=results/suffixes/beast_toy/qwen_0.5b ;;
    cfhasr*)   SRC=results/suffixes/cfh/qwen_0.5b ;;
    tamasasr*) SRC=results/suffixes/tamas_dpi/qwen_0.5b ;;
    *)         SRC=results/suffixes/attack_families/qwen_0.5b ;;
  esac
  M=(--model-key qwen_0.5b --model-name Qwen/Qwen2.5-0.5B-Instruct --task-ids "$T50"
     --suffix-source-dir "$SRC" --attack "$A")
  for V in centralized_d1_orig orchestrate_d2_orig tree_d2_orig mesh_d3_orig \
           centralized_plus2 orchestrate_plus2 tree_plus2 mesh_plus2 star_plus2; do
    log "$A / $V"
    $PY -u -m masflow.run_topology_depth_variants "${M[@]}" --variant "$V" \
      --results-dir "$OUT/results/run_topology_depth_variants/qwen_0.5b"
  done
  log "$A / star (fan-merge)"
  $PY -u -m masflow.run_topology_fanmerge "${M[@]}" \
    --results-dir "$OUT/results/run_topology_fanmerge/qwen_0.5b"
  for G in plain_chain plain_diamond; do
    log "$A / $G"
    $PY -u -m masflow.run_graph_topology "${M[@]}" --graph "$G" \
      --results-dir "$OUT/results/run_graph_topology/qwen_0.5b"
  done
done
log "Done. Compare with: python3 analysis/compare_replay.py --reproduced $OUT"
