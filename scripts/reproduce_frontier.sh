#!/usr/bin/env bash
# Frontier-model experiments (tab:frontier-toolsel, tab:frontier-estimator,
# tab:frontier-crossfam, tab:adaptive). Calls paid APIs with YOUR keys from .env:
# OPENAI_API_KEY and ANTHROPIC_API_KEY.
#
# Approximate API calls per frontier model:
#   toolsel (also used by estimator)  ~9,900
#   crossfam                          ~13,600
#   adaptive (Claude Haiku only)      ~232 target calls, plus a local mutator model on GPU
#
# Model behaviour may have changed since the paper's runs (Sept 2026), and
# temperature 0 is not deterministic on these APIs, so expect close but not
# identical numbers.
#
#   PART=toolsel|crossfam|adaptive   (required)
#   MODELS="gpt-4o-mini claude-haiku-4-5-20251001"   (default: both)
#   CONFIRM=1                        (required, acknowledges API cost)
source "$(dirname "$0")/_common.sh"
: "${PART:?set PART=toolsel, crossfam or adaptive}"
[ "${CONFIRM:-0}" = 1 ] || { echo "This calls paid APIs. Re-run with CONFIRM=1."; exit 1; }
[ -f .env ] && { set -a; source .env; set +a; }
MODELS="${MODELS:-gpt-4o-mini claude-haiku-4-5-20251001}"
TOPOS="centralized orchestrate tree mesh star centralized_d2 orchestrate_d3 mesh_d4 deep_tree centralized_plus2 orchestrate_plus2 tree_plus2 mesh_plus2"
tag() { case "$1" in claude-haiku*) echo claude-haiku ;; *) echo "$1" ;; esac; }
case "$PART" in
toolsel)
  for M in $MODELS; do for T in $TOPOS; do
    log "toolsel $M / $T"
    $PY -u -m masflow.run_topology_frontier --frontier-model "$M" --topology "$T" \
      --suffix-source-dir results/suffixes/beast_bfcl/qwen_0.5b --source-model-key qwen_0.5b \
      --task-ids "${TASKS:-all}" --results-dir "$OUT/results/run_topology_frontier/$(tag "$M")_from_qwen_0.5b/$T"
  done; done ;;
crossfam)
  for M in $MODELS; do for A in masleakasr1 tomaasr1 infectionasr1 flowsteerasr1 agentleakf1asr1 egasr1; do
    for T in $TOPOS; do
      log "crossfam ASR1 $M / $A / $T"
      $PY -u -m masflow.run_topology_frontier_generic --frontier-model "$M" --attack "$A" --topology "$T" \
        --suffix-source-dir results/suffixes/attack_families/qwen_0.5b --source-model-key qwen_0.5b \
        --task-ids "${TASKS:-all}" --results-dir "$OUT/results/run_topology_frontier_generic/$(tag "$M")_from_qwen_0.5b/$A/$T"
    done
    A0="${A%asr1}asr0"
    log "crossfam ASR0 $M / $A0 (single hop, centralized)"
    $PY -u -m masflow.run_topology_frontier_generic --frontier-model "$M" --attack "$A0" --topology centralized \
      --suffix-source-dir results/suffixes/attack_families/qwen_0.5b --source-model-key qwen_0.5b \
      --task-ids "${TASKS:-all}" --results-dir "$OUT/results/run_asr0_doublecheck/$(tag "$M")/$A0/centralized"
  done; done ;;
adaptive)
  log "adaptive attack vs Claude Haiku (mutator: Phi-3.5-mini on local GPU)"
  $PY -u -m masflow.run_adaptive_frontier_attack --frontier-model claude-haiku-4-5-20251001 \
    --mutator-model-name microsoft/Phi-3.5-mini-instruct --rounds 8 \
    --task-ids "${TASKS:-$($PY -c "print(','.join(map(str,range(30))))")}" \
    --results-dir "$OUT/results/run_adaptive_frontier/claude-haiku" ;;
*) echo "unknown PART=$PART"; exit 1 ;;
esac
log "Done. Compare with: python3 analysis/compare_replay.py --reproduced $OUT"
