#!/usr/bin/env bash
# Stage 1 for the eight reproduced attacks (optional, expensive): re-optimise
# the per-task ASR1 suffixes of each attack family against the 50 toy tasks
# with Qwen2.5-0.5B-Instruct, then convert them into the replay format that
# scripts/reproduce_families.sh and reproduce_frontier.sh (PART=crossfam) read.
# The paper's own suffixes are shipped, so reviewers normally skip this stage.
# ASR0 payloads are the attacks' fixed hand-written strings; they are shipped
# and are not regenerated here. One GPU; several GPU-hours per family.
#   FAMILIES="cfh tamas infection agentleakf1 masleak flowsteer toma eg"  (default: all)
source "$(dirname "$0")/_common.sh"
FAMILIES="${FAMILIES:-cfh tamas infection agentleakf1 masleak flowsteer toma eg}"
DST="$OUT/results/suffixes/attack_families/qwen_0.5b"
mkdir -p "$DST"
# NOTE: the family generators below write their raw outputs to results/run_0NN/
# (a path fixed in each module). Those folders are not part of the shipped
# results, so nothing shipped is overwritten.
for F in $FAMILIES; do
  case "$F" in
    cfh)
      log "CFH: optimise (run_033) and convert"
      $PY -u -m masflow.run_033_cfh --task-source toy
      $PY -u -m masflow.convert_cfh_to_suffix_format --out-dir "$OUT/results/suffixes/cfh/qwen_0.5b" ;;
    tamas)
      log "TAMAS-DPI: optimise against the toy tasks"
      $PY -u -m masflow.run_tamas_dpi_retarget --task-ids "$T50" --out-dir "$OUT/results/suffixes/tamas_dpi/qwen_0.5b" ;;
    infection)   log "Prompt Infection (run_035)"; $PY -u -m masflow.run_035_prompt_infection ;;
    agentleakf1) log "AgentLeak-F1 (run_039)";     $PY -u -m masflow.run_039_agentleak_f1 ;;
    masleak)     log "MASLEAK v2 (run_040)";       MAX_TASKS=50 $PY -u -m masflow.run_040_masleak_v2 ;;
    flowsteer)   log "FlowSteer (run_041)";        MAX_TASKS=50 $PY -u -m masflow.run_041_flowsteer ;;
    toma)        log "TOMA (run_043)";             MAX_TASKS=50 $PY -u -m masflow.run_043_toma ;;
    eg)          log "Evil Geniuses (run_045)";    MAX_TASKS=50 $PY -u -m masflow.run_045_evil_geniuses ;;
    *) echo "unknown family $F"; exit 1 ;;
  esac
done
log "Converting ASR1 outputs into replay format"
$PY scripts/convert_family_suffixes.py --src results --out "$DST"
log "Done. Replay your suffixes by pointing --suffix-source-dir at $OUT/results/..."
