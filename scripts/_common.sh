# Shared settings for the reproduction scripts. Sourced, not run directly.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # package root
PY="${PY:-python3}"
# Re-runs go to reproduced/ so the shipped results/ are never overwritten.
# (The drivers skip any output file that already exists.)
OUT="${OUT:-reproduced}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
# TASKS=0,1,2 overrides the task list in every script (quick partial runs).
T50="${TASKS:-$($PY -c "print(','.join(map(str,range(50))))")}"
T100="${TASKS:-$($PY -c "print(','.join(map(str,range(100))))")}"
log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
