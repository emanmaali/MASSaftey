#!/usr/bin/env bash
# Regenerate every paper table from the shipped results and compare with the
# paper. CPU only, standard library only, takes a few seconds.
source "$(dirname "$0")/_common.sh"
$PY analysis/verify_all.py "$@"
