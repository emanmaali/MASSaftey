#!/usr/bin/env bash
#SBATCH --job-name=qwen15b_beast_ext
#SBATCH --output=experiment_logs/qwen15b_beast_ext_%j.out
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

set -u
export TMPDIR=/tmp
mkdir -p /tmp

cd ${REPO_ROOT}
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi
source "${VENV}/bin/activate"

python3 -u -m infix_gcg.run_029_full_benchmark \
    --model-key qwen_1.5b --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --tasks 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49 \
    --protocols mcp --attacks beast --steps 256 \
    --results-dir results/run_aca_family_sweep/qwen_1.5b
