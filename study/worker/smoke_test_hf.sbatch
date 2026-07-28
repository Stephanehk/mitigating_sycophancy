#!/bin/bash
#SBATCH --job-name=syco_smoke_hf
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a4000:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/next/u/stephhk/sycophancy/study/logs/smokehf_%j.out
#SBATCH --error=/next/u/stephhk/sycophancy/study/logs/smokehf_%j.out
#
# GPU smoke test using the HF backend on 2x a4000 (next5/next6).
#
# This mirrors the configuration the repo already uses for 8B generation on
# these nodes (see run_gen_heldout1000_v2.sbatch: --gres=gpu:a4000:2 +
# eval_endorsement_heldout.py --device-map auto). accelerate splits the model
# by layer and dispatches sequentially, so it issues no NCCL collectives and
# is unaffected by the broken a4000 interconnect that kills vLLM's TP path.

set -euo pipefail
REPO=/next/u/stephhk/sycophancy
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
mkdir -p "$REPO/study/logs"

export HF_HOME=/next/u/stephhk/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

exec "$PY" "$REPO/study/worker/smoke_test.py" --backend hf --device-map auto
