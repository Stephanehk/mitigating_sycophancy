#!/bin/bash
# Submit with:
#   sbatch training_mitigations/run_test_v2_llama.sbatch
# (run from the repo root). Targets the `next` partition on node next7,
# which has 8x a6000-48G; we ask for 1 GPU since the 8B fits in bf16
# on a single card with ~32 GB to spare.
#
# Requires $HF_TOKEN to be set in the submitting shell so the gated
# Llama 3.1 8B Instruct weights can be downloaded into the cache dir.
# Logs land in training_mitigations/logs/.

#SBATCH --job-name=v2_llama_oeq
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=training_mitigations/logs/v2_llama_oeq_%j.out
#SBATCH --error=training_mitigations/logs/v2_llama_oeq_%j.err

set -euo pipefail

# Cache location chosen because $HOME doesn't have one yet and /next/u
# has ~900 GB free at time of writing. transformers + huggingface_hub
# both honor HF_HOME for everything (datasets, hub, etc.).
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"

# Forwarded from the submitting shell so the gated model download works
# non-interactively. Fail fast if it's not set instead of waiting for
# the HF call to surface a 401.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it in your shell before sbatch." >&2
    exit 1
fi
# `transformers` reads either HF_TOKEN or HUGGING_FACE_HUB_TOKEN. Export
# both so older code paths inside the libs both work.
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] HF_HOME=$HF_HOME"
echo "[sbatch] python=$PY"

"$PY" training_mitigations/test_v2_llama.py \
    --n 5 \
    --seed 0 \
    --max-tokens 2048
