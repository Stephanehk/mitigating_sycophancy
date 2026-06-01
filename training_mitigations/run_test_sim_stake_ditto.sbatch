#!/bin/bash
# Submit with:
#   sbatch training_mitigations/run_test_sim_stake_ditto.sbatch
# (run from the repo root). Targets next7 (8x a6000-48G); we ask for
# 2 GPUs so the primary (Qwen3-8B) and roleplay (Ditto-8B) models each
# get their own card. No overlap between the two — this just removes
# memory pressure on a single 48 GB card.
#
# Requires $HF_TOKEN in the submitting shell for any gated downloads.
# Qwen3-8B and sunweiwei/Ditto-8B are both currently ungated, but
# keeping the token export matches the rest of the sbatch wrappers.

#SBATCH --job-name=simstake_ditto
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=training_mitigations/logs/simstake_ditto_%j.out
#SBATCH --error=training_mitigations/logs/simstake_ditto_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] HF_HOME=$HF_HOME"
echo "[sbatch] python=$PY"

# Per-submission overrides via --export=ALL,VAR=value. Defaults:
# 2 prompts, seed 0, qqqq config. Per-prompt candidate-response count
# is fixed at 6 inside the script (2 × 3 prompt-suffix buckets).
CONFIG="${CONFIG:-qqqq}"
N="${N:-2}"
SEED="${SEED:-0}"
echo "[sbatch] config=$CONFIG n=$N seed=$SEED"

"$PY" training_mitigations/test_sim_stake_ditto.py \
    --n "$N" \
    --seed "$SEED" \
    --config "$CONFIG"
