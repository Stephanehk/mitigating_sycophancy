#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_pas_n1000.sbatch
# (from the repo root). Bigger run: iterative DPO+LoRA on the PAS env
# with train-n=1000, test-n=500 (held-out eval). TWO model copies on
# TWO GPUs: trainable policy on cuda:0, frozen sim_base on cuda:1.
# Writes to a NEW output dir so the 300-prompt run is not clobbered.
#
# Estimated wall-clock ~2.5 days at ~0.057 h/prompt (extrapolated from
# the 300-prompt run's 17.2 h). Requires $HF_TOKEN in the submitting
# shell.

#SBATCH --job-name=dpo_pas_n1000
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_pas_n1000_%j.out
#SBATCH --error=training_mitigations/logs/dpo_pas_n1000_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it in your shell before sbatch." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT_DIR=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] HF_HOME=$HF_HOME"
echo "[sbatch] python=$PY"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env PAS \
    --output-dir "$OUT_DIR" \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --num-iterations 3 \
    --train-n 1000 \
    --test-n 500 \
    --n-samples 10 \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-tokens 1024 \
    --sample-max-tokens 1024 \
    --beta 0.1 \
    --learning-rate 5e-5 \
    --num-train-epochs 1 \
    --per-device-batch-size 1 \
    --grad-accum 8 \
    --max-length 3072 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05
