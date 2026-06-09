#!/bin/bash
# MEMORY DIAGNOSTIC run — submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_oeq_memtest.sbatch
# (from the repo root). Short OEQ DPO run used only to capture the
# host/GPU memory curve via training_mitigations/mem_probe.py while we
# chase the OOM that killed the full runs (jobs 15601660 / 15601662).
#
# Deliberately identical to run_train_oeq.sbatch in every PER-CALL knob
# (n-samples, sample-max-tokens, max-tokens, lora/dpo params) so the
# memory footprint PER PROMPT matches the real run. Only the work VOLUME
# is cut: --train-n 10, --num-iterations 1. Writes to a SEPARATE output
# dir so it does not clobber the real run's split-id files.
#
# Read training_mitigations/runs/dpo_lora_oeq_memtest/mem.csv afterward:
# rss_mb climbing vs flat-high answers host-leak vs under-provisioned.
#
# Requires $HF_TOKEN in the submitting shell. Logs land in
# training_mitigations/logs/.

#SBATCH --job-name=dpo_oeq_memtest
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=training_mitigations/logs/dpo_oeq_memtest_%j.out
#SBATCH --error=training_mitigations/logs/dpo_oeq_memtest_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it in your shell before sbatch." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_memtest

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] HF_HOME=$HF_HOME"
echo "[sbatch] python=$PY"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ \
    --output-dir "$OUT_DIR" \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --num-iterations 1 \
    --train-n 6 \
    --test-n 10 \
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
