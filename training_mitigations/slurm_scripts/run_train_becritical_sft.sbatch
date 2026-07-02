#!/bin/bash
# "Be critical." SFT baseline: SFT base Qwen3-8B (LoRA, bf16) so the bare-prompt
# policy reproduces base+"Be critical." responses. Trains on the teacher CSV
# shards written by the gen pass (run_gen_becritical_teacher.sbatch). The 8B bf16
# weights don't fit one 16 GB a4000, so device_map=auto shards the base across a
# 2x a4000 pair (naive model-parallel; HF Trainer runs MP, not DDP). Same LoRA /
# optimizer config as the DPO mitigations for comparability — only the objective
# (plain NLL on the teacher response) differs.
#
# Parametrized via --export:
#   SFT_OUTDIR : run dir holding eval/ teacher shards; adapter saved here
#   SFT_EXPECT : expected teacher-pair count (the train-id set size, default 1000)

#SBATCH --job-name=sft_becritical
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=training_mitigations/logs/%x_%j.out
#SBATCH --error=training_mitigations/logs/%x_%j.err

set -euo pipefail

cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2; exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

: "${SFT_OUTDIR:?set SFT_OUTDIR}"
SFT_EXPECT="${SFT_EXPECT:-1000}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] output_dir=$SFT_OUTDIR  expect=$SFT_EXPECT"

"$PY" training_mitigations/train_sft_lora.py \
    --output-dir "$SFT_OUTDIR" \
    --expect "$SFT_EXPECT" \
    --device-map auto \
    --seed 0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --learning-rate 5e-5 \
    --per-device-batch-size 1 --grad-accum 8 \
    --num-train-epochs 1 \
    --max-length 3072
