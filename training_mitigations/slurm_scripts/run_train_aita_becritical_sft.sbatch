#!/bin/bash
# "Be critical." SFT baseline on AITA / AITA-NTA-FLIP: SFT base Qwen3-8B (LoRA,
# bf16) so the bare-prompt policy reproduces base+"Be critical." YTA/NTA responses
# (context distillation). Reads the teacher (sentence,response) shards written by
# run_gen_aita_becritical_teacher (<sft_dir>/eval/responses_*becritical*shard*.csv,
# the train_sft_lora default glob). Same LoRA/optimizer config as the DPO/OEQ-SFT
# runs for comparability; only the objective (NLL on the teacher response) differs.
#
# Runs on next7 a6000 (the 8B bf16 + LoRA fits one 48 GB card; --device-map auto
# pins to the single visible GPU). Gate afterok on run_gen_aita_becritical_teacher.
# 2 array tasks = AITA + FLIP (each expects 1000 teacher pairs: AITA 1000 posts,
# FLIP 500 pairs x 2).

#SBATCH --job-name=sft_aita_becritical
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --array=0-1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=training_mitigations/logs/%x_%A_%a.out
#SBATCH --error=training_mitigations/logs/%x_%A_%a.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [[ -z "${HF_TOKEN:-}" ]]; then echo "ERROR: HF_TOKEN not set" >&2; exit 1; fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
DIRS=(
    training_mitigations/runs/sft_lora_aita_qwen3-8b_n1000_becritical
    training_mitigations/runs/sft_lora_flip_qwen3-8b_n500_becritical
)
DIR=${DIRS[$SLURM_ARRAY_TASK_ID]}

echo "[sft $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sft $SLURM_ARRAY_TASK_ID] output_dir=$DIR"

"$PY" training_mitigations/train_sft_lora.py \
    --output-dir "$DIR" \
    --expect 1000 \
    --device-map auto \
    --seed 0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --learning-rate 5e-5 \
    --per-device-batch-size 1 --grad-accum 8 \
    --num-train-epochs 1 \
    --max-length 3072
echo "[sft $SLURM_ARRAY_TASK_ID] done."
