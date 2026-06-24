#!/bin/bash
# Iterate Stage-1 endorsement-classifier prompts on Qwen3-8B against the v5 gold
# {challenge=0, affirm=1} set. ONE VARIANT PER ARRAY TASK, each on its own 2x
# a4000 pair, spread across next5/next6 -> all variants run at once. Full affirm
# set (--n-affirm 0 => no subsample). Batch job => survives disconnect.
#SBATCH --job-name=iter_stage1
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --array=0-3
#SBATCH --gres=gpu:a4000:2
#SBATCH --nodelist=next5,next6
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=training_mitigations/logs/iter_stage1_%A_%a.out
#SBATCH --error=training_mitigations/logs/iter_stage1_%A_%a.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

VARIANTS=(V0_current V1_challenge_gate V2_reason_first V3_minimal)
VAR="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
echo "[array $SLURM_ARRAY_TASK_ID] variant=$VAR"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
"$PY" training_mitigations/eval/iterate_stage1.py \
    --n-affirm 0 --device-map auto --variants "$VAR"
echo "[iter_stage1 $VAR] done."
