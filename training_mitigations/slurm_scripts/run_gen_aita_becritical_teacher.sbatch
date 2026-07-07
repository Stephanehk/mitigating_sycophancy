#!/bin/bash
# Teacher-generation pass for the "Be critical." SFT baseline on AITA / AITA-NTA-
# FLIP: sample ONE response per TRAIN prompt from the UNTRAINED base Qwen3-8B with
# "\n\nBe critical." appended (eval_aita_label_heldout.py --base --be-critical
# --teacher-out). The teacher CSV records `sentence` = the BARE student surface
# (the post, NO steer, no verdict nudge) and `response` = the critical target, so
# the SFT step (run_train_aita_becritical_sft) distills bare-prompt -> critical.
# AITA-NTA-FLIP emits TWO teacher rows per pair (original + flipped).
#
# Runs on next7 a6000 (one 48 GB card fits the 8B bf16; --device-map auto pins it
# to the single visible GPU). Held-out TRAIN ids come from each SFT run dir's
# train_ids.json (disjoint from the canonical test_ids.json reused for eval).
#
# 4 array tasks = AITA 2 shards + FLIP 2 shards. After this, run_train_*_sft reads
# <sft_dir>/eval/responses_base-becritical_train1000_*shard*.csv (its default glob).

#SBATCH --job-name=gen_aita_sft_teacher
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --array=0-3%3
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=training_mitigations/logs/gen2/%x_%A_%a.out
#SBATCH --error=training_mitigations/logs/gen2/%x_%A_%a.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME" training_mitigations/logs/gen2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [[ -z "${HF_TOKEN:-}" ]]; then echo "ERROR: HF_TOKEN not set" >&2; exit 1; fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
EV=training_mitigations/eval/eval_aita_label_heldout.py
AITA_DIR=training_mitigations/runs/sft_lora_aita_qwen3-8b_n1000_becritical
FLIP_DIR=training_mitigations/runs/sft_lora_flip_qwen3-8b_n500_becritical

# Job table: ENV  SFT_DIR  NUM_SHARDS  SHARD_ID.
JOBS=(
    "AITA          $AITA_DIR 2 0"
    "AITA          $AITA_DIR 2 1"
    "AITA-NTA-FLIP $FLIP_DIR 2 0"
    "AITA-NTA-FLIP $FLIP_DIR 2 1"
)
read -r ENV DIR N S <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

echo "[teacher $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[teacher $SLURM_ARRAY_TASK_ID] env=$ENV shard=$S/$N dir=$DIR"

"$PY" "$EV" --run-dir "$DIR" --env "$ENV" \
    --base --be-critical --teacher-out \
    --test-ids-file "$DIR/train_ids.json" --tag-extra train1000 \
    --device-map auto --seed 0 \
    --sample-max-tokens 1024 --temperature 1.0 --top-p 0.95 \
    --num-shards "$N" --shard-id "$S"
echo "[teacher $SLURM_ARRAY_TASK_ID] done."
