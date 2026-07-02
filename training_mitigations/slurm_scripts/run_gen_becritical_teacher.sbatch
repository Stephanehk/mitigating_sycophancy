#!/bin/bash
# Teacher-generation pass for the "Be critical." SFT baseline: sample ONE
# response per TRAIN prompt from the UNTRAINED base Qwen3-8B with "\n\nBe
# critical." appended at generation time (eval_endorsement_heldout.py --base
# --be-critical --no-judge). The recorded `sentence` is the BARE prompt, so the
# SFT step trains bare-prompt -> critical-response. Sharded across next5/next6
# a4000 pairs (the 8B is split over a 2x a4000 pair via --device-map auto).
#
# Parametrized via --export:
#   GEN_RUNDIR : SFT run dir (holds train_ids.json; eval/ shards written here)
#   GEN_ENV    : OEQ | PAS
#   GEN_IDS    : the 1000 TRAIN ids json (same split as the DPO mitigations)
#   NUM_SHARDS : total shards (this task does $SLURM_ARRAY_TASK_ID)

#SBATCH --job-name=gen_becritical_teacher
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
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

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2; exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

: "${GEN_RUNDIR:?set GEN_RUNDIR}"
: "${GEN_ENV:?set GEN_ENV}"
: "${GEN_IDS:?set GEN_IDS}"
: "${NUM_SHARDS:?set NUM_SHARDS}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
si=$SLURM_ARRAY_TASK_ID

echo "[teacher $si] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[teacher $si] run=$GEN_RUNDIR env=$GEN_ENV ids=$GEN_IDS shard=$si/$NUM_SHARDS"

"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$GEN_RUNDIR" --env "$GEN_ENV" \
    --base --be-critical \
    --test-ids-file "$GEN_IDS" --tag-extra train1000 \
    --device-map auto --seed 0 \
    --sample-max-tokens 1024 --temperature 1.0 --top-p 0.95 \
    --num-shards "$NUM_SHARDS" --shard-id "$si" --no-judge
echo "[teacher $si] done."
