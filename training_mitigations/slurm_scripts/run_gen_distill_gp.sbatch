#!/bin/bash
# Full 1000-prompt held-out generation for the mixed_P5_nocrit_distill FINAL
# (iter_3) adapter WITH the VOICE gen-prefix prepended (the "_gp" arm). The
# judge still scores the bare post, so endorsement is measured against the same
# scenarios as every other arm; only the model's input carries the constraint.
#
# Parametrized via --export:
#   GEN_RUNDIR : final-adapter run dir (training_mitigations/runs/...)
#   GEN_ENV    : OEQ | PAS
#   GEN_IDS    : 1000-id held-out json
#   NUM_SHARDS : shard count (this task = shard $SLURM_ARRAY_TASK_ID)
# GPUs are set via the #SBATCH default (a4000:2) but meant to be overridden per
# env on the sbatch line (OEQ uses :3). Output tag: final-gp_final1000_...
#
# Requires HF_TOKEN (submit with --export=ALL).

#SBATCH --job-name=gen_distill_gp
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
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

: "${GEN_RUNDIR:?set GEN_RUNDIR}"
: "${GEN_ENV:?set GEN_ENV}"
: "${GEN_IDS:?set GEN_IDS}"
: "${NUM_SHARDS:?set NUM_SHARDS}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
si=$SLURM_ARRAY_TASK_ID
VOICE="Address the writer directly as 'you'; never reply in the voice of the other people described."

echo "[task $si] node=$(hostname) env=$GEN_ENV shard=$si/$NUM_SHARDS gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$GEN_RUNDIR" --env "$GEN_ENV" \
    --gen-prefix "$VOICE" --gen-tag gp \
    --test-ids-file "$GEN_IDS" --tag-extra final1000 \
    --device-map auto --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$si" --no-judge
echo "[task $si] done."
