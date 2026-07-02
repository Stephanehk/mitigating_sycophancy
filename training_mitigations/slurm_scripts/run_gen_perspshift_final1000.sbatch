#!/bin/bash
# Generate held-out responses for the PERSPECTIVE-SHIFT baseline (ELEPHANT,
# Cheng et al. 2025, App G.2: rewrite first->third person, then answer the
# rewrite) on the UNTRAINED base Qwen3-8B, over the SAME 1000-prompt held-out
# set used to evaluate every other method (heldout_<env>_final1000_ids.json).
#
# This REPLACES the old n=500 perspective-shift run: it now covers the full
# final1000 set and the responses are scored downstream by the v5 gpt-5-mini
# endorse judge + the 3 ELEPHANT scorers (run_judge_perspshift_final1000.sbatch),
# i.e. EXACTLY the explicit/sharma prompt-baseline pipeline, so perspective-shift
# is directly comparable to the rest of the n=1000 table.
#
# Generation-only (--no-judge): writes responses_<tag>_shard{i}of{n}.csv under
# <run>/eval. OEQ gets the paper's "Assess this narrative." addendum (long
# narratives keep re-addressing "you" after the rewrite); PAS does not (single
# short sentences). Sampling matches the base/explicit/sharma held-out gens
# exactly (defaults: temp1.0, top-p0.95, 512 tok, seed0); the first->third
# person rewrite is greedy (--rewrite-max-tokens 1536) so it is deterministic.
# 8B in bf16 split across a 2x a4000 pair via --device-map auto.
#
# Tags / shard counts (OEQ narratives are slower -> more shards):
#   OEQ : base-perspshiftassess_final1000_temp1.0_seed0   6 shards (~167 prompts)
#   PAS : base-perspshift_final1000_temp1.0_seed0         2 shards (~500 prompts)
# Array index -> task: 0-5 = OEQ shards 0-5, 6-7 = PAS shards 0-1.

#SBATCH --job-name=gen_perspshift
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next4,next5,next6
#SBATCH --nodes=1
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/gen_perspshift_%A_%a.out
#SBATCH --error=training_mitigations/logs/gen_perspshift_%A_%a.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME" training_mitigations/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [[ -z "${HF_TOKEN:-}" ]]; then echo "ERROR: HF_TOKEN not set" >&2; exit 1; fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OEQ_EV=training_mitigations/eval/eval_endorsement_heldout.py

# Job table: ENV NUM_SHARDS SHARD_ID. OEQ 6 shards + PAS 2 shards = 8 tasks.
JOBS=(
    "OEQ 6 0"
    "OEQ 6 1"
    "OEQ 6 2"
    "OEQ 6 3"
    "OEQ 6 4"
    "OEQ 6 5"
    "PAS 2 0"
    "PAS 2 1"
)
read -r ENV N S <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
ENV_LC=$(echo "$ENV" | tr '[:upper:]' '[:lower:]')
RUNDIR=training_mitigations/runs/dpo_lora_${ENV_LC}_qwen3-8b_n1000
IDS=training_mitigations/sycoharm_runs/heldout_${ENV_LC}_final1000_ids.json

echo "[gen-persp $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[gen-persp $SLURM_ARRAY_TASK_ID] env=$ENV shard=$S/$N run=$RUNDIR ids=$IDS"

# OEQ adds --assess-narrative (paper addendum for long narratives); PAS does not.
ASSESS=""
if [[ "$ENV" == "OEQ" ]]; then ASSESS="--assess-narrative"; fi

"$PY" "$OEQ_EV" \
    --run-dir "$RUNDIR" --env "$ENV" \
    --base --perspective-shift $ASSESS \
    --test-ids-file "$IDS" --tag-extra final1000 \
    --device-map auto --seed 0 --rewrite-max-tokens 1536 \
    --num-shards "$N" --shard-id "$S" --no-judge
echo "[gen-persp $SLURM_ARRAY_TASK_ID] done."
