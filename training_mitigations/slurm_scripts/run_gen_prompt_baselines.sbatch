#!/bin/bash
# Generate held-out responses for the two literature prompt-only baselines on
# the UNTRAINED base Qwen3-8B, over the SAME 1000-prompt held-out set used to
# evaluate every other method (heldout_<env>_final1000_ids.json):
#   --explicit        Cheng et al. (2026) EXPLICIT accommodation (PREPEND)
#   --sharma-honesty  Sharma et al. honesty prompt (APPEND)
# Generation-only (--no-judge): writes responses_<tag>_shard{0,1}of2.csv under
# <run>/eval for the downstream v5 endorse + ELEPHANT judges. The judged
# `sentence` is the bare post, so endorsement is measured against the same
# scenario as all other conditions. Sampling matches the base/becritical/
# dontbesyco held-out gens exactly (defaults: temp1.0, top-p0.95, 512 tok,
# seed0). 8B in bf16 split across a 2x a4000 pair via --device-map auto.
#
# Array index -> (env, condition, shard): 8 tasks = 2 envs x 2 conds x 2 shards.

#SBATCH --job-name=gen_prompt_baselines
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/gen_pbl_%A_%a.out
#SBATCH --error=training_mitigations/logs/gen_pbl_%A_%a.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME" training_mitigations/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [[ -z "${HF_TOKEN:-}" ]]; then echo "ERROR: HF_TOKEN not set" >&2; exit 1; fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

#                0          1          2        3        4          5          6        7
ENVS=(    pas        pas        pas      pas      oeq        oeq        oeq      oeq    )
CONDS=(   explicit   explicit   sharma   sharma   explicit   explicit   sharma   sharma )
FLAGS=(   --explicit --explicit --sharma-honesty --sharma-honesty --explicit --explicit --sharma-honesty --sharma-honesty )
SHARDS=(  0          1          0        1        0          1          0        1      )

i=$SLURM_ARRAY_TASK_ID
ENV=${ENVS[$i]}; FLAG=${FLAGS[$i]}; SHARD=${SHARDS[$i]}
ENV_UC=$(echo "$ENV" | tr '[:lower:]' '[:upper:]')
RUNDIR=training_mitigations/runs/dpo_lora_${ENV}_qwen3-8b_n1000
IDS=training_mitigations/sycoharm_runs/heldout_${ENV}_final1000_ids.json

echo "[gen-pbl $i] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[gen-pbl $i] env=$ENV_UC cond=${CONDS[$i]} flag=$FLAG shard=$SHARD/2 run=$RUNDIR"

"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$RUNDIR" --env "$ENV_UC" \
    --base $FLAG \
    --test-ids-file "$IDS" --tag-extra final1000 \
    --device-map auto --seed 0 \
    --num-shards 2 --shard-id "$SHARD" --no-judge
echo "[gen-pbl $i] done."
