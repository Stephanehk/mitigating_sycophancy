#!/bin/bash
# Sharded held-out eval for the PAS base-method run, as a SLURM array.
# Submit (from repo root, with AZURE_OPENAI_* and HF_TOKEN exported):
#   sbatch training_mitigations/slurm_scripts/run_eval_pas_sharded.sbatch
#
# Two models x NUM_SHARDS shards = 4 independent array tasks, each on its own
# 2x a4000 (the 8B is sharded across the pair; no quantization):
#   task 0,1 -> FINAL adapter of runs/dpo_lora_pas_qwen3-8b_n1000, shards 0,1
#   task 2,3 -> untrained base Qwen3-8B (--base),                  shards 0,1
# Each shard evaluates 250 of the 500 held-out PAS prompts, so all 4 run in
# parallel (GPUs permitting) instead of 1000 generations back-to-back.
#
# After the array finishes, recombine each model's shards with:
#   PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
#   RUN=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000
#   $PY training_mitigations/eval/aggregate_endorsement_shards.py --run-dir $RUN --adapter-tag final --num-shards 2
#   $PY training_mitigations/eval/aggregate_endorsement_shards.py --run-dir $RUN --adapter-tag base  --num-shards 2

#SBATCH --job-name=eval_pas_shard
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=training_mitigations/logs/eval_pas_shard_%A_%a.out
#SBATCH --error=training_mitigations/logs/eval_pas_shard_%A_%a.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2; exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
if [[ -z "${AZURE_OPENAI_API_KEY:-}" || -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
    echo "ERROR: AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT must be set." >&2; exit 1
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
RUN=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000
NUM_SHARDS=2

mi=$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))   # 0 -> final, 1 -> base
si=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))   # shard id
MODELS=(final base)
MODEL=${MODELS[$mi]}

echo "[task $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[task $SLURM_ARRAY_TASK_ID] model=$MODEL shard=$si/$NUM_SHARDS"

COMMON=(--run-dir "$RUN" --device-map auto --seed 0 --num-shards "$NUM_SHARDS" --shard-id "$si")
if [[ "$MODEL" == "base" ]]; then
    "$PY" training_mitigations/eval/eval_endorsement_heldout.py "${COMMON[@]}" --base
else
    "$PY" training_mitigations/eval/eval_endorsement_heldout.py "${COMMON[@]}"
fi
echo "[task $SLURM_ARRAY_TASK_ID] done."
