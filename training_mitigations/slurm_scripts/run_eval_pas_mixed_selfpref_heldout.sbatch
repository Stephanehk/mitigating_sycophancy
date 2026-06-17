#!/bin/bash
# Sharded held-out generation for the two newly finished PAS DPO runs
# (n1000_mixed and n1000_selfpref), using each run's FINAL saved adapter.
# GENERATION ONLY (--no-judge): no Azure endorse judge -- the sycoharm judge
# is the only judge in this pipeline. Submit (from repo root, HF_TOKEN exported):
#   sbatch training_mitigations/slurm_scripts/run_eval_pas_mixed_selfpref_heldout.sbatch
#
# 2 runs x NUM_SHARDS shards = 4 independent array tasks, each on its own
# 2x a4000 (the 8B is sharded across the pair via --device-map auto; no quant):
#   task 0,1 -> FINAL adapter of runs/dpo_lora_pas_qwen3-8b_n1000_mixed,    shards 0,1
#   task 2,3 -> FINAL adapter of runs/dpo_lora_pas_qwen3-8b_n1000_selfpref, shards 0,1
# Each shard samples 250 of the 500 held-out PAS prompts (temp1.0/seed0,
# matching the existing held-out evals). The sycoharm judge runs afterwards as
# a dependent job (run_sycoharm_pas_mixed_selfpref.sbatch) over the responses.
#
# Outputs per run under <run>/eval/:
#   responses_final_temp1.0_seed0_shard{0,1}of2.csv   (id, sentence, response)
#   responses_final_temp1.0_seed0_shard{0,1}of2.jsonl
#   summary_final_temp1.0_seed0_shard{0,1}of2.json    (judged=false)

#SBATCH --job-name=eval_pas_mix_self
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=training_mitigations/logs/eval_pas_mix_self_%A_%a.out
#SBATCH --error=training_mitigations/logs/eval_pas_mix_self_%A_%a.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2; exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# No Azure creds needed: generation runs with --no-judge.

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
RUNS=(training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_mixed \
      training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_selfpref)
NUM_SHARDS=2

ri=$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))   # 0 -> mixed, 1 -> selfpref
si=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))   # shard id
RUN=${RUNS[$ri]}

echo "[task $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[task $SLURM_ARRAY_TASK_ID] run=$RUN shard=$si/$NUM_SHARDS (FINAL adapter)"

# No --adapter / --base => evaluates the FINAL saved adapter (default = run dir).
# --no-judge => generation only, no Azure endorse judge.
"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$RUN" --device-map auto --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$si" --no-judge
echo "[task $SLURM_ARRAY_TASK_ID] done."
