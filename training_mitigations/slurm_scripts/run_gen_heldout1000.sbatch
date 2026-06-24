#!/bin/bash
# Generation-only (--no-judge) over a 1000-prompt held-out set for ONE trained
# run, using its FINAL saved adapter. Sharded x2 (array tasks 0,1), each shard
# samples 500 of the 1000 held-out prompts (temp1.0/seed0). The 8B is split
# across a 2x a4000 pair via --device-map auto (no quant).
#
# Parametrized via --export (set by run_gen_heldout1000_submit.sh):
#   GEN_RUNDIR : run dir holding the final adapter (and test_ids.json)
#   GEN_IDS    : explicit 1000-id held-out json (--test-ids-file)
#   GEN_ENV    : OEQ | PAS (passed through; also inferable from run name)
#
# Outputs per run under <run>/eval/:
#   responses_final_final1000_temp1.0_seed0_shard{0,1}of2.csv  (id,sentence,response)
#   responses_final_final1000_temp1.0_seed0_shard{0,1}of2.jsonl
#   summary_final_final1000_temp1.0_seed0_shard{0,1}of2.json   (judged=false)

#SBATCH --job-name=gen_hd1000
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/gen_hd1000_%A_%a.out
#SBATCH --error=training_mitigations/logs/gen_hd1000_%A_%a.err

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
NUM_SHARDS=2
si=$SLURM_ARRAY_TASK_ID

echo "[task $si] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[task $si] run=$GEN_RUNDIR env=${GEN_ENV:-auto} ids=$GEN_IDS shard=$si/$NUM_SHARDS (FINAL adapter)"

# No --adapter / --base => evaluates the FINAL saved adapter (= run dir).
# --tag-extra final1000 matches the existing OEQ final1000 response filenames.
"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$GEN_RUNDIR" --env "${GEN_ENV}" \
    --test-ids-file "$GEN_IDS" --tag-extra final1000 \
    --device-map auto --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$si" --no-judge
echo "[task $si] done."
