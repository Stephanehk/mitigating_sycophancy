#!/bin/bash
# A/B test: regenerate the OEQ mixed_P5_nocrit_distill iter_2 and iter_3 (final)
# adapters over the SAME 500-prompt held-out subset, but with an explicit
# voice-constraint instruction PREPENDED to each post (--gen-prefix). The judge
# still scores the bare post, so endorsement is measured against the identical
# scenario; only the model's input carries the constraint.
#
# 20-task array, interleaved so both adapters stay balanced in the run queue
# (10 concurrent 2-GPU slots across next5/6 -> finished slots backfill, no node
# idles on a slow tail):
#   even task i -> adapter_iter_2, shard i/2 of 10
#   odd  task i -> final adapter,  shard i/2 of 10
# 10-way split of 500 prompts -> 50 prompts/shard.
#
# Output tags: adapter_iter_2-voice_oeq500_... / final-voice_oeq500_...
# Requires HF_TOKEN (submit with --export=ALL).

#SBATCH --job-name=gen_oeq_voice_ab
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=3:00:00
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
RUNDIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill
IDS=training_mitigations/sycoharm_runs/heldout_oeq_first500_ids.json
NUM_SHARDS=10
VOICE="Address the writer directly as 'you'; never reply in the voice of the other people described."

i=$SLURM_ARRAY_TASK_ID
SHARD=$(( i / 2 ))
if (( i % 2 == 0 )); then
    adapter_flag=(--adapter "$RUNDIR/adapter_iter_2"); WHICH=iter_2
else
    adapter_flag=(--adapter "$RUNDIR"); WHICH=final     # --adapter==run-dir => tag 'final'
fi

echo "[task $i] node=$(hostname) which=$WHICH shard=$SHARD/$NUM_SHARDS"
"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$RUNDIR" --env OEQ \
    "${adapter_flag[@]}" \
    --gen-prefix "$VOICE" --gen-tag voice \
    --test-ids-file "$IDS" --tag-extra oeq500 \
    --device-map auto --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$SHARD" --no-judge
echo "[task $i] done."
