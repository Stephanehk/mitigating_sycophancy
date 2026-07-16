#!/bin/bash
# FINAL-RUN held-out GENERATION on `next` for the NEUTRAL DPO arms, over the
# frozen final_training_eval_data/<env>_test split (1000 prompts, temp 1.0,
# seed 0, 1 sample). Generation only (--no-judge); the v5 endorse judge runs
# afterwards as a separate stage.
#
# WHY THIS EXISTS (vs. marlowe_final_gen.sbatch): the 4 neutral adapters are
# trained on `next` because their veto is the OpenRouter gpt-5-mini judge and
# Marlowe's compute nodes are air-gapped. FINAL_RUNS.md Phase 1b relays them to
# Marlowe for generation; this script is the alternative that keeps them here.
# It MIRRORS marlowe_final_gen.sbatch's OEQ/PAS path exactly — same driver, same
# flags, same 2-shard regime, same tag — so the outputs are drop-in compatible
# with stage_heldout_inputs.py / marlowe_final_endorse_all.sh:
#     tag = final_finaldata_temp1.0_seed0
#     <run>/eval/responses_<tag>_shard{0,1}of2.csv
# The ONLY differences are the site (SLURM account/GPUs) and --device-map auto
# in place of --device cuda:0, so the 8B fits either a single a6000 or an
# a4000 pair. Decode is unchanged, so this is the same experiment on different
# silicon (a6000 rather than H100).
#
# Parametrized via --export:
#   GEN_RUNDIR : run dir holding the FINAL adapter (adapter_config.json at root)
#   GEN_ENV    : OEQ | PAS
# Submit with --array=0-1 (one task per shard). Idempotent: a shard whose
# responses csv already exists is skipped.

#SBATCH --job-name=fgen_neutral
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=training_mitigations/logs/gen2/%x_%A_%a.out
#SBATCH --error=training_mitigations/logs/gen2/%x_%A_%a.err

set -euo pipefail

cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME" training_mitigations/logs/gen2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1   # policy weights are pre-cached; never hit the hub mid-run

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it in your shell before sbatch." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

: "${GEN_RUNDIR:?set GEN_RUNDIR via --export}"
: "${GEN_ENV:?set GEN_ENV via --export}"
case "$GEN_ENV" in
  OEQ|PAS) ;;
  *) echo "ERROR: GEN_ENV must be OEQ|PAS (got $GEN_ENV)" >&2; exit 1 ;;
esac

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS=2   # MUST be 2: stage_heldout_inputs.py reads responses_<tag>_shard{i}of2.csv

# Trained arm: the final merged adapter must be at the run root (written with
# .complete when the 3-iteration loop finishes).
[[ -f "$GEN_RUNDIR/adapter_config.json" ]] || {
    echo "ERROR: no final adapter in $GEN_RUNDIR (training unfinished?)" >&2; exit 1; }

TAG="final_finaldata_temp1.0_seed0"
DONE_CSV="$GEN_RUNDIR/eval/responses_${TAG}_shard${SHARD}of${NUM_SHARDS}.csv"
if [[ -f "$DONE_CSV" ]]; then
    echo "[fgen] shard $SHARD already done ($DONE_CSV) — skipping."
    exit 0
fi

echo "[fgen] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[fgen] env=$GEN_ENV tag=$TAG shard=$SHARD/$NUM_SHARDS run=$GEN_RUNDIR"

"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$GEN_RUNDIR" --env "$GEN_ENV" \
    --use-final-data --final-split test --tag-extra finaldata \
    --device-map auto --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$SHARD" \
    --no-judge --sample-max-tokens 512
echo "[fgen] shard $SHARD done -> $DONE_CSV"
