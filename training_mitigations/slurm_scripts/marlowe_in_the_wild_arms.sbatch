#!/bin/bash
# In-the-wild comparison arms, run over the SAME 50 Study-3 utterances as the
# Our Model (3+1) dump (the selection is deterministic from the source CSV, so
# every arm sees an identical prompt set).
#
# Array task -> condition:
#   0  base            Qwen3-8B, untrained
#   1  base-becritical Qwen3-8B + the trainer's CRITICAL_SUFFIX ("\n\nBe critical.")
#   2  neutralmixed    dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final
#
# All three write into $ITW_OUT so the four conditions sit side by side.
# GPU generation only, nothing judged. Idempotent: skips a condition whose CSV
# already exists.
#SBATCH --job-name=itwarms
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/itwarms_%A_%a.out
#SBATCH --error=/projects/m000254/logs/itwarms_%A_%a.err

source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_eval_env.sh
set -e

ITW_OUT="${ITW_OUT:-/projects/m000254/runs/in_the_wild}"
mkdir -p "$ITW_OUT"
NEUTRAL_MIXED=/projects/m000254/runs/dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) TAG=base;            ARGS=(--base) ;;
  1) TAG=base-becritical; ARGS=(--base --be-critical) ;;
  2) TAG=neutralmixed;    ARGS=(--adapter "$NEUTRAL_MIXED")
     [[ -f "$NEUTRAL_MIXED/adapter_config.json" ]] || { echo "ERROR: no adapter in $NEUTRAL_MIXED" >&2; exit 1; } ;;
  *) echo "ERROR: unexpected array id ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

DONE_CSV="$ITW_OUT/in_the_wild_study3_${TAG}_temp1.0_seed0.csv"
if [[ -f "$DONE_CSV" ]]; then
  echo "[itwarms] $TAG already done ($DONE_CSV) — skipping."
  exit 0
fi

echo "[itwarms] condition=$TAG out=$ITW_OUT"
"$PY" training_mitigations/eval/in_the_wild_response_analysis.py \
    --n 50 --device cuda:0 --seed 0 --tag "$TAG" --out-dir "$ITW_OUT" \
    "${ARGS[@]}"
echo "[itwarms] $TAG done -> $DONE_CSV"
