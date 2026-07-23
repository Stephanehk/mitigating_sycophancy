#!/bin/bash
#
# "Voice-clean" SFT arm: supervised fine-tuning on every CLEAN candidate response
# (assistant-mode label 0) from the voice-DPO candidate pool, as an alternative to
# the voice-DPO preference objective.
#
# RATIONALE. voice-DPO's repetition is length exploitation: its pairs carry a
# systematic length skew (chosen/rejected mean chars = 1.141), so the preference
# signal rewards longer text and the policy buys length with degenerate loops
# (repeaters' median length 2190 vs 878 overall; P6, whose pairs are not built
# this way, sits at 0.0%). Plain NLL has no contrastive term to exploit, and the
# existing SFT arm ("be critical") is the only trained arm at 0.0% repetition,
# 0.0% lowercase-register collapse, and baseline-level voice collapse.
#
# TARGETS. Unlike marlowe_final_train_becritical_sft.sbatch there is NO teacher-gen
# stage: the targets already exist. They are the label-0 candidates from
#   runs/dpo_lora_<stem>_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo/pairs/
#     candidates.jsonl + labels.csv
# staged to <SFT_OUTDIR>/eval/responses_voiceclean_finaltrain_temp1.0_seed0_shard0of1.csv
# as (sentence, response), keeping ALL clean draws per prompt (~8-9 each), and
# dropping unfinished draws (finished=False) so SFT is never taught truncation.
#   OEQ: 9250 clean -> 284 unfinished dropped -> 8966 over 998 prompts
#   PAS: 8167 clean ->   2 unfinished dropped -> 8165 over 999 prompts
# Max response length is 1530 tokens, well inside --max-length 3072.
#
# Hyperparameters are IDENTICAL to the becritical SFT arm and the DPO arms
# (r=16/alpha=32/dropout=0.05, lr 5e-5, effective batch 8, 1 epoch, max_length
# 3072) so the arms stay comparable. 1 epoch over ~9k examples is ~1120 steps,
# roughly 9x the becritical arm's step count, because the dataset is ~9x larger.
#
# Writes to a NEW run dir; nothing existing is touched.
#
#   sbatch --export=ALL,SFT_ENV=OEQ marlowe_train_voiceclean_sft.sbatch
#   sbatch --export=ALL,SFT_ENV=PAS marlowe_train_voiceclean_sft.sbatch
#
#SBATCH --job-name=sft_voiceclean
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/sft_voiceclean_%x_%j.out
#SBATCH --error=/projects/m000254/logs/sft_voiceclean_%x_%j.err

set -euo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
ENV_PREFIX=/projects/$PROJECT/envs/llm126
export HF_HOME=/projects/$PROJECT/hf_cache
RUNS_ROOT=/projects/$PROJECT/runs
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${SFT_ENV:?set SFT_ENV=OEQ|PAS via --export}"
[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: HF_TOKEN not set (Qwen3-8B is gated)." >&2; exit 1; }
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

case "$SFT_ENV" in
  OEQ) STEM=oeq ;;
  PAS) STEM=pas ;;
  *) echo "ERROR: SFT_ENV must be OEQ or PAS (got $SFT_ENV)" >&2; exit 1 ;;
esac

SFT_OUTDIR="$RUNS_ROOT/sft_lora_${STEM}_qwen3-8b_voiceclean_final"
TEACHER="$SFT_OUTDIR/eval/responses_voiceclean_finaltrain_temp1.0_seed0_shard0of1.csv"

cd "$REPO"
PY="$ENV_PREFIX/bin/python"
echo "[voiceclean-sft] env=$SFT_ENV out=$SFT_OUTDIR node=$(hostname)"
echo "[voiceclean-sft] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"

if [[ -f "$SFT_OUTDIR/.complete" ]]; then
  echo "[voiceclean-sft] .complete exists — already trained. Exiting."
  exit 0
fi

# The targets are prebuilt and immutable; fail loudly rather than training on a
# silently-missing or resized set.
[[ -f "$TEACHER" ]] || { echo "ERROR: missing teacher targets $TEACHER" >&2; exit 1; }
EXPECT=$("$PY" -c "import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])[['sentence','response']].dropna()))" "$TEACHER")
echo "[voiceclean-sft] teacher rows = $EXPECT"

"$PY" training_mitigations/train_sft_lora.py \
    --output-dir "$SFT_OUTDIR" \
    --teacher-glob "$TEACHER" \
    --expect "$EXPECT" \
    --device-map auto \
    --seed 0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --learning-rate 5e-5 \
    --per-device-batch-size 1 --grad-accum 8 \
    --num-train-epochs 1 \
    --max-length 3072

echo "[voiceclean-sft] done -> $SFT_OUTDIR"
