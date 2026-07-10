#!/bin/bash
#
# FINAL-RUN "Be critical." SFT baseline on Marlowe (batch/free), parametrized by
# env. Two stages IN ONE JOB on a single H100 (the 8B bf16 fits 80GB, so no
# device_map sharding is needed the way the 2x a4000 `next` port required):
#
#   (1) TEACHER-GEN: sample ONE response per FINAL-TRAIN prompt from the UNTRAINED
#       base Qwen3-8B with "\n\nBe critical." appended (eval_endorsement_heldout.py
#       --base --be-critical --no-judge --use-final-data --final-split train). The
#       recorded `sentence` is the BARE prompt, so the SFT step learns
#       bare-prompt -> critical-response. Writes
#       responses_base-becritical_finaltrain_temp1.0_seed0_shard0of1.csv into
#       <SFT_OUTDIR>/eval/ (the SFT trainer's default teacher glob matches it).
#   (2) SFT: train_sft_lora.py distills those teacher targets into the bare-prompt
#       policy (plain NLL on the response, same LoRA/optimizer config as the DPO
#       arms). Adapter + .complete land in <SFT_OUTDIR>.
#
# Both stages are LOCAL (no OpenRouter/judge egress), so this runs fine on an
# air-gapped Marlowe compute node — unlike the neutral DPO arms.
#
# Parametrized via --export:
#   SFT_ENV    : OEQ | PAS
#   SFT_EXPECT : teacher-pair count = final-train size (default 1000; OEQ/PAS=1000)
# Submit (one per env), from a Marlowe login node:
#   export HF_TOKEN=...
#   sbatch --export=ALL,SFT_ENV=OEQ training_mitigations/slurm_scripts/marlowe_final_train_becritical_sft.sbatch
#   sbatch --export=ALL,SFT_ENV=PAS training_mitigations/slurm_scripts/marlowe_final_train_becritical_sft.sbatch
#SBATCH --job-name=sft_becritical_final
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/sft_becritical_final_%x_%j.out
#SBATCH --error=/projects/m000254/logs/sft_becritical_final_%x_%j.err

# -e so a failed stage aborts (FAILED) instead of falling through to SFT / success.
set -euo pipefail

# ===================== Marlowe site paths (mirror _marlowe_dpo_body.sh) =======
PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
ENV_PREFIX=/projects/$PROJECT/envs/llm126
export HF_HOME=/projects/$PROJECT/hf_cache
RUNS_ROOT=/projects/$PROJECT/runs
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# =============================================================================

: "${SFT_ENV:?set SFT_ENV=OEQ|PAS via --export}"
SFT_EXPECT="${SFT_EXPECT:-1000}"
[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: HF_TOKEN not set (Qwen3-8B is gated)." >&2; exit 1; }
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

case "$SFT_ENV" in
  OEQ) STEM=oeq ;;
  PAS) STEM=pas ;;
  *) echo "ERROR: SFT_ENV must be OEQ or PAS (got $SFT_ENV)" >&2; exit 1 ;;
esac
SFT_OUTDIR="$RUNS_ROOT/sft_lora_${STEM}_qwen3-8b_becritical_final"
mkdir -p "$SFT_OUTDIR/eval" "$HF_HOME"

cd "$REPO"
PY="$ENV_PREFIX/bin/python"
echo "[sft-final] env=$SFT_ENV out=$SFT_OUTDIR expect=$SFT_EXPECT node=$(hostname)"
echo "[sft-final] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"

# Already trained? nothing to do.
if [[ -f "$SFT_OUTDIR/.complete" ]]; then
  echo "[sft-final] $SFT_OUTDIR/.complete exists — already trained. Exiting."
  exit 0
fi

# ----- (1) teacher-gen over the FINAL TRAIN split (idempotent) -----------------
# MUST be run with num-shards >= 2: the endorsement eval only appends the
# _shard{i}of{N} suffix when N>1, and train_sft_lora.py's teacher glob
# (responses_*becritical*shard*.csv) REQUIRES "shard" in the filename. Two
# sequential 500-prompt shards on the one H100 -> shard{0,1}of2, concat = 1000.
NUM_TSHARDS=2
for si in 0 1; do
  csv="$SFT_OUTDIR/eval/responses_base-becritical_finaltrain_temp1.0_seed0_shard${si}of${NUM_TSHARDS}.csv"
  if [[ -f "$csv" ]]; then
    echo "[sft-final] teacher shard $si present — skipping ($csv)"
    continue
  fi
  echo "[sft-final] generating be-critical teacher targets, shard $si/$NUM_TSHARDS ($SFT_ENV final-train)..."
  "$PY" training_mitigations/eval/eval_endorsement_heldout.py \
      --run-dir "$SFT_OUTDIR" --env "$SFT_ENV" \
      --base --be-critical \
      --use-final-data --final-split train --tag-extra finaltrain \
      --device cuda:0 --seed 0 \
      --sample-max-tokens 1024 --temperature 1.0 --top-p 0.95 \
      --num-shards "$NUM_TSHARDS" --shard-id "$si" --no-judge
done

# ----- (2) SFT distill the teacher targets into the bare-prompt policy ----------
echo "[sft-final] SFT on teacher targets (expect=$SFT_EXPECT)..."
"$PY" training_mitigations/train_sft_lora.py \
    --output-dir "$SFT_OUTDIR" \
    --expect "$SFT_EXPECT" \
    --device-map auto \
    --seed 0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --learning-rate 5e-5 \
    --per-device-batch-size 1 --grad-accum 8 \
    --num-train-epochs 1 \
    --max-length 3072
echo "[sft-final] done -> $SFT_OUTDIR"
