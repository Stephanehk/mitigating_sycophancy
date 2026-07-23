#!/bin/bash
#
# Voice-collapse round 2 ("voice2"): RPO on P6, filtered by the EXTENDED voice
# rubric (eval/voice_collapse_prompt_v2.py) which covers collective voice as
# well as singular role-play.
#
# Pipeline (one parametrized job per stage, selected by VS_STAGE):
#   validate -> classify_voice_only.py on an EXISTING labelled candidate pool,
#               into a side file, so v2 can be compared against v1 before any
#               training. Does not touch the pool.
#   sample   -> eval/sample_train_candidates.py : N=10 candidates per FINAL-TRAIN
#               prompt, drawn from the P6 adapter (on-policy).
#   classify -> eval/classify_voice_only.py     : v2 voice rubric ONLY (the
#               addressee/message-critique judge is deliberately not run).
#   train    -> train_dpo_offline_lora.py --rpo-alpha 0.5 : one RPO epoch
#               continuing the P6 adapter. Same alpha/beta/epochs as the P6 run.
# build_dpo_pairs.py (pure CPU) runs inline in the driver between classify and
# train: rejected = every collapsed draw, chosen = a uniformly-sampled clean
# draw for the SAME prompt, capped at 10 pairs/prompt.
#
# Env:
#   VS_STAGE   : validate | sample | classify | train
#   VS_ENV     : OEQ | PAS
#   VS_ADAPTER : abs path to the P6 adapter to sample from / continue
#   VS_OUTDIR  : abs path to the NEW run dir
#   VS_ALPHA   : RPO alpha (default 0.5, matching P6's rpo0p5)
#   VS_BETA    : DPO beta  (default 0.1, matching P6)
#   VS_VALSRC  : validate only -- run dir whose pairs/candidates.jsonl to label
#   VS_VALOUT  : validate only -- output csv
#
#SBATCH --job-name=voice2
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/voice2_%x_%j.out
#SBATCH --error=/projects/m000254/logs/voice2_%x_%j.err

set -euo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
ENV_PREFIX=/projects/$PROJECT/envs/llm126
export HF_HOME=/projects/$PROJECT/hf_cache
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

: "${VS_STAGE:?set VS_STAGE=validate|sample|classify|train}"
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

cd "$REPO"
PY="$ENV_PREFIX/bin/python"
echo "[voice2] stage=$VS_STAGE env=${VS_ENV:-?} job=${SLURM_JOB_ID:-?} node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -

case "$VS_STAGE" in
  validate)
    : "${VS_VALSRC:?}"; : "${VS_VALOUT:?}"
    "$PY" training_mitigations/eval/classify_voice_only.py \
        --out-dir "$VS_VALSRC" --labels-out "$VS_VALOUT" \
        --env "${VS_ENV:-PAS}" --device cuda:0 \
        --limit "${VS_VALN:-1500}" --shuffle-seed 0
    ;;
  sample)
    : "${VS_ENV:?}"; : "${VS_ADAPTER:?}"; : "${VS_OUTDIR:?}"
    "$PY" training_mitigations/eval/sample_train_candidates.py \
        --env "$VS_ENV" --adapter "$VS_ADAPTER" --out-dir "$VS_OUTDIR" \
        --n 10 --temperature 1.0 --top-p 0.95 \
        --max-tokens 1536 --device cuda:0 --seed 0
    ;;
  classify)
    : "${VS_ENV:?}"; : "${VS_OUTDIR:?}"
    "$PY" training_mitigations/eval/classify_voice_only.py \
        --out-dir "$VS_OUTDIR" --env "$VS_ENV" --device cuda:0
    ;;
  train)
    : "${VS_ADAPTER:?}"; : "${VS_OUTDIR:?}"
    if [[ -f "$VS_OUTDIR/.complete" ]]; then
      echo "[voice2] adapter already complete, skipping train"; exit 0
    fi
    "$PY" training_mitigations/train_dpo_offline_lora.py \
        --adapter "$VS_ADAPTER" \
        --out-dir "$VS_OUTDIR" \
        --beta "${VS_BETA:-0.1}" \
        --rpo-alpha "${VS_ALPHA:-0.5}" \
        --num-train-epochs 1 \
        --device cuda:0 --seed 0
    ;;
  *) echo "ERROR: unknown VS_STAGE=$VS_STAGE" >&2; exit 1 ;;
esac

echo "[voice2] OK stage=$VS_STAGE env=${VS_ENV:-?} end=$(date -Is)"
