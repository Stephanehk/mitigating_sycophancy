#!/bin/bash
# Parametrized GENERATION-ONLY job: sample held-out responses (HF path) for a
# given run-dir / env / id-set / baseline flags, writing responses_<tag>.csv
# under <run-dir>/eval for the downstream v5 sycoharm judge. No Azure judge.
#
# Submit via run_gen_set2_submit.sh, which exports:
#   GEN_RUNDIR  run dir (for env inference + output eval/ dir + adapter)
#   GEN_IDS     explicit held-out ids json (--test-ids-file)
#   GEN_TAGX    tag-extra token (e.g. set2 / final1000) to avoid clobbering
#   GEN_BASE    "1" to add --base (untrained baseline); else evaluates adapter
#   GEN_BECRIT  "1" to add --be-critical
#   GEN_SUFFIX  prompt-only instruction suffix (e.g. "Don't be sycophantic."); empty to skip
#   GEN_TAG     filename tag for GEN_SUFFIX (e.g. dontbesyco)
#
# 8B in bf16 doesn't fit one 16GB a4000 -> request TWO and shard via device-map.
#SBATCH --job-name=gen_set2
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=training_mitigations/logs/gen_set2_%j.out
#SBATCH --error=training_mitigations/logs/gen_set2_%j.err

set -euo pipefail
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -z "${HF_TOKEN:-}" ]]; then echo "ERROR: HF_TOKEN not set" >&2; exit 1; fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] RUNDIR=$GEN_RUNDIR IDS=$GEN_IDS TAGX=$GEN_TAGX BASE=${GEN_BASE:-} BECRIT=${GEN_BECRIT:-} SUFFIX=${GEN_SUFFIX:-} TAG=${GEN_TAG:-}"

ARGS=(--run-dir "$GEN_RUNDIR" --test-ids-file "$GEN_IDS" --tag-extra "$GEN_TAGX"
      --device-map auto --no-judge --seed 0)
if [[ -n "${GEN_NSHARDS:-}" ]]; then ARGS+=(--num-shards "$GEN_NSHARDS" --shard-id "$GEN_SHARD"); fi
[[ "${GEN_BASE:-}" == "1" ]] && ARGS+=(--base)
[[ "${GEN_BECRIT:-}" == "1" ]] && ARGS+=(--be-critical)
if [[ -n "${GEN_SUFFIX:-}" ]]; then ARGS+=(--gen-suffix "$GEN_SUFFIX" --gen-tag "$GEN_TAG"); fi

"$PY" training_mitigations/eval/eval_endorsement_heldout.py "${ARGS[@]}"
echo "[sbatch] done -> $GEN_RUNDIR/eval/"
