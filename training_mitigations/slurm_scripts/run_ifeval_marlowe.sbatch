#!/bin/bash
# IFEval (541 instruction-following prompts) generate + score for ONE model, on
# MARLOWE (H100). This is the next-cluster run_ifeval.sbatch ported to Marlowe:
# single H100 (the 8B fits 80GB, no device_map split needed), llm126 env,
# project paths, and the vendored google-research scorer at the sycophancy repo
# root (fetched 2026-07-22). Generation is greedy/seed0; scoring is CPU and runs
# inline right after. One job per adapter.
#
# Parametrized via --export:
#   IFE_TAG      : output stem (e.g. ourmodel_oeq). Used for responses + summary.
#   IFE_ADAPTER  : PEFT adapter dir. Empty => bare base Qwen3-8B.
#
# Outputs (under training_mitigations/sycoharm_runs/ifeval/):
#   responses/responses_ifeval_<tag>_greedy_seed0.jsonl
#   scored/summary_<tag>.json  (+ perprompt_<tag>.csv)
#
#SBATCH --job-name=ifeval
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/ifeval_%x_%j.out
#SBATCH --error=/projects/m000254/logs/ifeval_%x_%j.err

set -euo pipefail
PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
PY=/projects/$PROJECT/envs/llm126/bin/python
export HF_HOME=/projects/$PROJECT/hf_cache
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1              # base model cached; IFEval data is a local jsonl
export TOKENIZERS_PARALLELISM=false
# vendored google-research scorer lives at the repo root -> importable by score_ifeval
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

: "${IFE_TAG:?set IFE_TAG}"
IFE_ADAPTER="${IFE_ADAPTER:-}"
cd "$REPO"

RESP_DIR=training_mitigations/sycoharm_runs/ifeval/responses
SCORE_DIR=training_mitigations/sycoharm_runs/ifeval/scored
RESP_FILE="$RESP_DIR/responses_ifeval_${IFE_TAG}_greedy_seed0.jsonl"
SUMMARY="$SCORE_DIR/summary_${IFE_TAG}.json"

if [[ -f "$SUMMARY" ]]; then echo "[ifeval] $SUMMARY exists — done, skipping."; exit 0; fi

src=(--base)
[[ -n "$IFE_ADAPTER" ]] && { [[ -f "$IFE_ADAPTER/adapter_config.json" ]] || { echo "ERROR: no adapter at $IFE_ADAPTER" >&2; exit 1; }; src=(--adapter "$IFE_ADAPTER"); }

echo "[ifeval] node=$(hostname) tag=$IFE_TAG adapter='${IFE_ADAPTER:-BASE}' start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -

# 1. generate (single shard; 8B on one H100, greedy, seed 0)
"$PY" training_mitigations/eval/eval_ifeval.py \
    "${src[@]}" --tag "$IFE_TAG" --device cuda:0 --seed 0 --out-dir "$RESP_DIR"

# 2. score (CPU; vendored instruction_following_eval strict + loose)
"$PY" training_mitigations/eval/score_ifeval.py \
    --responses "$RESP_FILE" --tag "$IFE_TAG" --out-dir "$SCORE_DIR"

echo "[ifeval] done tag=$IFE_TAG end=$(date -Is) -> $SUMMARY"
