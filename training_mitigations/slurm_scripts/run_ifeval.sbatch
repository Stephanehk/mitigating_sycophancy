#!/bin/bash
# Generate + programmatically score ONE model configuration on IFEval (541
# instruction-following prompts). Generation runs on a 2x a4000 pair (the 8B is
# split via device_map=auto; ~17GB does not fit one 16GB a4000); scoring is
# CPU-only and runs in the same job right after. Fan out one job per condition.
#
# Parametrized via --export:
#   IFE_TAG     : output stem (e.g. base, becritical, dontbesyco,
#                 pas_mixed_P5, oeq_mixed_P5). Used for BOTH the responses file
#                 and the score summary so they line up.
#   IFE_ADAPTER : (optional) PEFT adapter dir. Empty => bare base model.
#   IFE_SUFFIX  : (optional) prompt-only baseline: "becritical" or "dontbesyco".
#                 Empty => no suffix.
#   IFE_LIMIT   : (optional) only the first N prompts (smoke test).
#
# Requires $HF_TOKEN. Outputs:
#   sycoharm_runs/ifeval/responses/responses_ifeval_<tag>_greedy_seed0.jsonl
#   sycoharm_runs/ifeval/scored/summary_<tag>.json  (+ perprompt_<tag>.csv)

#SBATCH --job-name=ifeval
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=training_mitigations/logs/ifeval_%x_%j.out
#SBATCH --error=training_mitigations/logs/ifeval_%x_%j.err

set -euo pipefail

cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1   # Qwen3-8B pre-cached; IFEval dataset is tiny/cached

if [[ -z "${HF_TOKEN:-}" ]]; then echo "ERROR: HF_TOKEN not set." >&2; exit 1; fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
: "${IFE_TAG:?set IFE_TAG}"
IFE_ADAPTER="${IFE_ADAPTER:-}"
IFE_SUFFIX="${IFE_SUFFIX:-}"
IFE_LIMIT="${IFE_LIMIT:-}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
RESP_DIR=training_mitigations/sycoharm_runs/ifeval/responses
SCORE_DIR=training_mitigations/sycoharm_runs/ifeval/scored

# Model-source flags: adapter dir, or bare base.
src=(--base)
[[ -n "$IFE_ADAPTER" ]] && src=(--adapter "$IFE_ADAPTER")

# Prompt-only suffix flags (match eval_endorsement_heldout baselines exactly).
suf=()
case "$IFE_SUFFIX" in
    becritical) suf=(--be-critical) ;;
    dontbesyco) suf=(--gen-suffix "Don't be sycophantic." --gen-tag dontbesyco) ;;
    "")         suf=() ;;
    *) echo "ERROR: unknown IFE_SUFFIX '$IFE_SUFFIX'" >&2; exit 1 ;;
esac

lim=()
[[ -n "$IFE_LIMIT" ]] && lim=(--limit "$IFE_LIMIT")

echo "[ifeval] node=$(hostname) tag=$IFE_TAG adapter='${IFE_ADAPTER:-BASE}' suffix='${IFE_SUFFIX:-none}'"

"$PY" training_mitigations/eval/eval_ifeval.py \
    "${src[@]}" "${suf[@]}" "${lim[@]}" \
    --tag "$IFE_TAG" --device-map auto --seed 0 --out-dir "$RESP_DIR"

RESP_FILE="$RESP_DIR/responses_ifeval_${IFE_TAG}_greedy_seed0.jsonl"
partial=()
[[ -n "$IFE_LIMIT" ]] && partial=(--allow-partial)

"$PY" training_mitigations/eval/score_ifeval.py \
    --responses "$RESP_FILE" --tag "$IFE_TAG" --out-dir "$SCORE_DIR" "${partial[@]}"
echo "[ifeval] done."
