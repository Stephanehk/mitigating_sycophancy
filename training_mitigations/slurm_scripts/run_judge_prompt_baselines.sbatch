#!/bin/bash
# Score the two prompt-only baseline (explicit / sharma) held-out responses with
# the v5 gpt-5-mini ACTION-ENDORSEMENT judge and the three ELEPHANT scorers
# (validation / indirectness / framing). CPU-only, network-bound (OpenRouter);
# gated afterok on run_gen_prompt_baselines (all 8 gen shards).
#
# Steps:
#   1. stage_heldout_inputs.py  -> sycoharm_runs/inputs/<env>_<cond>.csv
#   2. run_endorse_gpt5mini.py  -> sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_<env>_<cond>.csv
#   3. run_elephant_scorers.py  -> sycoharm_runs/elephant_scorers_<env>/<metric>_<cond>.csv
#
# Requires OPENROUTER_API_KEY in the job env (submit with --export=ALL).

#SBATCH --job-name=judge_prompt_baselines
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/judge/%x_%j.out
#SBATCH --error=training_mitigations/logs/judge/%x_%j.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
mkdir -p training_mitigations/logs/judge
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set" >&2; exit 1
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-by-elephant-scorers}"
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

# (env cond flag-tag) parallel arrays. flag-tag matches eval_endorsement_heldout
# adapter_tag (base-<flag>) so the response filenames resolve.
ENVS=(pas pas oeq oeq)
CONDS=(explicit sharma explicit sharma)
TAGS=(base-explicit base-sharma base-explicit base-sharma)

echo "[stage] 4 conditions -> sycoharm_runs/inputs/"
NAMES=()
for i in "${!ENVS[@]}"; do
    name="${ENVS[$i]}_${CONDS[$i]}"
    NAMES+=("$name")
    "$PY" training_mitigations/eval/stage_heldout_inputs.py \
        --run-dir "training_mitigations/runs/dpo_lora_${ENVS[$i]}_qwen3-8b_n1000" \
        --name "$name" \
        --tag "${TAGS[$i]}_final1000_temp1.0_seed0" \
        --num-shards 2 --expect 1000
done

echo "[endorse] v5 gpt-5-mini action-endorsement judge over: ${NAMES[*]}"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "${NAMES[@]}" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100

echo "[elephant] validation/indirectness/framing over explicit+sharma, PAS+OEQ"
for env in PAS OEQ; do
    "$PY" training_mitigations/eval/run_elephant_scorers.py \
        --env "$env" --conditions explicit sharma \
        --workers 16 --checkpoint 100 --max-cost-per-run 3.0
done
echo "[done] scored frames in endorse_gpt5mini_v5_temp0/ + elephant_scorers_{pas,oeq}/"
