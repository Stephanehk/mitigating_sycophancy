#!/bin/bash
# Stage + score the 1000-prompt held-out responses for a NEUTRAL-MIXED DPO run,
# then run the 4 external quality judges. Parametrized so one file serves both
# the OEQ and PAS neutral-mixed runs; CPU-only / network-bound (OpenRouter).
# Gate afterok on the matching 10-shard gen array.
#
# Parametrized via --export:
#   JUDGE_RUN  : run dir holding the final adapter + eval/ shards
#   JUDGE_NAME : staged-input stem (e.g. dpo_oeq_neutral_mixed_final1000)
#   JUDGE_ENV  : OEQ | PAS  (selects the quality-judge env)
#   JUDGE_COND : quality-judge condition key (neutral_mixed)
#
# Produces:
#   sycoharm_runs/inputs/<JUDGE_NAME>.csv
#   sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_<JUDGE_NAME>.csv
#   sycoharm_runs/quality_judges_<env>/{addressee,empathy_*}_<JUDGE_COND>.csv
#
# Requires OPENROUTER_API_KEY (submit with --export=ALL,...).

#SBATCH --job-name=judge_neutral_mixed
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/judge/%x_%j.out
#SBATCH --error=training_mitigations/logs/judge/%x_%j.err

set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2; exit 1
fi
: "${JUDGE_RUN:?set JUDGE_RUN}"
: "${JUDGE_NAME:?set JUDGE_NAME}"
: "${JUDGE_ENV:?set JUDGE_ENV}"
: "${JUDGE_COND:?set JUDGE_COND}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
cd /next/u/stephhk/sycophancy
mkdir -p training_mitigations/logs/judge

echo "[judge] staging $JUDGE_NAME from $JUDGE_RUN (10 shards, FINAL adapter)"
"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir "$JUDGE_RUN" --name "$JUDGE_NAME" \
    --tag final_final1000_temp1.0_seed0 \
    --num-shards 10 --expect 1000

echo "[judge] v5 gpt-5-mini endorse judge -> endorse_gpt5mini_v5_temp0/"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "$JUDGE_NAME" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100

echo "[qj] 4 quality judges -> quality_judges_${JUDGE_ENV,,}/ ($JUDGE_COND)"
"$PY" training_mitigations/eval/run_quality_judges_pas.py \
    --env "$JUDGE_ENV" --conditions "$JUDGE_COND" \
    --workers 16 --checkpoint 100 --max-cost-per-run 3.0

echo "[judge] done."
