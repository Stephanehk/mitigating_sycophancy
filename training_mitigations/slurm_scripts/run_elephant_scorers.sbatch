#!/bin/bash
# Run the ELEPHANT scorers (validation / indirectness / framing) over the saved
# 1000-response held-out frames with the gpt-5-mini v5 judge. CPU-ONLY /
# network-bound (OpenRouter) — it reads responses already on disk, so it needs
# NO GPU and coexists with GPU jobs on the same node. Select env via JUDGE_ENV.
#
# Parametrized via --export:
#   JUDGE_ENV : OEQ | PAS
#
# Requires OPENROUTER_API_KEY (submit with --export=ALL,JUDGE_ENV=...).
# Outputs: sycoharm_runs/elephant_scorers_<env>/<metric>_<condition>.csv

#SBATCH --job-name=elephant_scorers
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
: "${JUDGE_ENV:?set JUDGE_ENV (OEQ|PAS)}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
cd /next/u/stephhk/sycophancy
mkdir -p training_mitigations/logs/judge

# Optional: restrict to a subset of conditions (space-separated). Empty => the
# script's DEFAULT_CONDITIONS for the env.
cond_flag=()
if [[ -n "${ELE_CONDS:-}" ]]; then
    cond_flag=(--conditions ${ELE_CONDS})
fi

echo "[elephant] env=$JUDGE_ENV conds='${ELE_CONDS:-DEFAULT}' node=$(hostname)"
"$PY" training_mitigations/eval/run_elephant_scorers.py \
    --env "$JUDGE_ENV" "${cond_flag[@]}" \
    --workers 16 --checkpoint 100 --max-cost-per-run 3.0
echo "[elephant] done -> sycoharm_runs/elephant_scorers_${JUDGE_ENV,,}/"
