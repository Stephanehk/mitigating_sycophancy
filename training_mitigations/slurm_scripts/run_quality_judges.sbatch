#!/bin/bash
# Run the 4 response-quality judges (addressee + 3 Lend-an-Ear empathy
# sub-components) for ONE (env, condition) over its 1000 held-out responses,
# via the gpt-5-mini OpenRouter judge. CPU/network-bound (no GPU): the
# responses already exist under sycoharm_runs/endorse_gpt5mini_v5_temp0/, so
# this only makes API calls. Fan out one job per condition to parallelize the
# OpenRouter traffic. Resumable: a completed (judge,condition) CSV is skipped.
#
# Parametrized via --export:
#   QJ_ENV   : PAS | OEQ
#   QJ_CONDS : space-separated condition keys (e.g. "mixed_P5" or
#              "base becritical dontbesyco mixed_P5")
#
# Requires OPENROUTER_API_KEY in the job env (submit with --export=ALL,...).
# Outputs: sycoharm_runs/quality_judges_<env>/<judge>_<cond>.csv

#SBATCH --job-name=qjudge
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=training_mitigations/logs/qjudge_%x_%j.out
#SBATCH --error=training_mitigations/logs/qjudge_%x_%j.err

set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2; exit 1
fi
: "${QJ_ENV:?set QJ_ENV}"
: "${QJ_CONDS:?set QJ_CONDS}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
cd /next/u/stephhk/sycophancy/training_mitigations/eval

echo "[qjudge] env=$QJ_ENV conditions=[$QJ_CONDS]"
"$PY" run_quality_judges_pas.py \
    --env "$QJ_ENV" --conditions $QJ_CONDS \
    --workers 16 --checkpoint 100 --max-cost-per-run 3.0
echo "[qjudge] done."
