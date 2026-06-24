#!/bin/bash
# Run the 4 external quality judges (J1 addressee + 3 Lend-an-Ear empathy
# sub-components: validating_emotions, demonstrating_understanding,
# dismissing_emotions) over the held-out responses of the NEUTRALITY-baseline
# runs, so they sit on the same axes as base / prompts / mixed_P5. These judges
# are INDEPENDENT of the v5 endorse judge the DPO-neutral model trained against.
#
#   PAS: dpo_neutral, grpo_neutral
#   OEQ: grpo_neutral            (dpo_oeq_neutral still training -> add later)
#
# gpt-5-mini via OpenRouter, CPU-only / network-bound. Resumable: any
# (judge,condition) already complete under quality_judges_<env>/ is skipped.
# Requires OPENROUTER_API_KEY (submit with --export=ALL).
# Outputs: sycoharm_runs/quality_judges_{pas,oeq}/<judge>_<condition>.csv

#SBATCH --job-name=qj_neutral
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

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
cd /next/u/stephhk/sycophancy
mkdir -p training_mitigations/logs/judge

echo "[qj] PAS neutral conditions"
"$PY" training_mitigations/eval/run_quality_judges_pas.py \
    --env PAS --conditions dpo_neutral grpo_neutral \
    --workers 16 --checkpoint 100 --max-cost-per-run 3.0

echo "[qj] OEQ neutral conditions"
"$PY" training_mitigations/eval/run_quality_judges_pas.py \
    --env OEQ --conditions grpo_neutral \
    --workers 16 --checkpoint 100 --max-cost-per-run 3.0

echo "[qj] done -> sycoharm_runs/quality_judges_{pas,oeq}/"
