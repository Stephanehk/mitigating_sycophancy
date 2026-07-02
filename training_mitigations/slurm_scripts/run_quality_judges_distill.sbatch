#!/bin/bash
# Finish ONLY the 4 quality judges (addressee + 3 Lend-an-Ear empathy
# sub-components) for the mixed_P5_nocrit_distill condition, OEQ + PAS.
# Resumable: run_quality_judges_pas.py skips any (judge,condition) whose output
# CSV is already complete, so this picks up where the cancelled combined judge
# job left off. Deliberately does NOT run the ELEPHANT scorers.
#
# Requires OPENROUTER_API_KEY (submit with --export=ALL).

#SBATCH --job-name=qjudge_distill
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

for env in OEQ PAS; do
    "$PY" training_mitigations/eval/run_quality_judges_pas.py \
        --env "$env" --conditions mixed_P5_nocrit_distill \
        --workers 16 --max-cost-per-run 3.0
done
echo "[qjudge] done (ELEPHANT scorers intentionally skipped)."
