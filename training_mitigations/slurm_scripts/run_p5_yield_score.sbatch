#!/bin/bash
# Score the P5-yield candidates with the v5 gpt-5-mini judge and print the
# GO/NO-GO verdict. CPU-only (OpenRouter); gated afterok on the veto array.
#SBATCH --job-name=p5_yield_score
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=training_mitigations/logs/p5_yield_score_%j.out
#SBATCH --error=training_mitigations/logs/p5_yield_score_%j.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
"$PY" training_mitigations/eval/p5_yield_score.py --workers 8
echo "[score] done."
