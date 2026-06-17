#!/bin/bash
# Run the sycoharm judge (judge_sycophancy.py) via OpenRouter gpt-5.4-mini
# with minimal reasoning effort and no max_tokens cap over 500-row input CSVs
# built by build_sycoharm_inputs.py (held-out Qwen3-8B responses for PAS/OEQ
# -- base / base+"Be critical." / stakeholder-DPO-trained -- plus 500-row
# samples of the paper's PAS_results and OEQ_results deepseek/gpt4o sets).
#
# Submit from the repo root with OPENROUTER_API_KEY exported (sbatch passes
# the submitter's environment through by default). Dataset names (basenames
# of sycoharm_runs/inputs/<name>.csv) are taken from the script args,
# defaulting to the original six:
#   sbatch training_mitigations/slurm_scripts/run_sycoharm_gpt54mini.sbatch \
#       [name ...]
#
# CPU-only (the judge is pure API calls, throttled at 4 req/s by the reused
# scorer -> ~13 min for 3000 calls; the 2h limit is pure slack). Outputs:
#   training_mitigations/sycoharm_runs/scored_<name>.csv
# Each run also appends its level histogram to the job log.

#SBATCH --job-name=sycoharm_gpt54mini
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=training_mitigations/logs/sycoharm_gpt54mini_%j.out
#SBATCH --error=training_mitigations/logs/sycoharm_gpt54mini_%j.err

set -euo pipefail

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
TM=/next/u/stephhk/sycophancy/training_mitigations
RUNS=$TM/sycoharm_runs

test -n "${OPENROUTER_API_KEY:-}" || { echo "OPENROUTER_API_KEY not set" >&2; exit 1; }

NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then
    NAMES=(heldout_pas heldout_oeq pas_deepseek pas_gpt4o oeq_deepseek oeq_gpt4o)
fi

for name in "${NAMES[@]}"; do
    test -f "$RUNS/inputs/$name.csv" || { echo "missing input $RUNS/inputs/$name.csv" >&2; exit 1; }
    echo "===== $name ====="
    $PY $TM/judge_sycophancy.py $RUNS/inputs/$name.csv \
        --advice response \
        --output $RUNS/scored_$name.csv \
        --workers 8 --checkpoint 100 \
        --openrouter --judge-deployment openai/gpt-5.4-mini \
        --reasoning-effort minimal --no-max-tokens \
        --price-in 0.75 --price-out 4.50 --max-cost 4.0
done
echo "ALL DONE"
