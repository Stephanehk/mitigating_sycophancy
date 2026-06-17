#!/bin/bash
# Run the sycoharm judge (judge_sycophancy.py, our training-mitigations judge)
# via OpenRouter gpt-5.4-mini over the 500 held-out PAS responses produced by
# the FINAL adapters of the n1000_mixed and n1000_selfpref DPO runs.
#
# Intended to run AFTER run_eval_pas_mixed_selfpref_heldout.sbatch via
#   sbatch --dependency=afterok:<gen_array_jobid> \
#       training_mitigations/slurm_scripts/run_sycoharm_pas_mixed_selfpref.sbatch
# (sbatch passes the submitter's env through, so export OPENROUTER_API_KEY).
#
# Step 1 assembles the two 500-row sentence+response input CSVs from the eval
# shards (build_sycoharm_inputs.py --variant); step 2 judges each on the 0-3
# sycoharm scale. CPU-only (pure throttled API calls). Outputs:
#   training_mitigations/sycoharm_runs/inputs/heldout_pas_{mixed,selfpref}.csv
#   training_mitigations/sycoharm_runs/scored_heldout_pas_{mixed,selfpref}.csv
# Each judge run appends its level histogram + mean to this job's log.

#SBATCH --job-name=sycoharm_pas_mix_self
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=training_mitigations/logs/sycoharm_pas_mix_self_%j.out
#SBATCH --error=training_mitigations/logs/sycoharm_pas_mix_self_%j.err

set -euo pipefail

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
TM=/next/u/stephhk/sycophancy/training_mitigations
RUNS=$TM/sycoharm_runs

test -n "${OPENROUTER_API_KEY:-}" || { echo "OPENROUTER_API_KEY not set" >&2; exit 1; }

# 1. Assemble the two 500-row input CSVs from the freshly generated eval shards.
$PY $TM/build_sycoharm_inputs.py \
    --variant dpo_lora_pas_qwen3-8b_n1000_mixed:heldout_pas_mixed \
    --variant dpo_lora_pas_qwen3-8b_n1000_selfpref:heldout_pas_selfpref

# 2. Run the sycoharm judge on each (same config as run_sycoharm_gpt54mini).
for name in heldout_pas_mixed heldout_pas_selfpref; do
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
