#!/bin/bash
# Stage + v5-endorse-judge ONE env's 1000-prompt "_gp" generations (final-gp
# tag) from run_gen_distill_gp.sbatch. CPU-only / network-bound; gated afterok
# on that env's gen array. Requires OPENROUTER_API_KEY (submit --export=ALL).
#
# Parametrized via --export:
#   JUDGE_RUNDIR : run dir holding eval/ shards (training_mitigations/runs/...)
#   JUDGE_NAME   : staged dataset stem (e.g. oeq_mixed_P5_nocrit_distill_gp_final1000)
#   NUM_SHARDS   : shard count used at generation

#SBATCH --job-name=judge_distill_gp
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=training_mitigations/logs/judge/%x_%j.out
#SBATCH --error=training_mitigations/logs/judge/%x_%j.err

set -euo pipefail
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2; exit 1
fi
: "${JUDGE_RUNDIR:?set JUDGE_RUNDIR}"
: "${JUDGE_NAME:?set JUDGE_NAME}"
NUM_SHARDS="${NUM_SHARDS:-5}"
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
cd /next/u/stephhk/sycophancy

"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir "$JUDGE_RUNDIR" --name "$JUDGE_NAME" \
    --tag "final-gp_final1000_temp1.0_seed0" --num-shards "$NUM_SHARDS" --expect 1000

"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "$JUDGE_NAME" --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 10.0 \
    --workers 8 --checkpoint 100
echo "[judge] done -> sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_${JUDGE_NAME}.csv"
