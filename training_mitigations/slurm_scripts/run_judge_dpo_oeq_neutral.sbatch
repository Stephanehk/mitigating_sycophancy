#!/bin/bash
# Stage + score the 1000-prompt held-out responses for the OEQ DPO-neutral
# baseline (runs/dpo_lora_oeq_qwen3-8b_n1000_neutral), the one neutral baseline
# that was missing from both the endorsement/neutral-rate table and the 4-judge
# quality table. Mirrors run_judge_v5_neutral.sbatch + run_quality_judges_neutral.sbatch
# but for this single run. CPU-only / network-bound (OpenRouter); gate afterok on
# the 10-shard generation array.
#
# Produces:
#   sycoharm_runs/inputs/dpo_oeq_neutral_final1000.csv
#   sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_dpo_oeq_neutral_final1000.csv
#   sycoharm_runs/quality_judges_oeq/{addressee,empathy_*}_dpo_neutral.csv
#
# Requires OPENROUTER_API_KEY (submit with --export=ALL).

#SBATCH --job-name=judge_dpo_oeq_neutral
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

RUN=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_neutral
NAME=dpo_oeq_neutral_final1000

echo "[judge] staging $NAME from $RUN (10 shards)"
"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir "$RUN" --name "$NAME" \
    --tag final_final1000_temp1.0_seed0 \
    --num-shards 10 --expect 1000

echo "[judge] v5 gpt-5-mini endorse judge -> endorse_gpt5mini_v5_temp0/"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "$NAME" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100

echo "[qj] 4 quality judges -> quality_judges_oeq/ (dpo_neutral)"
"$PY" training_mitigations/eval/run_quality_judges_pas.py \
    --env OEQ --conditions dpo_neutral \
    --workers 16 --checkpoint 100 --max-cost-per-run 3.0

echo "[judge] done."
