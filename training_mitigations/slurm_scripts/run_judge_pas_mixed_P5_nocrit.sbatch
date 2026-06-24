#!/bin/bash
# Stage + score the 1000-prompt held-out responses for the completed PAS
# mixed_P5_nocrit run (DPO+LoRA, runs/dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit,
# training job 15973539, finished iteration 3 / FINAL adapter on 2026-06-23),
# then run the 4 external quality judges. Mirrors run_judge_v5_heldout1000_pas_mixed_P5.sbatch
# + run_quality_judges_neutral.sbatch. CPU-only / network-bound (OpenRouter);
# gate afterok on the matching 10-shard gen array.
#
# Produces:
#   sycoharm_runs/inputs/pas_mixed_P5_nocrit_final1000.csv
#   sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_pas_mixed_P5_nocrit_final1000.csv
#   sycoharm_runs/quality_judges_pas/{addressee,empathy_*}_mixed_P5_nocrit.csv
#
# Requires OPENROUTER_API_KEY (submit with --export=ALL).

#SBATCH --job-name=judge_pas_mixed_P5_nocrit
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

RUN=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit
NAME=pas_mixed_P5_nocrit_final1000

echo "[judge] staging $NAME from $RUN (10 shards, FINAL adapter)"
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

echo "[qj] 4 quality judges -> quality_judges_pas/ (mixed_P5_nocrit)"
"$PY" training_mitigations/eval/run_quality_judges_pas.py \
    --env PAS --conditions mixed_P5_nocrit \
    --workers 16 --checkpoint 100 --max-cost-per-run 3.0

echo "[judge] done."
