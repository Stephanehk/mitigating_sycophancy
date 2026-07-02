#!/bin/bash
# Stage + v5-endorse-judge the 500-prompt OEQ iter_1 / iter_2 generations from
# run_gen_oeq_iter_eval.sbatch. CPU-only / network-bound (OpenRouter); gated
# afterok on the gen array. Requires OPENROUTER_API_KEY (submit --export=ALL).
#
# Outputs: sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_<name>.csv

#SBATCH --job-name=judge_oeq_iter_eval
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
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
cd /next/u/stephhk/sycophancy

RUNDIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill

for ITER in 1 2; do
    "$PY" training_mitigations/eval/stage_heldout_inputs.py \
        --run-dir "$RUNDIR" \
        --name "oeq_distill_iter${ITER}_500" \
        --tag "adapter_iter_${ITER}_oeq500_temp1.0_seed0" \
        --num-shards 5 --expect 500
done

"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets oeq_distill_iter1_500 oeq_distill_iter2_500 \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 10.0 \
    --workers 8 --checkpoint 100
echo "[judge] done -> sycoharm_runs/endorse_gpt5mini_v5_temp0/"
