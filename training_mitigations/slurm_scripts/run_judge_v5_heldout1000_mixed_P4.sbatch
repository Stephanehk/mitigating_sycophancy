#!/bin/bash
# Single-run version of run_judge_v5_heldout1000.sbatch: stage the 1000-prompt
# held-out responses for ONLY the completed OEQ mixed_P4 run (training job
# 15941142, which finished iteration 3 / the FINAL adapter on 2026-06-19) and
# score them with the v5 GPT-5-mini endorse judge (narrow action-endorsement
# rate #1/(#0+#1)). CPU-only, network-bound (OpenRouter); gate afterok on the
# matching gen array job so it runs once the iter-3 responses are written.
#
# Requires OPENROUTER_API_KEY in the job env (submit with --export=ALL).
#
# Outputs: sycoharm_runs/inputs/oeq_mixed_P4_final1000.csv         (staged input)
#          sycoharm_runs/endorse_gpt5mini_v5*/scored_oeq_mixed_P4_final1000.csv

#SBATCH --job-name=judge_v5_mixedP4
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/judge_v5_mixedP4_%j.out
#SBATCH --error=training_mitigations/logs/judge_v5_mixedP4_%j.err

set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2; exit 1
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
cd /next/u/stephhk/sycophancy

RUN=runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P4
NAME=oeq_mixed_P4_final1000

echo "[judge] staging $RUN -> sycoharm_runs/inputs/$NAME.csv (FINAL=iter3 adapter)"
"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir "training_mitigations/$RUN" \
    --name "$NAME" --num-shards 5 --expect 1000

echo "[judge] scoring with v5 gpt-5-mini endorse judge"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "$NAME" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100
echo "[judge] done."
