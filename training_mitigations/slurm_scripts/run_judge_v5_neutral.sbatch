#!/bin/bash
# Stage the 1000-prompt held-out responses for the 3 NEUTRALITY-baseline runs
# (dpo_pas_neutral, grpo_pas_neutral, grpo_oeq_neutral) and score them with the
# v5 GPT-5-mini endorse judge (0=challenge, 1=explicit-affirm, 2=implicit-affirm,
# 3=neutral -- the label the baselines were TRAINED to maximize). CPU-only,
# network-bound (OpenRouter); meant to be gated afterok on the three generation
# array jobs.
#
# All three were generated from their FINAL adapter in 5 shards, so the staged
# tag is the default final_final1000_temp1.0_seed0 with --num-shards 5.
#
# Requires OPENROUTER_API_KEY in the job env (submit with --export=ALL).
# Outputs: sycoharm_runs/inputs/<name>.csv  and
#          sycoharm_runs/endorse_gpt5mini_v5/scored_<name>.csv (+ printed counts
#          incl. the neutral label-3 rate).

#SBATCH --job-name=judge_v5_neutral
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

# Parallel arrays: run dir | judge-input stem (all final adapter, 5 shards).
RUNS=(runs/dpo_lora_pas_qwen3-8b_n1000_neutral \
      runs/grpo_lora_pas_qwen3-8b_n1000_neutral \
      runs/grpo_lora_oeq_qwen3-8b_n1000_neutral)
NAMES=(dpo_pas_neutral_final1000 \
       grpo_pas_neutral_final1000 \
       grpo_oeq_neutral_final1000)

echo "[judge] staging ${#RUNS[@]} runs into sycoharm_runs/inputs/"
for i in "${!RUNS[@]}"; do
    "$PY" training_mitigations/eval/stage_heldout_inputs.py \
        --run-dir "training_mitigations/${RUNS[$i]}" \
        --name "${NAMES[$i]}" \
        --tag final_final1000_temp1.0_seed0 \
        --num-shards 5 \
        --expect 1000
done

echo "[judge] scoring with v5 gpt-5-mini endorse judge"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "${NAMES[@]}" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100
echo "[judge] done -> sycoharm_runs/endorse_gpt5mini_v5/"
