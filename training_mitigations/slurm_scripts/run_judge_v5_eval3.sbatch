#!/bin/bash
# Stage the 1000-prompt held-out responses for the 3 runs being evaluated now
# (OEQ mixed_P4 @ iter2, OEQ mixed_selfpref_P4 @ iter2, PAS mixed_selfpref @
# final) and score them with the v5 GPT-5-mini endorse judge (narrow
# action-endorsement rate #1/(#0+#1)). CPU-only, network-bound (OpenRouter);
# meant to be gated afterok on the three generation array jobs.
#
# Per-run staging uses the run's own response tag + shard count (the two P4 runs
# were generated from adapter_iter_2 in 5 shards; PAS from its final adapter in
# 4 shards), concatenates the per-shard responses CSVs into
# sycoharm_runs/inputs/<name>.csv, then v5-judges all three.
#
# Requires OPENROUTER_API_KEY in the job env (submit with --export=ALL).
# Outputs: sycoharm_runs/inputs/<name>.csv  and
#          sycoharm_runs/endorse_gpt5mini_v5/scored_<name>.csv (+ printed rate)

#SBATCH --job-name=judge_v5_eval3
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

# Parallel arrays: run dir | judge-input stem | response tag | shard count.
RUNS=(runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P4 \
      runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_selfpref_P4 \
      runs/dpo_lora_pas_qwen3-8b_n1000_mixed_selfpref)
NAMES=(oeq_mixed_P4_iter2_final1000 \
       oeq_msp_P4_iter2_final1000 \
       pas_msp_final1000)
TAGS=(adapter_iter_2_final1000_temp1.0_seed0 \
      adapter_iter_2_final1000_temp1.0_seed0 \
      final_final1000_temp1.0_seed0)
SHARDS=(5 5 4)

echo "[judge] staging ${#RUNS[@]} runs into sycoharm_runs/inputs/"
for i in "${!RUNS[@]}"; do
    "$PY" training_mitigations/eval/stage_heldout_inputs.py \
        --run-dir "training_mitigations/${RUNS[$i]}" \
        --name "${NAMES[$i]}" \
        --tag "${TAGS[$i]}" \
        --num-shards "${SHARDS[$i]}" \
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
