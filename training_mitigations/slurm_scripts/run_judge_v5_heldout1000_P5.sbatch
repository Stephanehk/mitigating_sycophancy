#!/bin/bash
# Stage the 1000-prompt held-out responses for the two finished P5 post-training
# runs (OEQ n1000 P5 and OEQ n1000 mixed P5) and score them with the v5
# GPT-5-mini endorse judge (narrow action-endorsement rate #1/(#0+#1)).
# CPU-only, network-bound (OpenRouter); gated afterok on the two generation
# array jobs.
#
# Requires OPENROUTER_API_KEY in the job env (submit with --export=ALL).
#
# Outputs: sycoharm_runs/inputs/<name>.csv            (staged judge inputs)
#          sycoharm_runs/endorse_gpt5mini_v5/scored_<name>.csv  (+ printed rate)

#SBATCH --job-name=judge_v5_hd1000_P5
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/judge_v5_hd1000_P5_%j.out
#SBATCH --error=training_mitigations/logs/judge_v5_hd1000_P5_%j.err

set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2; exit 1
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
cd /next/u/stephhk/sycophancy

# run dir  ->  judge-input stem (parallel arrays).
RUNS=(runs/dpo_lora_oeq_qwen3-8b_n1000_P5 \
      runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5)
NAMES=(oeq_P5_final1000 \
       oeq_mixed_P5_final1000)

echo "[judge] staging ${#RUNS[@]} runs into sycoharm_runs/inputs/"
for i in "${!RUNS[@]}"; do
    "$PY" training_mitigations/eval/stage_heldout_inputs.py \
        --run-dir "training_mitigations/${RUNS[$i]}" \
        --name "${NAMES[$i]}" --num-shards 5 --expect 1000
done

echo "[judge] scoring with v5 gpt-5-mini endorse judge"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "${NAMES[@]}" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100
echo "[judge] done -> sycoharm_runs/endorse_gpt5mini_v5/"
