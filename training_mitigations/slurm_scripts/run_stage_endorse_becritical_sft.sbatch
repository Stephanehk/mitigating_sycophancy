#!/bin/bash
# Stage the held-out-1000 responses generated from the "Be critical." SFT
# adapters (OEQ + PAS) and score them with the DEFAULT v5 gpt-5-mini endorse
# judge (narrow action-endorsement rate #1/(#0+#1)). CPU-only, network-bound
# (OpenRouter); gated afterok on the two generation array jobs.
#
# The scored_*.csv frames this writes under endorse_gpt5mini_v5_temp0/ are the
# SAME frames the ELEPHANT scorers and quality judges read, so this step both
# (a) produces the default endorsement result and (b) feeds the other 7 judges.
#
# Requires OPENROUTER_API_KEY in the job env (submit with --export=ALL).
#
# Outputs: sycoharm_runs/inputs/{oeq,pas}_becritical_sft_final1000.csv
#          sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_<name>.csv (+ rate)

#SBATCH --job-name=endorse_becritical_sft
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

# run dir  ->  judge-input stem (parallel arrays). The SFT adapters live
# directly in these run dirs; gen used --tag-extra final1000 on the final
# adapter, so the response tag is final_final1000_temp1.0_seed0.
RUNS=(training_mitigations/runs/sft_lora_oeq_qwen3-8b_n1000_becritical \
      training_mitigations/runs/sft_lora_pas_qwen3-8b_n1000_becritical)
NAMES=(oeq_becritical_sft_final1000 \
       pas_becritical_sft_final1000)

echo "[stage] ${#RUNS[@]} SFT runs -> sycoharm_runs/inputs/"
for i in "${!RUNS[@]}"; do
    "$PY" training_mitigations/eval/stage_heldout_inputs.py \
        --run-dir "${RUNS[$i]}" --name "${NAMES[$i]}" \
        --tag final_final1000_temp1.0_seed0 --num-shards 2 --expect 1000
done

echo "[judge] scoring with default v5 gpt-5-mini endorse judge"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "${NAMES[@]}" \
    --prompt-version v5 --temp 0 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100
echo "[endorse] done -> sycoharm_runs/endorse_gpt5mini_v5_temp0/"
