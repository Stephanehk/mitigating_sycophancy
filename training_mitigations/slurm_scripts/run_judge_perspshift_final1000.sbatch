#!/bin/bash
# Score the PERSPECTIVE-SHIFT baseline held-out responses (OEQ + PAS, n=1000)
# with the v5 gpt-5-mini ACTION-ENDORSEMENT judge and the three ELEPHANT scorers
# (validation / indirectness / framing) -- EXACTLY the explicit/sharma pipeline,
# so perspective-shift lands on the same footing as the rest of the n=1000 table.
# CPU-only, network-bound (OpenRouter); gate afterok on run_gen_perspshift_final1000.
#
# Steps:
#   1. stage_heldout_inputs.py  -> sycoharm_runs/inputs/<env>_perspshift.csv
#   2. run_endorse_gpt5mini.py  -> sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_<env>_perspshift.csv
#   3. run_elephant_scorers.py  -> sycoharm_runs/elephant_scorers_<env>/<metric>_perspshift.csv
#
# Requires OPENROUTER_API_KEY in the job env (submit with --export=ALL).

#SBATCH --job-name=judge_perspshift
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/judge/%x_%j.out
#SBATCH --error=training_mitigations/logs/judge/%x_%j.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
mkdir -p training_mitigations/logs/judge
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set" >&2; exit 1
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-by-elephant-scorers}"
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

# Stage each env's perspective-shift responses (single final1000 tag, sharded as
# generated). OEQ used --assess-narrative -> base-perspshiftassess; PAS -> base-perspshift.
echo "[stage] OEQ + PAS perspective-shift -> sycoharm_runs/inputs/"
"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000 \
    --name oeq_perspshift \
    --tag base-perspshiftassess_final1000_temp1.0_seed0 \
    --num-shards 6 --expect 1000
"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000 \
    --name pas_perspshift \
    --tag base-perspshift_final1000_temp1.0_seed0 \
    --num-shards 2 --expect 1000

echo "[endorse] v5 gpt-5-mini action-endorsement judge over: oeq_perspshift pas_perspshift"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets oeq_perspshift pas_perspshift \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100

echo "[elephant] validation/indirectness/framing over perspshift, PAS+OEQ"
for env in PAS OEQ; do
    "$PY" training_mitigations/eval/run_elephant_scorers.py \
        --env "$env" --conditions perspshift \
        --workers 16 --checkpoint 100 --max-cost-per-run 3.0
done
echo "[done] scored frames in endorse_gpt5mini_v5_temp0/ + elephant_scorers_{pas,oeq}/"
