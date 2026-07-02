#!/bin/bash
# Full scoring of the mixed_P5_nocrit_distill_gp ("_gp" voice arm) held-out
# generations (OEQ + PAS, n=1000) from run_gen_distill_gp.sbatch. Runs ALL three
# judge families so the _gp arm lands on the same footing as every other n=1000
# method:
#   1. stage_heldout_inputs.py   -> sycoharm_runs/inputs/<env>_mixed_P5_nocrit_distill_gp_final1000.csv
#   2. run_endorse_gpt5mini.py   -> endorse_gpt5mini_v5_temp0/scored_<env>_mixed_P5_nocrit_distill_gp_final1000.csv
#   3. run_quality_judges_pas.py -> addressee + 3 Lend-an-Ear empathy sub-components
#   4. run_elephant_scorers.py   -> validation / indirectness / framing
# Steps 3-4 resolve the condition "mixed_P5_nocrit_distill_gp" via
# CONDITIONS_BY_ENV, which points at the scored_*.csv written in step 2.
#
# CPU-only / network-bound (OpenRouter). Gate afterok on BOTH gen arrays
# (OEQ 6 shards, PAS 4 shards). Requires OPENROUTER_API_KEY (submit --export=ALL).

#SBATCH --job-name=judge_distill_gp_full
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

TAG=final-gp_final1000_temp1.0_seed0
OEQ_RUN=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp
PAS_RUN=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp
OEQ_NAME=oeq_mixed_P5_nocrit_distill_gp_final1000
PAS_NAME=pas_mixed_P5_nocrit_distill_gp_final1000

# 1. stage (OEQ gen at 6 shards, PAS gen at 4 shards)
echo "[stage] OEQ + PAS _gp -> sycoharm_runs/inputs/"
"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir "$OEQ_RUN" --name "$OEQ_NAME" --tag "$TAG" --num-shards 6 --expect 1000
"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir "$PAS_RUN" --name "$PAS_NAME" --tag "$TAG" --num-shards 4 --expect 1000

# 2. v5 gpt-5-mini action-endorsement judge (both envs in one invocation)
echo "[endorse] v5 gpt-5-mini over: $OEQ_NAME $PAS_NAME"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "$OEQ_NAME" "$PAS_NAME" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100

# 3. quality judges (addressee + 3 empathy sub-components), resumable
echo "[quality] addressee + Lend-an-Ear empathy over mixed_P5_nocrit_distill_gp"
for env in OEQ PAS; do
    "$PY" training_mitigations/eval/run_quality_judges_pas.py \
        --env "$env" --conditions mixed_P5_nocrit_distill_gp \
        --workers 16 --max-cost-per-run 3.0
done

# 4. ELEPHANT scorers (validation / indirectness / framing)
echo "[elephant] validation/indirectness/framing over mixed_P5_nocrit_distill_gp"
for env in PAS OEQ; do
    "$PY" training_mitigations/eval/run_elephant_scorers.py \
        --env "$env" --conditions mixed_P5_nocrit_distill_gp \
        --workers 16 --checkpoint 100 --max-cost-per-run 3.0
done

echo "[done] _gp scored: endorse_gpt5mini_v5_temp0/ + quality_judges + elephant_scorers_{pas,oeq}/"
