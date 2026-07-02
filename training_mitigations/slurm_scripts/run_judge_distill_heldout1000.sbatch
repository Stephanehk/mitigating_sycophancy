#!/bin/bash
# Score the 1000-prompt held-out responses of the two mixed_P5_nocrit_distill
# runs (OEQ + PAS) with the full judge stack:
#   (a) v5 gpt-5-mini action-endorsement judge  -> endorse_gpt5mini_v5_temp0/scored_<name>.csv
#   (b) the 4 quality judges (addressee + 3 Lend-an-Ear empathy sub-components)
#   (c) the ELEPHANT framing + validation scorers
# CPU-only / network-bound (OpenRouter). Reads gen outputs already on disk, so
# it carries NO GPU and is gated afterok on the two 5-shard gen arrays.
#
# Requires OPENROUTER_API_KEY (submit with --export=ALL).

#SBATCH --job-name=judge_distill_hd1000
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

# run dir  ->  judge-input stem (parallel arrays).
RUNS=(runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill \
      runs/dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit_distill)
NAMES=(oeq_mixed_P5_nocrit_distill_final1000 \
       pas_mixed_P5_nocrit_distill_final1000)

echo "[judge] staging ${#RUNS[@]} runs into sycoharm_runs/inputs/"
for i in "${!RUNS[@]}"; do
    "$PY" training_mitigations/eval/stage_heldout_inputs.py \
        --run-dir "training_mitigations/${RUNS[$i]}" \
        --name "${NAMES[$i]}" --num-shards 5 --expect 1000
done

echo "[judge] (a) v5 gpt-5-mini action-endorsement judge"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "${NAMES[@]}" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100

echo "[judge] (b) 4 quality judges (addressee + 3 empathy sub-components)"
for env in OEQ PAS; do
    "$PY" training_mitigations/eval/run_quality_judges_pas.py \
        --env "$env" --conditions mixed_P5_nocrit_distill \
        --workers 16 --max-cost-per-run 3.0
done

echo "[judge] (c) ELEPHANT framing + validation scorers"
for env in OEQ PAS; do
    "$PY" training_mitigations/eval/run_elephant_scorers.py \
        --env "$env" --conditions mixed_P5_nocrit_distill \
        --metrics validation framing \
        --workers 16 --checkpoint 100 --max-cost-per-run 3.0
done

echo "[judge] done."
