#!/bin/bash
# Stage + v5-endorse-judge the voice-constrained iter_2 / iter_3(final) OEQ
# generations from run_gen_oeq_voice_ab.sbatch. Gated afterok on the gen array.
# Requires OPENROUTER_API_KEY (submit --export=ALL).

#SBATCH --job-name=judge_oeq_voice_ab
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

"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir "$RUNDIR" --name oeq_distill_iter2_voice_500 \
    --tag "adapter_iter_2-voice_oeq500_temp1.0_seed0" --num-shards 10 --expect 500
"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir "$RUNDIR" --name oeq_distill_iter3_voice_500 \
    --tag "final-voice_oeq500_temp1.0_seed0" --num-shards 10 --expect 500

"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets oeq_distill_iter2_voice_500 oeq_distill_iter3_voice_500 \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 10.0 \
    --workers 8 --checkpoint 100
echo "[judge] done -> sycoharm_runs/endorse_gpt5mini_v5_temp0/"
