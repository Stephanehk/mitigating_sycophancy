#!/bin/bash
# Launch the 1000-prompt held-out eval for the 3 NEUTRALITY-baseline runs:
#   dpo_pas_neutral   @ final adapter -> PAS held-out, 5 shards x (2x a4000)
#   grpo_pas_neutral  @ final adapter -> PAS held-out, 5 shards x (2x a4000)
#   grpo_oeq_neutral  @ final adapter -> OEQ held-out, 5 shards x (2x a4000)
# then one v5 gpt-5-mini judge job gated afterok on all three gen arrays.
#
# Each gen job depends on nothing (final adapters are on disk). Shards are 200
# prompts; the 8B is split across a 2x a4000 pair per shard (device-map auto).
# 15 shards total across next5+next6 (20 a4000) -> ~10 concurrent, rest queue.
#
# Run from repo root with HF_TOKEN and OPENROUTER_API_KEY exported.
set -euo pipefail
cd /next/u/stephhk/sycophancy

: "${HF_TOKEN:?export HF_TOKEN}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY}"

GEN=training_mitigations/slurm_scripts/run_gen_heldout1000_v2.sbatch
JUDGE=training_mitigations/slurm_scripts/run_judge_v5_neutral.sbatch
OEQ1000=training_mitigations/sycoharm_runs/heldout_oeq_final1000_ids.json
PAS1000=training_mitigations/sycoharm_runs/heldout_pas_final1000_ids.json

RUNS=training_mitigations/runs
DPO_PAS=$RUNS/dpo_lora_pas_qwen3-8b_n1000_neutral
GRPO_PAS=$RUNS/grpo_lora_pas_qwen3-8b_n1000_neutral
GRPO_OEQ=$RUNS/grpo_lora_oeq_qwen3-8b_n1000_neutral

# All gen jobs: FINAL adapter (GEN_ADAPTER=""), 5 shards, 2x a4000 on next5/next6.
submit_gen() {  # JOBNAME RUNDIR ENV IDS
    local name=$1 rundir=$2 env=$3 ids=$4
    GEN_RUNDIR="$rundir" GEN_ENV="$env" GEN_IDS="$ids" GEN_ADAPTER="" NUM_SHARDS=5 \
        sbatch --parsable --job-name="$name" \
        --array=0-4 --gres=gpu:a4000:2 --nodelist=next5,next6 \
        --export=ALL,GEN_RUNDIR,GEN_ENV,GEN_IDS,GEN_ADAPTER,NUM_SHARDS "$GEN"
}

J1=$(submit_gen gen_dpo_pas_neutral  "$DPO_PAS"  PAS "$PAS1000")
J2=$(submit_gen gen_grpo_pas_neutral "$GRPO_PAS" PAS "$PAS1000")
J3=$(submit_gen gen_grpo_oeq_neutral "$GRPO_OEQ" OEQ "$OEQ1000")

echo "gen arrays: dpo_pas=$J1  grpo_pas=$J2  grpo_oeq=$J3"

JUDGE_J=$(sbatch --parsable \
    --dependency=afterok:"$J1":"$J2":"$J3" \
    --export=ALL "$JUDGE")
echo "judge (afterok all gen): $JUDGE_J"
echo "submitted."
