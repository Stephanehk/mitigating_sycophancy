#!/bin/bash
# Launch the 1000-prompt held-out eval for the 3 runs requested:
#   OEQ mixed_P4            @ adapter_iter_2  -> next5, 5 shards x (2x a4000)
#   OEQ mixed_selfpref_P4   @ adapter_iter_2  -> next6, 5 shards x (2x a4000)
#   PAS mixed_selfpref      @ final adapter   -> next7, 4 shards x (1x a6000)
# then one v5 gpt-5-mini judge job gated afterok on all three gen arrays.
#
# Replaces the stalled pipeline (gen jobs that afterok-depended on FAILED
# training jobs, and 500-prompt/shard gen that TIMED OUT at 4h). Here every gen
# job depends on nothing (adapters are on disk), shards are 200-250 prompts, and
# all shards of all three models run concurrently on idle GPUs (no waves).
#
# Run from repo root with HF_TOKEN and OPENROUTER_API_KEY exported.
set -euo pipefail
cd /next/u/stephhk/sycophancy

: "${HF_TOKEN:?export HF_TOKEN}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY}"

GEN=training_mitigations/slurm_scripts/run_gen_heldout1000_v2.sbatch
JUDGE=training_mitigations/slurm_scripts/run_judge_v5_eval3.sbatch
OEQ1000=training_mitigations/sycoharm_runs/heldout_oeq_final1000_ids.json
PAS1000=training_mitigations/sycoharm_runs/heldout_pas_final1000_ids.json

RUNS=training_mitigations/runs
OEQ_MIX=$RUNS/dpo_lora_oeq_qwen3-8b_n1000_mixed_P4
OEQ_MSP=$RUNS/dpo_lora_oeq_qwen3-8b_n1000_mixed_selfpref_P4
PAS_MSP=$RUNS/dpo_lora_pas_qwen3-8b_n1000_mixed_selfpref

# --- OEQ mixed_P4 @ adapter_iter_2 : next5, 5 shards, 2x a4000 each -----------
J1=$(GEN_RUNDIR="$OEQ_MIX" GEN_ENV=OEQ GEN_IDS="$OEQ1000" \
     GEN_ADAPTER="$OEQ_MIX/adapter_iter_2" NUM_SHARDS=5 \
     sbatch --parsable --job-name=gen_oeq_mixed_P4_i2 \
        --array=0-4 --gres=gpu:a4000:2 --nodelist=next5 \
        --export=ALL,GEN_RUNDIR,GEN_ENV,GEN_IDS,GEN_ADAPTER,NUM_SHARDS "$GEN")

# --- OEQ mixed_selfpref_P4 @ adapter_iter_2 : next6, 5 shards, 2x a4000 -------
J2=$(GEN_RUNDIR="$OEQ_MSP" GEN_ENV=OEQ GEN_IDS="$OEQ1000" \
     GEN_ADAPTER="$OEQ_MSP/adapter_iter_2" NUM_SHARDS=5 \
     sbatch --parsable --job-name=gen_oeq_msp_P4_i2 \
        --array=0-4 --gres=gpu:a4000:2 --nodelist=next6 \
        --export=ALL,GEN_RUNDIR,GEN_ENV,GEN_IDS,GEN_ADAPTER,NUM_SHARDS "$GEN")

# --- PAS mixed_selfpref @ FINAL adapter : next7, 4 shards, 1x a6000 each ------
J3=$(GEN_RUNDIR="$PAS_MSP" GEN_ENV=PAS GEN_IDS="$PAS1000" \
     GEN_ADAPTER="" NUM_SHARDS=4 \
     sbatch --parsable --job-name=gen_pas_msp_final \
        --array=0-3 --gres=gpu:a6000:1 --nodelist=next7 \
        --export=ALL,GEN_RUNDIR,GEN_ENV,GEN_IDS,GEN_ADAPTER,NUM_SHARDS "$GEN")

echo "gen arrays: OEQ_mixed_P4=$J1  OEQ_msp_P4=$J2  PAS_msp=$J3"

JUDGE_J=$(sbatch --parsable \
    --dependency=afterok:"$J1":"$J2":"$J3" \
    --export=ALL "$JUDGE")
echo "judge (afterok all gen): $JUDGE_J"
echo "submitted."
