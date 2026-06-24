#!/bin/bash
# Submit the full 1000-prompt held-out eval pipeline for the 4 finished
# post-training runs: per-run sharded generation (FINAL adapter, --no-judge),
# then one v5 GPT-5-mini endorse-judge job gated afterok on all generations.
#
#   PAS msp      : already finished training -> gen runs immediately
#   OEQ mixed_P4 : afterok on training job 15923916
#   OEQ msp_P4   : afterok on training job 15923917
#   OEQ msp      : afterok on training job 15923918
#
# Run from repo root with HF_TOKEN and OPENROUTER_API_KEY exported.
set -euo pipefail
cd /next/u/stephhk/sycophancy

: "${HF_TOKEN:?export HF_TOKEN}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY}"

GEN=training_mitigations/slurm_scripts/run_gen_heldout1000.sbatch
JUDGE=training_mitigations/slurm_scripts/run_judge_v5_heldout1000.sbatch
OEQ1000=training_mitigations/sycoharm_runs/heldout_oeq_final1000_ids.json
PAS1000=training_mitigations/sycoharm_runs/heldout_pas_final1000_ids.json

# Training job ids the OEQ generations depend on (afterok).
TRAIN_OEQ_MIXED_P4=15923916
TRAIN_OEQ_MSP_P4=15923917
TRAIN_OEQ_MSP=15923918

submit_gen() {  # JOBNAME RUNDIR ENV IDS [DEP_JOBID]
    local name=$1 rundir=$2 env=$3 ids=$4 dep=${5:-}
    local depflag=()
    [[ -n "$dep" ]] && depflag=(--dependency=afterok:"$dep")
    GEN_RUNDIR="$rundir" GEN_ENV="$env" GEN_IDS="$ids" \
        sbatch --parsable --job-name="$name" "${depflag[@]}" \
        --export=ALL,GEN_RUNDIR,GEN_ENV,GEN_IDS "$GEN"
}

PAS_J=$(submit_gen gen_pas_msp \
    training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_mixed_selfpref PAS "$PAS1000")
OEQ_MIX_J=$(submit_gen gen_oeq_mixed_P4 \
    training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P4 OEQ "$OEQ1000" "$TRAIN_OEQ_MIXED_P4")
OEQ_MSPP4_J=$(submit_gen gen_oeq_msp_P4 \
    training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_selfpref_P4 OEQ "$OEQ1000" "$TRAIN_OEQ_MSP_P4")
OEQ_MSP_J=$(submit_gen gen_oeq_msp \
    training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_selfpref OEQ "$OEQ1000" "$TRAIN_OEQ_MSP")

echo "gen jobs: PAS=$PAS_J OEQ_mixed_P4=$OEQ_MIX_J OEQ_msp_P4=$OEQ_MSPP4_J OEQ_msp=$OEQ_MSP_J"

JUDGE_J=$(sbatch --parsable \
    --dependency=afterok:"$PAS_J":"$OEQ_MIX_J":"$OEQ_MSPP4_J":"$OEQ_MSP_J" \
    --export=ALL "$JUDGE")
echo "judge job (afterok all gen): $JUDGE_J"
echo "submitted."
