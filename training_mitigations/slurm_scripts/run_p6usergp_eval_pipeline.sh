#!/bin/bash
# Full 1000-prompt held-out eval pipeline for the two P6 user_gp mixed DPO runs,
# over the SAME held-out id sets every other method uses
# (sycoharm_runs/heldout_{pas,oeq}_final1000_ids.json). Per run, in order:
#
#   1. gen    : sharded generation of the FINAL adapter's responses (--no-judge),
#               10 shards over the 1000 held-out prompts (run_gen_heldout1000_v2)
#   2. endorse: stage the 10 shards + v5 gpt-5-mini action-endorsement judge
#               (run_judge_v5_heldout1000_generic)  [afterok: gen]
#   3. quality: 4 response-quality judges (addressee + 3 empathy)
#               (run_quality_judges)                [afterok: endorse]
#   4. elephant: ELEPHANT validation / indirectness / framing scorers
#               (run_elephant_scorers)              [afterok: endorse]
#
# PAS run is already trained -> its gen starts immediately.
# OEQ run's gen is gated afterok on its still-running DPO training job so the
# whole OEQ chain queues now and fires the moment training finishes.
#
# Run from repo root with HF_TOKEN and OPENROUTER_API_KEY exported.
set -euo pipefail
cd /next/u/stephhk/sycophancy

: "${HF_TOKEN:?export HF_TOKEN}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY}"

SS=training_mitigations/slurm_scripts
GEN=$SS/run_gen_heldout1000_v2.sbatch
ENDORSE=$SS/run_judge_v5_heldout1000_generic.sbatch
QUALITY=$SS/run_quality_judges.sbatch
ELEPHANT=$SS/run_elephant_scorers.sbatch

PAS1000=training_mitigations/sycoharm_runs/heldout_pas_final1000_ids.json
OEQ1000=training_mitigations/sycoharm_runs/heldout_oeq_final1000_ids.json

# OEQ DPO training job to gate the OEQ generation on (afterok).
TRAIN_OEQ=16059429
NUM_SHARDS=10

# submit_pipeline: NAMETAG RUNDIR ENV IDS COND [DEP_TRAIN_JOBID]
# RUNDIR is relative to repo root; COND is the registry key added to
# CONDITIONS_BY_ENV for the quality/elephant judges.
submit_pipeline() {
    local tag=$1 rundir=$2 env=$3 ids=$4 cond=$5 dep=${6:-}
    local name="${env,,}_${cond}_final1000"   # stage/dataset stem
    local jrun="${rundir#training_mitigations/}"

    local gdep=()
    [[ -n "$dep" ]] && gdep=(--dependency=afterok:"$dep")
    local GEN_J
    GEN_J=$(GEN_RUNDIR="$rundir" GEN_ENV="$env" GEN_IDS="$ids" NUM_SHARDS="$NUM_SHARDS" \
        sbatch --parsable --job-name="gen_${tag}" \
        --array=0-$((NUM_SHARDS-1)) "${gdep[@]}" \
        --export=ALL,GEN_RUNDIR="$rundir",GEN_ENV="$env",GEN_IDS="$ids",GEN_ADAPTER=,NUM_SHARDS="$NUM_SHARDS" \
        "$GEN")

    local END_J
    END_J=$(JRUN="$jrun" JNAME="$name" JSHARDS="$NUM_SHARDS" \
        sbatch --parsable --job-name="endorse_${tag}" \
        --dependency=afterok:"$GEN_J" \
        --export=ALL,JRUN="$jrun",JNAME="$name",JSHARDS="$NUM_SHARDS" \
        "$ENDORSE")

    local QJ_J
    QJ_J=$(QJ_ENV="$env" QJ_CONDS="$cond" \
        sbatch --parsable --job-name="qjudge_${tag}" \
        --dependency=afterok:"$END_J" \
        --export=ALL,QJ_ENV="$env",QJ_CONDS="$cond" \
        "$QUALITY")

    local ELE_J
    ELE_J=$(JUDGE_ENV="$env" ELE_CONDS="$cond" \
        sbatch --parsable --job-name="elephant_${tag}" \
        --dependency=afterok:"$END_J" \
        --export=ALL,JUDGE_ENV="$env",ELE_CONDS="$cond" \
        "$ELEPHANT")

    echo "[$tag] gen=$GEN_J -> endorse=$END_J -> quality=$QJ_J , elephant=$ELE_J  (dataset=$name)"
}

echo "=== PAS P6_user_gp (starts now) ==="
submit_pipeline pas_P6usergp \
    training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_mixed_P6_user_gp \
    PAS "$PAS1000" mixed_P6_user_gp

echo "=== OEQ P6_user_gp (gated afterok:$TRAIN_OEQ) ==="
submit_pipeline oeq_P6usergp \
    training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_user_gp \
    OEQ "$OEQ1000" mixed_P6_user_gp "$TRAIN_OEQ"

echo "submitted."
