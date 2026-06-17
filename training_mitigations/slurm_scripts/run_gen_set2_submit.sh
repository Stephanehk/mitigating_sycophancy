#!/bin/bash
# Submit the set2 / final1000 generation-only jobs. PAS base was smoke-tested
# separately. OEQ selfpref is skipped (no saved adapter). Run from repo root.
set -euo pipefail
SB=training_mitigations/slurm_scripts/run_gen_set2.sbatch
PAS_IDS=training_mitigations/sycoharm_runs/heldout_set2_pas_ids.json
OEQ_IDS=training_mitigations/sycoharm_runs/heldout_set2_oeq_ids.json
OEQ1000=training_mitigations/sycoharm_runs/heldout_oeq_final1000_ids.json
PASRUN=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000
OEQRUN=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000

sub() {  # RUNDIR IDS TAGX BASE BECRIT SUFFIX TAG  label
  GEN_RUNDIR="$1" GEN_IDS="$2" GEN_TAGX="$3" GEN_BASE="$4" GEN_BECRIT="$5" GEN_SUFFIX="$6" GEN_TAG="$7" \
    sbatch --job-name="gen_$8" \
    --export=ALL,GEN_RUNDIR,GEN_IDS,GEN_TAGX,GEN_BASE,GEN_BECRIT,GEN_SUFFIX,GEN_TAG "$SB"
}

# --- PAS new-500 (base already smoke-submitted) ---
sub "$PASRUN" "$PAS_IDS" set2 1 1 ""                      ""          pas_becrit
sub "$PASRUN" "$PAS_IDS" set2 1 0 "Don't be sycophantic." dontbesyco  pas_dbs
sub "$PASRUN" "$PAS_IDS" set2 0 0 ""                      ""          pas_trained
sub "${PASRUN}_mixed"    "$PAS_IDS" set2 0 0 "" "" pas_mixed
sub "${PASRUN}_selfpref" "$PAS_IDS" set2 0 0 "" "" pas_selfpref

# --- OEQ new-500 (selfpref skipped: no adapter) ---
sub "$OEQRUN" "$OEQ_IDS" set2 1 0 ""                      ""          oeq_base
sub "$OEQRUN" "$OEQ_IDS" set2 1 1 ""                      ""          oeq_becrit
sub "$OEQRUN" "$OEQ_IDS" set2 1 0 "Don't be sycophantic." dontbesyco  oeq_dbs
sub "$OEQRUN" "$OEQ_IDS" set2 0 0 ""                      ""          oeq_trained

# --- OEQ FINAL model (n1000_mixed) over combined 1000 ---
sub "${OEQRUN}_mixed"    "$OEQ1000" final1000 0 0 "" "" oeq_final1000

echo "submitted."
