#!/bin/bash
# Arm/launch FINAL-RUN held-out GENERATION on Marlowe for every method in the
# paper table, over the frozen final_training_eval_data/<env>_test split.
# Submits marlowe_final_gen.sbatch once per (env, condition) with the right
# --export + --array. GATED + IDEMPOTENT — safe to re-run as training lands:
#   - a TRAINED condition whose run dir has no .complete is skipped (still training)
#   - a condition whose gen shard csv(s) already exist is skipped (already done)
#   - base (untrained) conditions submit immediately (base model always present)
#
# Usage (Marlowe login node):
#   bash training_mitigations/slurm_scripts/marlowe_final_gen_submit.sh [--dry-run]
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
SS="$REPO/training_mitigations/slurm_scripts"
GENJOB="$SS/marlowe_final_gen.sbatch"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# adapter_tag mirrors marlowe_final_gen.sbatch / the eval tag logic.
adapter_tag() { # kind steer
  if [[ "$1" == "trained" ]]; then echo final; return; fi
  case "$2" in
    none) echo base ;; becritical) echo base-becritical ;;
    explicit) echo base-explicit ;; sharma) echo base-sharma ;;
    dontbesyco) echo base-dontbesyco ;; perspshift) echo base-perspshiftassess ;;
    *) echo "UNKNOWN" ;;
  esac
}

# Job table: ENV  KIND  STEER  RUNDIR(relative to $RUNS_ROOT)
# base conditions share a per-env baseline container; trained point at their run.
OEQ_BASE=baseline_final_OEQ ; PAS_BASE=baseline_final_PAS
AITA_BASE=baseline_final_AITA ; FLIP_BASE=baseline_final_AITA-NTA-FLIP
JOBS=(
  # ---- OEQ (6 base + 4 trained) ----
  "OEQ base none        $OEQ_BASE"
  "OEQ base becritical  $OEQ_BASE"
  "OEQ base explicit    $OEQ_BASE"
  "OEQ base sharma      $OEQ_BASE"
  "OEQ base dontbesyco  $OEQ_BASE"
  "OEQ base perspshift  $OEQ_BASE"
  "OEQ trained none dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"
  "OEQ trained none dpo_lora_oeq_qwen3-8b_n1000_neutral_final"
  "OEQ trained none dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final"
  "OEQ trained none sft_lora_oeq_qwen3-8b_becritical_final"
  # ---- PAS (6 base + 4 trained) ----
  "PAS base none        $PAS_BASE"
  "PAS base becritical  $PAS_BASE"
  "PAS base explicit    $PAS_BASE"
  "PAS base sharma      $PAS_BASE"
  "PAS base dontbesyco  $PAS_BASE"
  "PAS base perspshift  $PAS_BASE"
  "PAS trained none dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"
  "PAS trained none dpo_lora_pas_qwen3-8b_n1000_neutral_final"
  "PAS trained none dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final"
  "PAS trained none sft_lora_pas_qwen3-8b_becritical_final"
  # ---- AITA (6 base + 1 trained) ----
  "AITA base none        $AITA_BASE"
  "AITA base becritical  $AITA_BASE"
  "AITA base explicit    $AITA_BASE"
  "AITA base sharma      $AITA_BASE"
  "AITA base dontbesyco  $AITA_BASE"
  "AITA base perspshift  $AITA_BASE"
  "AITA trained none dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"
  # ---- AITA-NTA-FLIP (6 base + 1 trained) ----
  "AITA-NTA-FLIP base none        $FLIP_BASE"
  "AITA-NTA-FLIP base becritical  $FLIP_BASE"
  "AITA-NTA-FLIP base explicit    $FLIP_BASE"
  "AITA-NTA-FLIP base sharma      $FLIP_BASE"
  "AITA-NTA-FLIP base dontbesyco  $FLIP_BASE"
  "AITA-NTA-FLIP base perspshift  $FLIP_BASE"
  "AITA-NTA-FLIP trained none dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final"
)

printf "%-14s %-8s %-11s %-58s %-9s %s\n" ENV KIND STEER RUN STATE ACTION
for row in "${JOBS[@]}"; do
  read -r ENV KIND STEER RUN <<< "$row"
  rundir="$RUNS_ROOT/$RUN"
  case "$ENV" in
    OEQ|PAS)       ARRAY=0-1; N=2; PREFIX=responses ;;
    AITA)          ARRAY=0;   N=1; PREFIX=aita_responses ;;
    AITA-NTA-FLIP) ARRAY=0-1; N=2; PREFIX=aita_responses ;;  # 2-sharded: heaviest gen (2000 gens @1536tok)
  esac
  TAG="$(adapter_tag "$KIND" "$STEER")_finaldata_temp1.0_seed0"

  # trained arms must have finished training first.
  if [[ "$KIND" == "trained" && ! -f "$rundir/.complete" ]]; then
    printf "%-14s %-8s %-11s %-58s %-9s %s\n" "$ENV" "$KIND" "$STEER" "$RUN" "training" "wait (.complete absent)"
    continue
  fi
  # already generated? (all shards present)
  done=1
  for ((s=0; s<N; s++)); do
    [[ -f "$rundir/eval/${PREFIX}_${TAG}_shard${s}of${N}.csv" ]] || done=0
  done
  if (( done )); then
    printf "%-14s %-8s %-11s %-58s %-9s %s\n" "$ENV" "$KIND" "$STEER" "$RUN" "done" "skip (already generated)"
    continue
  fi
  if (( DRY )); then
    printf "%-14s %-8s %-11s %-58s %-9s %s\n" "$ENV" "$KIND" "$STEER" "$RUN" "ready" "would submit (array $ARRAY)"
  else
    jid=$(sbatch --parsable --array="$ARRAY" \
        --export=ALL,GEN_ENV="$ENV",GEN_RUNDIR="$rundir",GEN_KIND="$KIND",GEN_STEER="$STEER",GEN_NUM_SHARDS="$N" \
        "$GENJOB")
    printf "%-14s %-8s %-11s %-58s %-9s %s\n" "$ENV" "$KIND" "$STEER" "$RUN" "ready" "submitted $jid (array $ARRAY)"
  fi
done
