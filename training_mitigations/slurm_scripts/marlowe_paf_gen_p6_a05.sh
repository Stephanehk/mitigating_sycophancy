#!/bin/bash
# Durable held-out EVAL driver for the P6_perspshift alpha=0.5 arms on PAS, AITA,
# and AITA-NTA-FLIP (the "AITA-FLIPPED" env). For each arm: wait for training to
# finish (.complete), then run its held-out generation with the same generic gen
# sbatch the OEQ/AITA pipelines use, dispatched per env:
#
#   PAS            -> eval_endorsement_heldout.py --no-judge   (writes responses_*;
#                     the gpt-5-mini endorse judge scores it later -- SEPARATE step,
#                     exactly like OEQ. This driver stops after generation for PAS.)
#   AITA           -> eval_aita_label_heldout.py (self-parses the YTA/NTA verdict;
#   AITA-NTA-FLIP     writes aita_scored_* -- this IS the eval, no judge needed).
#
# ID SETS (match the established convention per env):
#   PAS  -> fixed heldout_pas_final1000_ids.json  (parallel to OEQ's final1000 set)
#   AITA -> the arm's OWN run-dir test_ids.json    (what marlowe_run_eval_aita_trained
#   FLIP    .sbatch uses; cross-arm comparability caveat noted there).
#
# Keys off $OUT_DIR/.complete, NOT the training job id (the dpo body queues an
# afterany successor up front -- see marlowe_oeq_eval_pipeline.sh for rationale).
# Survives session exit -- launch detached:
#   source ~/.bashrc                       # HF_TOKEN
#   setsid nohup bash training_mitigations/slurm_scripts/marlowe_paf_gen_p6_a05.sh \
#       > /projects/m000254/logs/paf_gen_p6_a05.log 2>&1 &
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
GEN=training_mitigations/slurm_scripts/marlowe_run_gen_heldout1000.sbatch
PAS1000=$REPO/training_mitigations/sycoharm_runs/heldout_pas_final1000_ids.json
cd "$REPO"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

: "${HF_TOKEN:?export HF_TOKEN (Qwen3-8B is gated)}"
[[ -f "$PAS1000" ]] || { log "FATAL: missing $PAS1000"; exit 1; }

# arm | GEN_ENV | id-source (OWN = the arm's run-dir test_ids.json)
TUPLES=(
  "dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5|PAS|$PAS1000"
  "dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5|AITA|OWN"
  "dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5|AITA-NTA-FLIP|OWN"
)

# done-marker basename differs by env. NOTE (bare-post refactor, 2026-07-07):
# eval_aita_label_heldout.py is GENERATION-ONLY and writes aita_responses_* (NOT
# aita_scored_*, which the pre-refactor self-parser wrote and which the shared gen
# sbatch's idempotency check still stale-references). The YTA/NTA verdict + metrics
# come from a downstream classify_aita_verdicts.py pass -- so aita/flip are gen->judge
# too, not self-scoring.
done_prefix(){ case "$1" in PAS|OEQ) echo responses;; *) echo aita_responses;; esac; }
gen_done(){ # arm env
  local pre; pre=$(done_prefix "$2")
  [[ -f "$RUNS_ROOT/$1/eval/${pre}_final_final1000_temp1.0_seed0_shard0of2.csv" \
   && -f "$RUNS_ROOT/$1/eval/${pre}_final_final1000_temp1.0_seed0_shard1of2.csv" ]]
}

# ---------------- phase 0: adopt any in-flight generation from a prior run ----------------
if squeue -u "$USER" -h -o "%j" | grep -q '^genpaf_'; then
  log "phase 0: in-flight genpaf_* from a previous driver — waiting them out"
  while squeue -u "$USER" -h -o "%j" | grep -q '^genpaf_'; do sleep 120; done
  log "phase 0: previous generations finished"
fi

# ---------------- phase 1: wait for training, submit gen per arm ----------------
declare -A GENJOB GENENV
pending=("${TUPLES[@]}")
dead=0
log "waiting for ${#pending[@]} adapters to train (.complete)"
while (( ${#pending[@]} )); do
  still=()
  for t in "${pending[@]}"; do
    IFS='|' read -r arm env idsrc <<<"$t"
    if [[ -f "$RUNS_ROOT/$arm/.complete" ]]; then
      ids="$idsrc"; [[ "$idsrc" == OWN ]] && ids="$RUNS_ROOT/$arm/test_ids.json"
      [[ -f "$ids" ]] || { log "ABORT: ids file missing for $arm: $ids"; exit 1; }
      # leakage guard: held-out ids must not appear in this arm's train split
      "$REPO/../envs/llm126/bin/python" - "$RUNS_ROOT/$arm" "$ids" <<'PY'
import json, sys
run, ids = sys.argv[1], sys.argv[2]
tr = set(map(str, json.load(open(f"{run}/train_ids.json"))))
hi = set(map(str, json.load(open(ids))))
n = len(tr & hi)
print(f"[leak] {run.split('/')[-1]}: heldout∩train = {n}")
sys.exit(1 if n else 0)
PY
      if (( $? != 0 )); then log "ABORT: leakage, heldout ids overlap $arm train_ids"; exit 1; fi
      if gen_done "$arm" "$env"; then
        log "trained: $arm -> gen already complete (both shards), skipping submit"
      else
        log "trained: $arm ($env) -> submitting gen  ids=$ids"
        jid=$(GEN_RUNDIR="$RUNS_ROOT/$arm" GEN_ENV="$env" GEN_IDS="$ids" \
              sbatch --parsable --job-name="genpaf_${arm: -18}" \
              --export=ALL,GEN_RUNDIR,GEN_ENV,GEN_IDS "$GEN")
        GENJOB[$arm]=$jid; GENENV[$arm]=$env
        log "  gen array = $jid"
      fi
    else
      still+=("$t")
    fi
  done
  pending=("${still[@]+"${still[@]}"}")
  (( ${#pending[@]} == 0 )) && break
  # abort if the whole dpo chain for these envs has vanished without finishing
  if squeue -u "$USER" -h -o "%j" | grep -qE '^dpo_(pas|aita|flip)_'; then dead=0; else dead=$((dead+1)); fi
  (( dead >= 3 )) && { log "ABORT: no dpo_(pas|aita|flip)_ jobs queued/running, ${#pending[@]} arms unfinished"; exit 1; }
  sleep 300
done

# ---------------- phase 2: wait for all generations ----------------
log "all adapters trained; waiting on ${#GENJOB[@]} gen arrays"
for arm in "${!GENJOB[@]}"; do
  jid=${GENJOB[$arm]}
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 120; done
  log "gen done: $arm ($jid)"
done

# ---------------- phase 3: verify shards ----------------
missing=0
for t in "${TUPLES[@]}"; do
  IFS='|' read -r arm env idsrc <<<"$t"
  pre=$(done_prefix "$env")
  for s in 0 1; do
    f="$RUNS_ROOT/$arm/eval/${pre}_final_final1000_temp1.0_seed0_shard${s}of2.csv"
    [[ -f "$f" ]] || { log "MISSING shard: $f"; missing=1; }
  done
done
(( missing )) && { log "ABORT: generation incomplete."; exit 1; }

log "GEN COMPLETE (generation stage only -- all three envs still need a judge/classify pass):"
log "  AITA + FLIP: aita_responses_* written; run classify_aita_verdicts.py (via marlowe_classify_all.sh)"
log "       -> summary_aita_<tag>_CLASSIFIED_AGG.json is the actual held-out metric."
log "  PAS: responses_* written; run the gpt-5-mini v5 endorse judge"
log "       (stage_heldout_inputs.py -> run_endorse_gpt5mini.py --prompt-version v5)."
