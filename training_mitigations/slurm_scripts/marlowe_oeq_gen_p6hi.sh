#!/bin/bash
# Durable GEN driver for the P6_perspshift rpo-alpha HIGH arms (alpha 0.6/0.7/0.8),
# extending the {0,0.25,0.5} sweep. For each of the 3 adapters: wait for training to
# finish (.complete), then generate its responses over the same 1000-prompt held-out
# set, exactly as marlowe_oeq_eval_pipeline.sh does (same GEN sbatch, same ids, same
# tag). Stops after generation -- the gpt-5-mini endorse judge + voice census are a
# SEPARATE step (their arm lists in marlowe_endorse_judge_all.sh / summarize_oeq_sweep.py
# would need these 3 arms added first), so this driver deliberately does not run them.
#
# Survives session exit -- launch detached:
#   source ~/.bashrc                       # HF_TOKEN + OPENROUTER_API_KEY
#   setsid nohup bash training_mitigations/slurm_scripts/marlowe_oeq_gen_p6hi.sh \
#       > /projects/m000254/logs/oeq_gen_p6hi.log 2>&1 &
#
# Keys off $OUT_DIR/.complete (written once, after the final adapter is saved), NOT
# the training job id, because _marlowe_dpo_body.sh queues an afterany successor up
# front -- see marlowe_oeq_eval_pipeline.sh for the full rationale.
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
GEN=training_mitigations/slurm_scripts/marlowe_run_gen_heldout1000.sbatch
OEQ1000=$REPO/training_mitigations/sycoharm_runs/heldout_oeq_final1000_ids.json
cd "$REPO"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# ---------------- preflight: fail loud, not silently ----------------
: "${HF_TOKEN:?export HF_TOKEN (Qwen3-8B is gated)}"
[[ -f "$OEQ1000" ]] || { log "FATAL: missing $OEQ1000"; exit 1; }
"$REPO/../envs/llm126/bin/python" - <<'PY' || exit 1
import sys; sys.path.insert(0, 'training_mitigations/eval')
from eval_endorsement_heldout import resolve_csv
try:
    print("[preflight] OEQ csv:", resolve_csv("OEQ"))
except AssertionError as e:
    print("FATAL:", e); print("Restore the OEQ csv before running this pipeline."); sys.exit(1)
PY

# The 3 high-alpha arms this driver covers, and nothing else.
ARMS=(
  dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p6
  dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p7
  dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p8
)

SHARD0='eval/responses_final_final1000_temp1.0_seed0_shard0of2.csv'
SHARD1='eval/responses_final_final1000_temp1.0_seed0_shard1of2.csv'
gen_done(){ [[ -f "$RUNS_ROOT/$1/$SHARD0" && -f "$RUNS_ROOT/$1/$SHARD1" ]]; }

# ---------------- phase 0: adopt any in-flight generation ----------------
if squeue -u "$USER" -h -o "%j" | grep -q '^genoeq_'; then
  log "phase 0: in-flight genoeq_* from a previous driver — waiting them out"
  while squeue -u "$USER" -h -o "%j" | grep -q '^genoeq_'; do sleep 120; done
  log "phase 0: previous generations finished"
fi

# ---------------- phase 1: wait for training, submit gen per arm ----------------
declare -A GENJOB
pending=("${ARMS[@]}")
dead=0
log "waiting for ${#pending[@]} adapters to train (.complete)"
while (( ${#pending[@]} )); do
  still=()
  for arm in "${pending[@]}"; do
    if [[ -f "$RUNS_ROOT/$arm/.complete" ]]; then
      # leakage guard: the held-out ids must not appear in this arm's train split
      "$REPO/../envs/llm126/bin/python" - "$RUNS_ROOT/$arm" "$OEQ1000" <<'PY'
import json, sys
run, ids = sys.argv[1], sys.argv[2]
tr = set(map(str, json.load(open(f"{run}/train_ids.json"))))
hi = set(map(str, json.load(open(ids))))
n = len(tr & hi)
print(f"[leak] {run.split('/')[-1]}: heldout∩train = {n}")
sys.exit(1 if n else 0)
PY
      if (( $? != 0 )); then log "ABORT: leakage, heldout ids overlap $arm train_ids"; exit 1; fi
      if gen_done "$arm"; then
        log "trained: $arm -> gen already complete (both shards present), skipping submit"
      else
        log "trained: $arm -> submitting gen"
        jid=$(GEN_RUNDIR="$RUNS_ROOT/$arm" GEN_ENV=OEQ GEN_IDS="$OEQ1000" \
              sbatch --parsable --job-name="genoeq_${arm: -18}" \
              --export=ALL,GEN_RUNDIR,GEN_ENV,GEN_IDS "$GEN")
        GENJOB[$arm]=$jid
        log "  gen array = $jid"
      fi
    else
      still+=("$arm")
    fi
  done
  pending=("${still[@]+"${still[@]}"}")
  (( ${#pending[@]} == 0 )) && break
  # abort if the whole dpo_oeq chain has vanished without finishing
  if squeue -u "$USER" -h -o "%j" | grep -q '^dpo_oeq_'; then dead=0; else dead=$((dead+1)); fi
  (( dead >= 3 )) && { log "ABORT: no dpo_oeq_ jobs queued/running, ${#pending[@]} arms unfinished"; exit 1; }
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
for arm in "${ARMS[@]}"; do
  for s in 0 1; do
    f="$RUNS_ROOT/$arm/eval/responses_final_final1000_temp1.0_seed0_shard${s}of2.csv"
    [[ -f "$f" ]] || { log "MISSING shard: $f"; missing=1; }
  done
done
(( missing )) && { log "ABORT: generation incomplete."; exit 1; }
log "GEN COMPLETE -> responses under runs/<arm>/eval/responses_final_final1000_temp1.0_seed0_shard{0,1}of2.{jsonl,csv}"
log "NEXT (separate): add these 3 arms to marlowe_endorse_judge_all.sh + summarize_oeq_sweep.py, then run the endorse judge + voice census."
