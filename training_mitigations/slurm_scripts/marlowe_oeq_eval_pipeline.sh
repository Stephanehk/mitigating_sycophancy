#!/bin/bash
# Durable driver for the OEQ rpo-alpha sweep: wait for each adapter to finish
# training, generate its responses over the 1000-prompt held-out set, then score
# everything with the GPT-5-mini action-endorsement judge (the default eval for
# OEQ/PAS). Survives session exit -- launch detached:
#
#   source ~/.bashrc                       # HF_TOKEN + OPENROUTER_API_KEY
#   setsid nohup bash training_mitigations/slurm_scripts/marlowe_oeq_eval_pipeline.sh \
#       > /projects/m000254/logs/oeq_pipeline.log 2>&1 &
#
# WHY POLL .complete INSTEAD OF sbatch --dependency=afterok:
# training runs as a SELF-RESUBMITTING preempt chain (_marlowe_dpo_body.sh queues
# an afterany successor each 4h window). The job id we submit is only the FIRST
# link, and it terminates non-zero on preemption -- an afterok on it would never
# release, and an afterany would fire while the chain is still running. The chain
# writes $OUT_DIR/.complete exactly once, after the final adapter is saved. That
# marker is the only reliable completion signal. (eval_pipeline.sh does the same.)
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
GEN=training_mitigations/slurm_scripts/marlowe_run_gen_heldout1000.sbatch
JUDGE=training_mitigations/slurm_scripts/marlowe_endorse_judge_all.sh
OEQ1000=$REPO/training_mitigations/sycoharm_runs/heldout_oeq_final1000_ids.json
cd "$REPO"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# ---------------- preflight: fail loud, not silently ----------------
: "${HF_TOKEN:?export HF_TOKEN (Qwen3-8B is gated)}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY (needed by the endorse judge)}"
[[ -f "$OEQ1000" ]] || { log "FATAL: missing $OEQ1000"; exit 1; }
"$REPO/../envs/llm126/bin/python" - <<'PY' || exit 1
import sys; sys.path.insert(0, 'training_mitigations/eval')
from eval_endorsement_heldout import resolve_csv
try:
    print("[preflight] OEQ csv:", resolve_csv("OEQ"))
except AssertionError as e:
    print("FATAL:", e); print("Restore the OEQ csv before running this pipeline."); sys.exit(1)
PY

# Every arm the pipeline generates + judges. The 7 SWEEP arms are being trained now;
# the 3 REF arms (alpha=1.0) finished training 2026-07-03..05 and already carry
# .complete, so phase 1 submits their generation on the first poll -- they are the
# top of each dose-response curve and had NO eval until now (the OEQ csv was missing).
declare -A ARMS=(
  # --- sweep (training now) ---
  [dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_perspshift_gp]=sweep
  [dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_gp]=sweep
  [dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p25]=sweep
  [dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5]=sweep
  # NOTE: P5_nocrit_gp_frozenstake (alpha=0) is deliberately NOT run. Its wrapper
  # exists (marlowe_run_train_oeq_n1000_mixed_P5_nocrit_gp_frozenstake.sbatch) but
  # the arm was cancelled; the frozenstake curve is therefore alpha in {0.25,0.5,1.0}
  # with no zero, so the distill hypothesis is slope-measurable but not falsifiable
  # for that recipe. If you re-add it, add it to marlowe_endorse_judge_all.sh's
  # OEQ_ALPHA_SWEEP and summarize_oeq_sweep.py's ARMS too -- all three must agree.
  [dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25]=sweep
  [dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p5]=sweep
  # --- alpha=1.0 references (already trained) ---
  [dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp]=ref
  [dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake]=ref
  [dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_perspshift_distill_gp]=ref
)

SHARD0='eval/responses_final_final1000_temp1.0_seed0_shard0of2.csv'
SHARD1='eval/responses_final_final1000_temp1.0_seed0_shard1of2.csv'
gen_done(){ [[ -f "$RUNS_ROOT/$1/$SHARD0" && -f "$RUNS_ROOT/$1/$SHARD1" ]]; }

# ---------------- phase 0: adopt any in-flight generation ----------------
# RESTART SAFETY. This driver may be relaunched after being killed (e.g. the arm
# list changed). A previous instance may have left genoeq_* arrays RUNNING; those
# jobs are writing into <run>/eval/. Resubmitting gen for the same run while they
# are in flight would have two jobs writing the same csv. Wait them out, then let
# phase 1's gen_done() guard skip whatever they finished.
if squeue -u "$USER" -h -o "%j" | grep -q '^genoeq_'; then
  log "phase 0: in-flight genoeq_* from a previous driver — waiting them out"
  while squeue -u "$USER" -h -o "%j" | grep -q '^genoeq_'; do sleep 120; done
  log "phase 0: previous generations finished"
fi

# ---------------- phase 1: wait for training, submit gen per arm ----------------
declare -A GENJOB
pending=("${!ARMS[@]}")
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

# ---------------- phase 3: verify shards, then judge ----------------
# The gen job is `set -euo pipefail` so a driver failure FAILS the job, but verify
# the done-marker csvs anyway: a COMPLETED state has silently masked zero output
# before (the --no-judge argparse bug).
missing=0
for arm in "${!ARMS[@]}"; do
  for s in 0 1; do
    f="$RUNS_ROOT/$arm/eval/responses_final_final1000_temp1.0_seed0_shard${s}of2.csv"
    [[ -f "$f" ]] || { log "MISSING shard: $f"; missing=1; }
  done
done
(( missing )) && { log "ABORT: generation incomplete; not judging."; exit 1; }

log "all shards present -> gpt-5-mini action-endorsement judge (v5)"
bash "$JUDGE" oeq_all      # 7 sweep arms + the 3 alpha=1.0 references
log "endorse judge complete -> sycoharm_runs/endorse_gpt5mini_v5/"

# ---------------- phase 4: voice census ----------------
# WHY THIS IS NOT OPTIONAL: the v5 endorse prompt scopes "challenge" to the USER's
# OWN actions (see run_endorse_gpt5mini.py:175 -- v5 exists because Qwen aimed its
# criticism at the other party and v4 mislabeled that ~30% of the time). A response
# that ROLE-PLAYS a stakeholder and berates the user therefore reads as a genuine
# challenge (label 0) and DRIVES THE ENDORSEMENT RATE DOWN. That is indistinguishable
# from a real sycophancy reduction unless you also know how often the policy stopped
# answering as an assistant. On AITA/PAS the perspshift arms role-play 30.7% / 37.8%
# of the time (classify_response_voice.py). Read the two columns together or not at all.
PY_BIN=/projects/$PROJECT/envs/llm126/bin/python
log "voice census over the same 6 response sets"
for arm in "${!ARMS[@]}"; do
  "$PY_BIN" training_mitigations/eval/classify_response_voice.py \
      --run-dir "$RUNS_ROOT/$arm" --env OEQ \
      --tag final_final1000_temp1.0_seed0 --num-shards 2 \
    || { log "WARN: voice census failed for $arm (endorse results still valid)"; continue; }
done
log "voice census complete -> <run>/eval/voice_final_final1000_temp1.0_seed0.csv"

# ---------------- phase 5: joint summary ----------------
# Endorsement rate and role-play rate side by side, so neither is read alone.
# The arm -> judge-name map MUST match marlowe_endorse_judge_all.sh's OEQ_ALPHA_SWEEP.
"$PY_BIN" training_mitigations/eval/summarize_oeq_sweep.py || log "WARN: summary failed"
log "PIPELINE COMPLETE -> endorse: sycoharm_runs/endorse_gpt5mini_v5/  voice: <run>/eval/voice_*.csv"
