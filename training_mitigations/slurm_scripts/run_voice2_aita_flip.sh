#!/usr/bin/env bash
# Driver for the "voice2" arm -- Our Model (3+1) -- on AITA and AITA-NTA-FLIP.
#
#   sample -> classify -> build pairs -> [HALT: report yield] -> train
#
# This is run_voice2_all.sh's pipeline for the two envs it does not cover. Three
# things differ from that script and are the reason this is a separate file
# rather than a parameter:
#
#   1. RUN-DIR STEM. run_voice2_all.sh derives paths as "${env,,}" over a
#      hardcoded n1000_mixed. Neither holds here: AITA-NTA-FLIP lowercases to
#      "aita-nta-flip" but the run-dir convention is "flip", and the FLIP P6 run
#      is n500_mixed (500 train prompts) while AITA is n1000_mixed. So the stem
#      and the P6 path are looked up per env instead of derived.
#
#   2. NO VALIDATION GATE. run_voice2_all.sh refuses to train until the v2 voice
#      rubric demonstrably separates collective voice. That gate validates the
#      RUBRIC, not the env, it was run on PAS on 2026-07-21 and PASSED
#      (collective x3.8, singular x6.0, rate 19.2%), and it reads a PAS run dir.
#      Re-running it here would re-measure the same rubric property, so it is
#      deliberately skipped -- the rubric is the same file.
#
#   3. THE YIELD HALT. Pairs only exist for prompts where the policy actually
#      collapsed: OEQ yielded 448 pairs from 256/1000 contributing prompts, PAS
#      2016 from 684/1000. AITA/FLIP responses are verdict-style and may collapse
#      far more rarely, and FLIP has only 500 train prompts to draw from. One DPO
#      epoch over a few dozen pairs is noise, so this script STOPS after
#      build_dpo_pairs and reports the yield rather than spending GPU time to
#      train on a pool that cannot support the claim. Re-invoke with TRAIN=1
#      once the yield has been reviewed.
#
# Stage idempotency matches run_voice2_all.sh: sample/classify resume from
# partial output, train skips on .complete.
#
#   bash run_voice2_aita_flip.sh          # sample -> classify -> pairs, then stop
#   TRAIN=1 bash run_voice2_aita_flip.sh  # same, and train (skips finished stages)
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
SS=$REPO/training_mitigations/slurm_scripts
PY=/projects/$PROJECT/envs/llm126/bin/python
RUNS=/projects/$PROJECT/runs
EXCLUDE="${EXCLUDE:-n06}"
ATTEMPTS=4
TRAIN="${TRAIN:-0}"
MIN_PAIRS="${MIN_PAIRS:-200}"

# The per-env stages are independent -- separate P6 adapters, separate out-dirs,
# separate candidate files -- so the two envs only run sequentially because this
# loop blocks on each. Set VOICE2_ENVS to run one env per process and get them
# concurrently (the cluster imposes no MaxJobs on this QOS). Each process needs
# its own LOG or the two tee streams interleave.
#
#   VOICE2_ENVS=AITA          bash run_voice2_aita_flip.sh &
#   VOICE2_ENVS=AITA-NTA-FLIP bash run_voice2_aita_flip.sh &
#
# Never point two processes at the same env: stage 1 appends to a single
# candidates.jsonl and concurrent writers would interleave partial lines.
read -r -a ENVS <<< "${VOICE2_ENVS:-AITA AITA-NTA-FLIP}"
LOG="${LOG:-/projects/$PROJECT/logs/voice2_$(echo "${ENVS[*]}" | tr ' ' '_')_driver.log}"

# Per-env lookups. Keep these explicit: deriving them bit the OEQ/PAS driver.
p6_adapter(){
  case "$1" in
    AITA)          echo "$RUNS/dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final" ;;
    AITA-NTA-FLIP) echo "$RUNS/dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final"  ;;
  esac
}
out_dir(){
  case "$1" in
    AITA)          echo "$RUNS/dpo_lora_aita_qwen3-8b_P6_voice2_rpo0p5" ;;
    AITA-NTA-FLIP) echo "$RUNS/dpo_lora_flip_qwen3-8b_P6_voice2_rpo0p5" ;;
  esac
}

say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# submit_retry <name> <sbatch args...>
# Returns 0 only when sacct State for EVERY array task is exactly COMPLETED.
# `sbatch --wait` cannot be trusted: a root scancel at ~92s exits 0:0.
submit_retry(){
  local name="$1"; shift
  local a jid st nd
  for (( a=1; a<=ATTEMPTS; a++ )); do
    jid=$(sbatch --parsable --no-requeue --exclude="$EXCLUDE" "$@") || { say "$name: sbatch failed"; return 1; }
    say "$name: submitted $jid (attempt $a/$ATTEMPTS, exclude=$EXCLUDE)"
    while squeue -j "$jid" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|COMPLETING|CONFIGURING"; do sleep 60; done
    sleep 5
    st=$(sacct -j "$jid" -X -n -o State 2>/dev/null | sort -u | tr -d ' ' | paste -sd, -)
    if [[ "$st" == "COMPLETED" ]]; then say "$name: OK ($jid)"; return 0; fi
    say "$name: attempt $a ended state=[$st]"

    # A Python traceback is a CODE fault: it will reproduce on every node, so
    # retrying is pure waste and the node-exclusion heuristic below would
    # blacklist healthy nodes for it (this happened on 2026-07-22, job 387150:
    # a KeyError got n10 excluded before the run was killed by hand). Only the
    # flaky failures -- root scancel, co-tenant OOM -- are worth another
    # attempt, and neither produces a traceback.
    if grep -qm1 "^Traceback (most recent call last):" \
         /projects/$PROJECT/logs/voice2_*_"$jid".err 2>/dev/null; then
      say "$name: traceback in job $jid — code fault, not retrying"
      grep -m1 -A3 "^Traceback" /projects/$PROJECT/logs/voice2_*_"$jid".err 2>/dev/null | tee -a "$LOG"
      return 1
    fi

    nd=$(sacct -j "$jid" -X -n -o NodeList 2>/dev/null | tr -d ' ' | head -1)
    if [[ "$st" == *FAILED* && -n "$nd" && "$nd" != None* && ",$EXCLUDE," != *",$nd,"* ]]; then
      EXCLUDE="$EXCLUDE,$nd"; say "$name: excluding $nd -> $EXCLUDE"
    fi
  done
  say "$name: FAILED after $ATTEMPTS attempts"; return 1
}

say "============ voice2 AITA/FLIP driver start (TRAIN=$TRAIN) ============"

declare -A YIELD

for env in "${ENVS[@]}"; do
  ADAPTER=$(p6_adapter "$env"); OUTDIR=$(out_dir "$env")
  say "===== env=$env  from=$ADAPTER  into=$OUTDIR ====="
  [[ -f "$ADAPTER/adapter_config.json" ]] || { say "no P6 adapter at $ADAPTER"; exit 1; }
  [[ "$ADAPTER" != "$OUTDIR" ]] || { say "refusing to overwrite source adapter"; exit 1; }
  mkdir -p "$OUTDIR"

  submit_retry "sample/$env" --job-name="v2samp_${env}" \
    --export=ALL,VS_STAGE=sample,VS_ENV="$env",VS_ADAPTER="$ADAPTER",VS_OUTDIR="$OUTDIR" \
    "$SS/marlowe_voice2_stage.sbatch" || exit 1

  submit_retry "classify/$env" --job-name="v2class_${env}" \
    --export=ALL,VS_STAGE=classify,VS_ENV="$env",VS_OUTDIR="$OUTDIR" \
    "$SS/marlowe_voice2_stage.sbatch" || exit 1

  say "building pairs for $env (rejected=collapsed, chosen=uniform clean draw, same prompt)"
  "$PY" "$REPO/training_mitigations/eval/build_dpo_pairs.py" \
      --out-dir "$OUTDIR" --max-pairs-per-prompt 10 --seed 0 2>&1 | tee -a "$LOG" || exit 1

  n=$(wc -l < "$OUTDIR/pairs/pairs.jsonl")
  YIELD[$env]=$n
  say "$env: n_pairs=$n"
done

say "--- YIELD REPORT (OEQ reference: 448 pairs / PAS: 2016) ---"
for env in "${ENVS[@]}"; do say "  $env: ${YIELD[$env]} pairs"; done

if [[ "$TRAIN" != "1" ]]; then
  say "HALT: not training. Review yield above, then re-invoke with TRAIN=1."
  exit 0
fi

for env in "${ENVS[@]}"; do
  ADAPTER=$(p6_adapter "$env"); OUTDIR=$(out_dir "$env")
  if (( ${YIELD[$env]} < MIN_PAIRS )); then
    say "$env: SKIP train — ${YIELD[$env]} pairs < MIN_PAIRS=$MIN_PAIRS"
    continue
  fi
  submit_retry "train/$env" --job-name="v2train_${env}" \
    --export=ALL,VS_STAGE=train,VS_ENV="$env",VS_ADAPTER="$ADAPTER",VS_OUTDIR="$OUTDIR",VS_ALPHA=0.5,VS_BETA=0.1 \
    "$SS/marlowe_voice2_stage.sbatch" || exit 1
  [[ -f "$OUTDIR/.complete" ]] || { say "$env trained but no .complete"; exit 1; }
  say "$env trained -> $OUTDIR"
done

# ------------------------------------------------------------------- 5. gen
# Held-out generation on the FROZEN TEST split (--use-final-data --final-split
# test inside marlowe_final_gen.sbatch), disjoint from the TRAIN ids the pairs
# were built from. Shard count is per-env, not global: AITA generates at 1,
# AITA-NTA-FLIP at 2 (marlowe_final_gen.sbatch:53-69 owns those defaults, so
# they are left unset here rather than duplicated).
#
# Submitted PER SHARD, not as one array: an array row in squeue hides a
# scancelled task behind a healthy sibling, which is how an earlier run shipped
# half-complete shards unnoticed.
for env in "${ENVS[@]}"; do
  OUTDIR=$(out_dir "$env")
  [[ -f "$OUTDIR/.complete" ]] || { say "gen/$env: SKIP — no trained adapter"; continue; }
  case "$env" in
    AITA)          NSH=1 ;;
    AITA-NTA-FLIP) NSH=2 ;;
  esac
  for (( sh=0; sh<NSH; sh++ )); do
    f="$OUTDIR/eval/aita_responses_final_finaldata_temp1.0_seed0_shard${sh}of${NSH}.csv"
    [[ -f "$f" ]] && { say "gen/$env shard$sh already present"; continue; }
    submit_retry "gen/$env/shard$sh" --job-name="v2gen_${env}${sh}" --array=$sh \
      --export=ALL,GEN_ENV="$env",GEN_RUNDIR="$OUTDIR",GEN_KIND=trained,GEN_STEER=none,GEN_NUM_SHARDS=$NSH \
      "$SS/marlowe_final_gen.sbatch" || exit 1
  done
  say "gen/$env done -> $OUTDIR/eval/"
done

say "============ voice2 AITA/FLIP driver DONE ============"
for env in "${ENVS[@]}"; do say "  $env -> $(out_dir "$env")/eval/"; done
