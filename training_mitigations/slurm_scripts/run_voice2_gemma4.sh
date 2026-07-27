#!/usr/bin/env bash
# Gemma-4 "voice2" (+1) driver: the extra RPO epoch that mitigates voice
# collapse, run AFTER each base P6 rpo0p5 Gemma adapter completes.
#
# Per env: wait for base .complete -> sample (10/prompt from the P6 adapter) ->
# classify (v2 collective-voice rubric, Gemma judge) -> build_dpo_pairs (inline)
# -> offline RPO train (1 epoch continuing the P6 adapter). Uses
# marlowe_voice2_stage_gemma4.sbatch (gemma env + CUDA-13 forward-compat +
# --model-id gemma + --cache-dir hub) for every GPU stage.
#
# Skips the one-time validation gate in run_voice2_all.sh: that gate validated
# the v2 RUBRIC (prompt), which is model-independent and already passed; only the
# judge backend changed (Qwen -> Gemma).
#
# Idempotent: sample/classify resume from partial output, train skips on
# .complete. submit_retry re-checks sacct State (a root scancel exits 0:0) and
# resubmits, excluding a node that OOMs.
#
# Launch detached so it survives the session (base runs for many hours):
#   nohup bash run_voice2_gemma4.sh > /projects/m000254/logs/voice2_gemma4_driver.log 2>&1 &
set -uo pipefail

PROJECT=m000254
SS=/projects/$PROJECT/mitigating_sycophancy/training_mitigations/slurm_scripts
PY=/projects/$PROJECT/envs/gemma4vllm/bin/python
RUNS=/projects/$PROJECT/runs
LOG=/projects/$PROJECT/logs/voice2_gemma4_driver.log
EXCLUDE="${EXCLUDE:-}"
ATTEMPTS=4
ENVS=(OEQ PAS)

say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
base_adapter(){ echo "$RUNS/dpo_lora_${1,,}_gemma4-12b-it_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"; }
out_dir(){ echo "$RUNS/dpo_lora_${1,,}_gemma4-12b-it_P6_voice2_rpo0p5"; }

submit_retry(){
  local name="$1"; shift
  local a jid st exarg=()
  [[ -n "$EXCLUDE" ]] && exarg=(--exclude="$EXCLUDE")
  for (( a=1; a<=ATTEMPTS; a++ )); do
    jid=$(sbatch --parsable --no-requeue "${exarg[@]}" "$@") || { say "$name: sbatch failed"; return 1; }
    say "$name: submitted $jid (attempt $a/$ATTEMPTS, exclude='${EXCLUDE:-none}')"
    while squeue -j "$jid" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|COMPLETING|CONFIGURING"; do sleep 60; done
    sleep 5
    st=$(sacct -j "$jid" -X -n -o State 2>/dev/null | sort -u | tr -d ' ' | paste -sd, -)
    [[ "$st" == "COMPLETED" ]] && { say "$name: OK ($jid)"; return 0; }
    say "$name: attempt $a ended state=[$st]"
    local nd; nd=$(sacct -j "$jid" -X -n -o NodeList 2>/dev/null | tr -d ' ' | head -1)
    if [[ "$st" == *FAILED* && -n "$nd" && "$nd" != None* && ",$EXCLUDE," != *",$nd,"* ]]; then
      EXCLUDE="${EXCLUDE:+$EXCLUDE,}$nd"; say "$name: excluding $nd -> $EXCLUDE"; exarg=(--exclude="$EXCLUDE")
    fi
  done
  say "$name: FAILED after $ATTEMPTS attempts"; return 1
}

say "================= voice2 gemma4 driver start ================="
for env in "${ENVS[@]}"; do
  ADAPTER=$(base_adapter "$env"); OUTDIR=$(out_dir "$env")
  say "===== env=$env  from=$ADAPTER  into=$OUTDIR ====="

  # 0. wait for the base P6 adapter to finish (self-resubmit chain -> .complete).
  say "$env: waiting for base .complete ..."
  while [[ ! -f "$ADAPTER/.complete" ]]; do sleep 120; done
  [[ -f "$ADAPTER/adapter_config.json" ]] || { say "$env: .complete but no adapter_config.json at $ADAPTER"; exit 1; }
  [[ "$ADAPTER" != "$OUTDIR" ]] || { say "$env: refusing to overwrite source adapter"; exit 1; }
  mkdir -p "$OUTDIR"
  say "$env: base adapter ready"

  # 1. sample 10/prompt from the P6 adapter (on-policy).
  submit_retry "sample/$env" --job-name="v2gsamp_$env" \
    --export=ALL,VS_STAGE=sample,VS_ENV="$env",VS_ADAPTER="$ADAPTER",VS_OUTDIR="$OUTDIR" \
    "$SS/marlowe_voice2_stage_gemma4.sbatch" || exit 1

  # 2. classify voice collapse (v2 rubric, Gemma judge).
  submit_retry "classify/$env" --job-name="v2gclass_$env" \
    --export=ALL,VS_STAGE=classify,VS_ENV="$env",VS_OUTDIR="$OUTDIR" \
    "$SS/marlowe_voice2_stage_gemma4.sbatch" || exit 1

  # 3. build pairs (CPU, inline): rejected=collapsed draw, chosen=uniform clean draw.
  say "$env: building pairs"
  "$PY" /projects/$PROJECT/mitigating_sycophancy/training_mitigations/eval/build_dpo_pairs.py \
      --out-dir "$OUTDIR" --max-pairs-per-prompt 10 --seed 0 2>&1 | tee -a "$LOG" || exit 1

  # 4. offline RPO train, 1 epoch continuing the P6 adapter.
  submit_retry "train/$env" --job-name="v2gtrain_$env" \
    --export=ALL,VS_STAGE=train,VS_ENV="$env",VS_ADAPTER="$ADAPTER",VS_OUTDIR="$OUTDIR",VS_ALPHA=0.5,VS_BETA=0.1 \
    "$SS/marlowe_voice2_stage_gemma4.sbatch" || exit 1

  [[ -f "$OUTDIR/.complete" ]] || { say "$env: trained but no .complete"; exit 1; }
  say "$env: voice2 DONE -> $OUTDIR"
done
say "================= voice2 gemma4 driver DONE ================="
