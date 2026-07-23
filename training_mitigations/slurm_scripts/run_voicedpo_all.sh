#!/bin/bash
# END-TO-END driver for the filtered-pair DPO round (voice-DPO arm), OEQ + PAS.
#
# Runs unattended from a LOGIN node. The chain necessarily straddles two worlds:
# GPU stages must run as sbatch jobs on air-gapped compute nodes, while every
# gpt-5-mini judge needs outbound HTTPS and therefore must run HERE. So GPU
# stages are submitted with `sbatch --wait` (blocks until the job exits, no
# polling logic) and the judge stages run inline.
#
#   Per env:  sample -> classify -> build pairs -> DPO train -> held-out gen
#   Then:     (a) action-endorsement judge   (login, gpt-5-mini)
#             (b) addressee + voice judges   (login, gpt-5-mini)
#             (c) ArmoRM                     (GPU, pinned armorm env)
#             summary table
#
# Every stage is idempotent: the sample/classify stages resume from partial
# output, the trainer skips a completed adapter, and the judge drivers skip any
# condition already scored. A re-run after an interruption picks up where it
# stopped. That also means the judge drivers will fill in any MISSING baseline
# for the arms they already know about -- notably the addressee judge on the P6
# arm itself, which has never been run and which (b) needs as its comparison.
#
# Usage (from the repo root or anywhere):
#   export OPENROUTER_API_KEY=...        # in ~/.bashrc on Marlowe
#   nohup bash training_mitigations/slurm_scripts/run_voicedpo_all.sh all \
#         > /projects/m000254/logs/voicedpo_driver.log 2>&1 &
#
# Args: [all|oeq|pas]   (default all)
# Env overrides: SKIP_TRAIN=1 SKIP_EVAL=1 DRY_RUN=1
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
LOGS=/projects/$PROJECT/logs
PY=/projects/$PROJECT/envs/llm126/bin/python
SS=$REPO/training_mitigations/slurm_scripts
cd "$REPO"

GROUP="${1:-all}"
DRY_RUN="${DRY_RUN:-}"
mkdir -p "$LOGS"

case "$GROUP" in
  oeq) ENVS=(OEQ) ;;
  pas) ENVS=(PAS) ;;
  all) ENVS=(OEQ PAS) ;;
  *) echo "usage: $0 [all|oeq|pas]" >&2; exit 2 ;;
esac

# Fail fast, before hours of GPU time: the judge stages at the end are useless
# without this and there is nobody watching to notice.
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY before starting (judge stages need it)}"

p6_adapter() {   # env -> abs path to the trained P6 adapter (read-only input)
  local e=$1; local lc=${e,,}
  echo "$RUNS_ROOT/dpo_lora_${lc}_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"
}
# VD_SUFFIX distinguishes hyperparameter variants so they never share a run dir.
# Empty (default) reproduces the original beta=0.1 arm's path exactly.
VD_SUFFIX="${VD_SUFFIX:-}"
export VD_BETA="${VD_BETA:-0.1}"
out_dir() {      # env -> abs path to the NEW run dir
  local e=$1; local lc=${e,,}
  echo "$RUNS_ROOT/dpo_lora_${lc}_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo${VD_SUFFIX}"
}

run() {
  echo "+ $*"
  [[ -n "$DRY_RUN" ]] && return 0
  "$@"
}

# sbatch --wait blocks until the job terminates and propagates its exit status.
#
# Retries because the batch partition shares nodes: a stage can land on a GPU
# another job has already filled and vLLM then refuses to start ("Free memory on
# device cuda:0 ... is less than desired GPU memory utilization"). Every GPU
# stage here is idempotent and resumable, so a retry costs only the work not yet
# done. Without this, one unlucky allocation kills an unattended overnight chain.
ATTEMPTS="${ATTEMPTS:-4}"
RETRY_SLEEP="${RETRY_SLEEP:-120}"
# Nodes that failed a stage get excluded from every later submission. Slurm will
# otherwise keep handing back the same bad GPU (observed: three consecutive
# allocations to the same occupied device), so a plain retry loop never escapes.
#
# --no-requeue on every submission: a requeued job can come back as
# "user_env_retrieval_failed_requeued_held", which parks it in HELD forever --
# and `sbatch --wait` then blocks forever with it, silently stalling the whole
# unattended chain. Every stage here is idempotent and resumable, so it is
# strictly better for a dead job to stay dead and let the retry above resubmit.
# Pre-seed with EXCLUDE_NODES=n06,... for nodes already known to be occupied,
# so the first attempt of every stage does not have to rediscover them.
EXCLUDE="${EXCLUDE_NODES:-}"
submit_wait() {
  local name=$1; shift
  [[ -n "$DRY_RUN" ]] && { echo "+ sbatch $*"; return 0; }
  local a jid node rc state
  for (( a=1; a<=ATTEMPTS; a++ )); do
    echo "[driver] submitting $name (attempt $a/$ATTEMPTS${EXCLUDE:+, excluding $EXCLUDE}, $(date -Is))"
    if [[ -n "$EXCLUDE" ]]; then
      jid=$(sbatch --parsable --wait --no-requeue --exclude="$EXCLUDE" "$@") ; rc=$?
    else
      jid=$(sbatch --parsable --wait --no-requeue "$@") ; rc=$?
    fi
    # sbatch --wait exits 0 for a job that was CANCELLED (scancel reports
    # ExitCode 0:0), so rc alone cannot distinguish success from a job that was
    # killed out from under us. Observed: job 383761 scancelled by uid 0 on n01
    # 93 s in, rc=0, the driver logged "finished", and the next stage then died
    # on the output the cancelled job never wrote. Require COMPLETED explicitly
    # so anything else falls through to the retry-and-exclude path below.
    # -X gives one row per array task, so gen (--array=0-1) yields two: require
    # EVERY task to be COMPLETED, not just the first, or a killed task 1 hides
    # behind a healthy task 0.
    if (( rc == 0 )); then
      state=$(sacct -j "${jid%%;*}" -X -n -o State 2>/dev/null | awk 'NF{print $1}' | sort -u | tr '\n' ',' | sed 's/,$//')
      if [[ "$state" == "COMPLETED" ]]; then
        echo "[driver] finished $name (job ${jid:-?}, $(date -Is))"
        return 0
      fi
      echo "[driver] job ${jid:-?} exited 0 but state=${state:-UNKNOWN} — treating as failure" >&2
      rc=1
    fi
    node=$(sacct -j "${jid%%;*}" -X -n -o NodeList 2>/dev/null | head -1 | tr -d ' ')
    echo "[driver] attempt $a of $name failed (job ${jid:-?} on ${node:-?})" >&2
    if [[ -n "$node" && "$node" != "None"* ]]; then
      EXCLUDE="${EXCLUDE:+$EXCLUDE,}$node"
      echo "[driver] excluding $node from subsequent submissions" >&2
    fi
    (( a < ATTEMPTS )) && sleep "$RETRY_SLEEP"
  done
  echo "[driver] FAILED after $ATTEMPTS attempts: $name" >&2
  return 1
}

echo "=================================================================="
echo "[driver] voice-DPO chain start $(date -Is)  envs=${ENVS[*]}  beta=$VD_BETA  suffix='${VD_SUFFIX:-<none>}'"
echo "=================================================================="

# ---------------------------------------------------------------- train chain
if [[ -z "${SKIP_TRAIN:-}" ]]; then
for env in "${ENVS[@]}"; do
  ADAPTER=$(p6_adapter "$env")
  OUTDIR=$(out_dir "$env")
  echo "------------------------------------------------------------------"
  echo "[driver] env=$env"
  echo "[driver]   from: $ADAPTER"
  echo "[driver]   into: $OUTDIR"

  [[ -f "$ADAPTER/adapter_config.json" ]] || {
    echo "[driver] ERROR: no adapter at $ADAPTER" >&2; exit 1; }
  [[ "$ADAPTER" != "$OUTDIR" ]] || {
    echo "[driver] ERROR: refusing to overwrite the source adapter" >&2; exit 1; }
  mkdir -p "$OUTDIR"

  submit_wait "sample/$env" \
      --job-name="vdsample_${env}" \
      --export=ALL,VD_STAGE=sample,VD_ENV="$env",VD_ADAPTER="$ADAPTER",VD_OUTDIR="$OUTDIR" \
      "$SS/marlowe_voicedpo_stage.sbatch" || exit 1

  submit_wait "classify/$env" \
      --job-name="vdclass_${env}" \
      --export=ALL,VD_STAGE=classify,VD_ENV="$env",VD_ADAPTER="$ADAPTER",VD_OUTDIR="$OUTDIR" \
      "$SS/marlowe_voicedpo_stage.sbatch" || exit 1

  # CPU/IO only -- no reason to queue for a GPU.
  echo "[driver] building pairs for $env"
  run "$PY" training_mitigations/eval/build_dpo_pairs.py \
        --out-dir "$OUTDIR" --max-pairs-per-prompt 10 --seed 0 || exit 1

  submit_wait "train/$env" \
      --job-name="vdtrain_${env}" \
      --export=ALL,VD_STAGE=train,VD_ENV="$env",VD_ADAPTER="$ADAPTER",VD_OUTDIR="$OUTDIR",VD_BETA="$VD_BETA" \
      "$SS/marlowe_voicedpo_stage.sbatch" || exit 1

  [[ -n "$DRY_RUN" ]] || [[ -f "$OUTDIR/.complete" ]] || {
    echo "[driver] ERROR: $env trained but no .complete sentinel in $OUTDIR" >&2; exit 1; }
done
else
  echo "[driver] SKIP_TRAIN set — going straight to eval"
fi

# ----------------------------------------------------------------- evaluation
if [[ -n "${SKIP_EVAL:-}" ]]; then
  echo "[driver] SKIP_EVAL set — stopping after training"
  exit 0
fi

# Held-out generation: reuse the frozen final-run job verbatim (--use-final-data
# --final-split test --tag-extra finaldata --sample-max-tokens 512). OEQ/PAS MUST
# run at 2 shards -- stage_heldout_inputs.py hardcodes _shard{i}of2.
for env in "${ENVS[@]}"; do
  OUTDIR=$(out_dir "$env")
  submit_wait "gen/$env" \
      --job-name="vdgen_${env}" --array=0-1 \
      --export=ALL,GEN_ENV="$env",GEN_RUNDIR="$OUTDIR",GEN_KIND=trained,GEN_STEER=none,GEN_NUM_SHARDS=2 \
      "$SS/marlowe_final_gen.sbatch" || exit 1
done

# (a) action-endorsement judge, and the OEQ voice census, on the login node.
#     Idempotent across every arm it knows about; only the new one is missing.
echo "[driver] (a) endorsement judge $(date -Is)"
run bash "$SS/marlowe_final_endorse_all.sh" "$GROUP"

# (b) addressee (J1) + voice-collapse judges. This also backfills the addressee
#     judge for the P6 arm, which has never been run and is the baseline (b)
#     needs. run_finaldata_judges.py asserts its slug/method maps agree, so a
#     mis-registered arm fails loud here rather than mislabelling a column.
echo "[driver] (b) addressee + voice judges $(date -Is)"
run bash "$SS/run_finaldata_judges_all.sh" all

# (c) ArmoRM, back on a GPU in its own pinned env (transformers 4.43.x). Scores
#     every arm of the env whose rm_armorm_<tag>.csv is absent, so this picks up
#     the new arm and leaves the rest alone.
#
#     SKIP_ARMORM=1 skips it. This is the ONLY remaining GPU stage after gen, and
#     `sbatch --wait` blocks the driver on it -- so when the cluster is busy it can
#     hold up the summary for hours to produce a column that is cheap to backfill
#     later (`sbatch --export=ALL,SCORE_ENV=OEQ marlowe_final_armorm_score.sbatch`).
if [[ -n "${SKIP_ARMORM:-}" ]]; then
  echo "[driver] SKIP_ARMORM set — skipping ArmoRM; backfill later with:"
  for env in "${ENVS[@]}"; do
    echo "[driver]   sbatch --export=ALL,SCORE_ENV=$env $SS/marlowe_final_armorm_score.sbatch"
  done
else
for env in "${ENVS[@]}"; do
  submit_wait "armorm/$env" \
      --job-name="vdarmorm_${env}" \
      --export=ALL,SCORE_ENV="$env" \
      "$SS/marlowe_final_armorm_score.sbatch" || echo "[driver] WARN armorm failed for $env"
done
fi

# ------------------------------------------------------------------- summary
echo "[driver] summary $(date -Is)"
run "$PY" training_mitigations/eval/summarize_voicedpo.py --envs "${ENVS[@]}" \
  || echo "[driver] WARN summary failed"

echo "=================================================================="
echo "[driver] voice-DPO chain complete $(date -Is)"
echo "=================================================================="
