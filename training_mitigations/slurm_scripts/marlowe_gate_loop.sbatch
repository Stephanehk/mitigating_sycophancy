#!/bin/bash
#SBATCH --job-name=gateloop
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/gateloop_%A_%a.out
#SBATCH --error=/projects/m000254/logs/gateloop_%A_%a.err
#
# Assistant-mode gate rejection sampling, one array task per arm.
#
# For every held-out prompt whose ATTEMPT-1 response the gate rejected, resample
# from the SAME arm and judge each draw inline, stopping at the first pass or at
# --gate-max-attempts. Prompts exhausting all attempts are recorded with
# gate_passed=False and are tallied, not scored.
#
# The judge is called from the compute node: these nodes DO have outbound HTTPS
# (verified on n31 -> openrouter.ai 200), contrary to the "air-gapped" comments
# elsewhere in this directory.
#
# Arm flags mirror marlowe_final_gen.sbatch:73-93 exactly -- a resample drawn
# from a different arm is not a resample.
set -euo pipefail
source /projects/m000254/mitigating_sycophancy/training_mitigations/slurm_scripts/_marlowe_eval_env.sh
: "${OPENROUTER_API_KEY:?gate loop judges inline; OPENROUTER_API_KEY must be exported}"

JOBS=/projects/m000254/logs/gateloop_jobs.tsv
IFS=$'\t' read -r ENV RUN KIND STEER TAG WL NIDS < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$JOBS")
RUNDIR=/projects/m000254/runs/$RUN

case "$ENV" in
  OEQ|PAS)            DRIVER=training_mitigations/eval/eval_endorsement_heldout.py
                      PATH_ARGS=(--no-judge --sample-max-tokens 512) ;;
  AITA|AITA-NTA-FLIP) DRIVER=training_mitigations/eval/eval_aita_label_heldout.py
                      PATH_ARGS=(--sample-max-tokens 1536) ;;
esac

MODEL_ARGS=()
if [[ "$KIND" == "trained" ]]; then
  [[ -f "$RUNDIR/adapter_config.json" ]] || { echo "ERROR: no adapter in $RUNDIR" >&2; exit 1; }
else
  MODEL_ARGS=(--base)
  case "$STEER" in
    none)       ;;
    becritical) MODEL_ARGS+=(--be-critical) ;;
    explicit)   MODEL_ARGS+=(--explicit) ;;
    sharma)     MODEL_ARGS+=(--sharma-honesty) ;;
    dontbesyco) MODEL_ARGS+=(--gen-suffix "Don't be sycophantic." --gen-tag dontbesyco) ;;
    perspshift) MODEL_ARGS+=(--perspective-shift --assess-narrative) ;;
    stakeholder) MODEL_ARGS+=(--gen-suffix "consider stakeholders that may be affected by user behavior, and provide an objective response" --gen-tag stakeholder) ;;
    *) echo "ERROR: unknown STEER=$STEER" >&2; exit 1 ;;
  esac
fi

echo "[gateloop] env=$ENV kind=$KIND steer=$STEER run=$RUN prompts=$NIDS"
"$PY" "$DRIVER" \
    --run-dir "$RUNDIR" --env "$ENV" \
    --use-final-data --final-split test --tag-extra finaldata \
    --device cuda:0 --seed 0 --num-shards 1 --shard-id 0 \
    --gate-loop --gate-max-attempts 5 --resample-ids-file "$WL" \
    "${PATH_ARGS[@]}" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
echo "[gateloop] done $RUN"
