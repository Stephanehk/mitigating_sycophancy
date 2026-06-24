# Shared body for the Marlowe DPO training wrappers
# (marlowe_run_train_*.sbatch source this). The wrapper sets RUN_ENV, OUT_NAME,
# and SELF_BASENAME before sourcing; ALL Marlowe site paths are configured HERE
# so there is a single place to edit.
#
# WHY this exists: Marlowe Basic Access runs on the `preempt` partition, which
# caps jobs at 12h and may preempt them (with a ~30-min grace signal). Our DPO
# runs are multi-hour/multi-iteration, so instead of one long job we run a
# SELF-RESUBMITTING CHAIN of <=12h jobs. Each (re)start passes --allow-preempt,
# which auto-detects the last finished iteration + partial sampling and
# continues from there (see train_dpo_lora.py / core/trainer.py). A successor
# job is queued up front with afterany so it never overlaps this one, and is
# cancelled if this job finishes the run.

set -uo pipefail

# ===================== EDIT FOR YOUR MARLOWE SITE =====================
PROJECT=m000254                                  # your Marlowe project ID
REPO=/projects/$PROJECT/sycophancy               # where you put the repo
ENV_PREFIX=/projects/$PROJECT/envs/llm           # the conda env prefix
export HF_HOME=/projects/$PROJECT/hf_cache       # HF model cache (1T projects quota)
RUNS_ROOT=/projects/$PROJECT/runs                 # training outputs
LOGS_ROOT=/projects/$PROJECT/logs
MAX_ATTEMPTS=20                                  # cap on the resubmit chain
# =====================================================================

ATTEMPT="${MARLOWE_ATTEMPT:-1}"
OUT_DIR="$RUNS_ROOT/$OUT_NAME"
SCRIPT_PATH="$REPO/training_mitigations/slurm_scripts/$SELF_BASENAME"
mkdir -p "$HF_HOME" "$RUNS_ROOT" "$LOGS_ROOT" "$OUT_DIR"

# Already finished on a previous attempt? Nothing to do. ($OUT_DIR/.complete is
# written by trainer.train() after the final adapter is saved.)
if [[ -f "$OUT_DIR/.complete" ]]; then
  echo "[marlowe] $OUT_DIR/.complete exists — run already finished. Exiting."
  exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set; export it before sbatch (Qwen3-8B is gated)." >&2
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$REPO"
PY=$(command -v python)

echo "[marlowe] env=$RUN_ENV attempt=$ATTEMPT/$MAX_ATTEMPTS jobid=${SLURM_JOB_ID:-?} node=$(hostname)"
echo "[marlowe] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[marlowe] out=$OUT_DIR python=$PY"

# Queue the successor up front. --dependency=afterany means it cannot start
# until THIS job reaches a TERMINAL state, so the two never write OUT_DIR
# concurrently. It covers BOTH the signalled grace path and an unsignalled hard
# kill (hardware failure). Bounded by MAX_ATTEMPTS; cancelled below if this job
# finishes the run.
#
# CRITICAL invariant: this job must TERMINATE on preemption, not requeue. If the
# preempt partition's PreemptMode requeued it, the requeued job (not terminal)
# would run again and queue ANOTHER afterany successor on the same job id; when
# the job finally ended, every queued successor would release at once and run
# OUT_DIR concurrently. The wrapper sets #SBATCH --no-requeue to force
# termination so exactly one successor ever continues the chain.
SUCCESSOR_ID=""
if (( ATTEMPT < MAX_ATTEMPTS )); then
  SUCCESSOR_ID=$(sbatch --parsable \
      --dependency=afterany:"$SLURM_JOB_ID" \
      --export=ALL,MARLOWE_ATTEMPT=$((ATTEMPT + 1)) \
      "$SCRIPT_PATH")
  echo "[marlowe] queued successor job $SUCCESSOR_ID (attempt $((ATTEMPT + 1)))"
else
  echo "[marlowe] at MAX_ATTEMPTS=$MAX_ATTEMPTS; no successor queued." >&2
fi

# Two ways a window ends early on the preempt partition:
#   - WALL expiry: the wrapper's --signal=B:USR1@300 delivers USR1 ~5 min before
#     the 4h limit.
#   - PREEMPTION: SLURM delivers SIGTERM, then SIGKILL after GraceTime (~15 min).
# Trap both and stop the trainer cleanly. Progress is durable regardless of a
# clean stop: per-prompt sampling flushes to responses_iter_*.jsonl, and the DPO
# train phase checkpoints every --dpo-save-steps; the queued successor resumes
# from the latest of those. Leave .complete absent so the successor knows to
# resume.
on_signal() {
  echo "[marlowe] grace signal (USR1/TERM) — stopping trainer for checkpoint."
  kill -TERM "${PY_PID:-}" 2>/dev/null || true
  wait "${PY_PID:-}" 2>/dev/null || true
  exit 0
}
trap on_signal USR1 TERM

# Optional RPO / "distill" term. A wrapper sets RPO_ALPHA before sourcing this
# body to add an SFT(chosen) term to the DPO loss (--rpo-alpha); unset => plain
# DPO. Re-established by the wrapper on every resubmit attempt, like RUN_ENV.
EXTRA_ARGS=()
if [[ -n "${RPO_ALPHA:-}" ]]; then
  EXTRA_ARGS+=(--rpo-alpha "$RPO_ALPHA")
  echo "[marlowe] RPO/distill mode: --rpo-alpha $RPO_ALPHA"
fi

# Run the trainer in the background so the trap can fire while it runs.
"$PY" training_mitigations/train_dpo_lora.py \
    --env "$RUN_ENV" \
    --output-dir "$OUT_DIR" \
    --allow-preempt \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --veto-prompt P5 \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --sim-backend vllm \
    --veto-concurrency 16 \
    --precompute-ref-batch-size 8 \
    --dpo-save-steps 25 \
    --num-iterations 3 \
    --train-n 1000 \
    --test-n 500 \
    --n-samples 10 \
    --mixed-sampling \
    --n-source-model 4 \
    --n-source-critical 0 \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-tokens 1024 \
    --sample-max-tokens 1024 \
    --beta 0.1 \
    --learning-rate 5e-5 \
    --num-train-epochs 1 \
    --per-device-batch-size 1 \
    --grad-accum 8 \
    --max-length 3072 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 &
PY_PID=$!
wait "$PY_PID"
rc=$?
echo "[marlowe] trainer exited rc=$rc"

# Clean finish? Cancel the queued successor so it doesn't spin up just to no-op.
if [[ -f "$OUT_DIR/.complete" ]]; then
  echo "[marlowe] run complete."
  [[ -n "$SUCCESSOR_ID" ]] && { echo "[marlowe] cancelling successor $SUCCESSOR_ID"; scancel "$SUCCESSOR_ID" 2>/dev/null || true; }
else
  echo "[marlowe] not complete (rc=$rc); successor ${SUCCESSOR_ID:-<none>} will resume."
fi
exit "$rc"
