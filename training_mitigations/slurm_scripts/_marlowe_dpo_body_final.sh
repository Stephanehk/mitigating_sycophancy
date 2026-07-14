# FINAL-DATA variant of _marlowe_dpo_body.sh. FROZEN COPY of that body differing
# in EXACTLY ONE line: it passes --use-final-data to train_dpo_lora.py, so the
# arm trains on final_training_eval_data/<env>_train and holds out
# <env>_test (sizes from the frozen id files; --train-n/--test-n/--seed for the
# split are ignored). Kept as a separate file (NOT a toggle) so the paper's
# final-run recipe is immune to later edits of the tuning body. If you change
# site paths or hyperparameters, mirror the edit in _marlowe_dpo_body.sh.
#
# Shared body for the Marlowe DPO training wrappers
# (marlowe_final_train_*.sbatch source this). The wrapper sets RUN_ENV, OUT_NAME,
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
ENV_PREFIX=/projects/$PROJECT/envs/llm126        # cu126 conda env (matches Marlowe's r565/CUDA-12.7 driver)
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
# ENV_PREFIX (llm126) is fully self-contained; ignore ~/.local (which still holds
# the old CUDA-13 torch/vllm that crash on the r565/CUDA-12.7 driver) so imports
# are deterministic and never shadowed by user-site packages.
export PYTHONNOUSERSITE=1

# Use the env's interpreter DIRECTLY. `conda activate "$ENV_PREFIX"` does not
# reliably switch `python` on the compute nodes (the original runs silently fell
# back to base Mambaforge python + ~/.local user-site packages — which is how the
# CUDA-13 stack crept in). The llm126 env is self-contained (PYTHONNOUSERSITE=1),
# so invoking its python directly is deterministic. Verified on a GPU node.
cd "$REPO"
PY="$ENV_PREFIX/bin/python"

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

# Targeted ("self") distill knob. When DISTILL_SELF_ONLY is set (requires
# RPO_ALPHA), the RPO/distill SFT term is masked to ONLY the responses sampled
# on-policy from the model itself (mixed-sampling source "model") that survived
# the veto, <= DISTILL_MAX_PER_PROMPT distinct per prompt. The DPO cross-pairs
# are unchanged. Isolates the distill target from the critical-skewed
# stakeholder/perspshift/frozen-base exemplars. See core/self_distill_trainer.py.
if [[ -n "${DISTILL_SELF_ONLY:-}" ]]; then
  EXTRA_ARGS+=(--distill-self-only --distill-max-per-prompt "${DISTILL_MAX_PER_PROMPT:-4}")
  echo "[marlowe] targeted distill (self-source, <= ${DISTILL_MAX_PER_PROMPT:-4}/prompt) ON"
fi

# Candidate-mix knobs. Wrappers override these before sourcing; the DEFAULTS
# reproduce the original nocrit_distill recipe (4 model / 0 critical / 6
# stakeholder role-play), so every existing wrapper is unchanged. The
# perspshift_distill_gp arm sets N_SOURCE_MODEL=2 N_SOURCE_CRITICAL=2 PERSPSHIFT=1
# STAKEHOLDER_VOICE=1 (+ PERSPSHIFT_ASSESS=1 for OEQ). See core/trainer.py.
N_SOURCE_MODEL="${N_SOURCE_MODEL:-4}"
N_SOURCE_CRITICAL="${N_SOURCE_CRITICAL:-0}"
EXTRA_ARGS+=(--n-source-model "$N_SOURCE_MODEL" --n-source-critical "$N_SOURCE_CRITICAL")
echo "[marlowe] mix: n_source_model=$N_SOURCE_MODEL n_source_critical=$N_SOURCE_CRITICAL"
if [[ -n "${STAKEHOLDER_VOICE:-}" ]]; then
  EXTRA_ARGS+=(--stakeholder-voice)
  echo "[marlowe] stakeholder-voice (gp) ON"
fi
if [[ -n "${FROZEN_STAKEHOLDER:-}" ]]; then
  EXTRA_ARGS+=(--frozen-stakeholder)
  echo "[marlowe] frozen stakeholder (base teacher) ON"
fi
if [[ -n "${PERSPSHIFT:-}" ]]; then
  EXTRA_ARGS+=(--perspective-shift-sampling --rewrite-max-tokens "${REWRITE_MAX_TOKENS:-1536}")
  [[ -n "${PERSPSHIFT_ASSESS:-}" ]] && EXTRA_ARGS+=(--perspshift-assess)
  echo "[marlowe] perspective-shift sampling ON (assess=${PERSPSHIFT_ASSESS:-0})"
fi
# FUTURE_USER=1 turns the N_SOURCE_CRITICAL steer slots into the future-self
# voice (mutually exclusive with PERSPSHIFT; both consume that budget). Set
# N_SOURCE_CRITICAL>0 alongside it.
if [[ -n "${FUTURE_USER:-}" ]]; then
  EXTRA_ARGS+=(--future-user-sampling)
  echo "[marlowe] future-user voice sampling ON (fills n_source_critical slots)"
fi
# ADDRESSEE_FILTER=1 drops candidates that critique the message's FORM instead of
# the narrator's situation, before the veto panel. Composes with any VETO_PROMPT.
if [[ -n "${ADDRESSEE_FILTER:-}" ]]; then
  EXTRA_ARGS+=(--addressee-filter)
  echo "[marlowe] addressee filter ON (drop message-critique candidates)"
fi

# Veto panel. VETO_PROMPT selects the veto method (default P5, so every existing
# wrapper is unchanged). USER_STAKEHOLDER=1 adds the 4th "user"/narrator voice to
# the veto panel (veto-only, NOT sampled for candidates); train_dpo_lora.py asserts
# this is P6-only, so we guard it here too for a clearer failure. See
# core/simulated_stakeholders.py VETO_PROMPT_USER / include_user_stakeholder.
if [[ -n "${USER_STAKEHOLDER:-}" ]]; then
  [[ "${VETO_PROMPT:-P5}" == "P6" ]] || { echo "ERROR: USER_STAKEHOLDER=1 requires VETO_PROMPT=P6 (the 4th narrator voice is P6-only)." >&2; exit 1; }
  EXTRA_ARGS+=(--user-stakeholder)
  echo "[marlowe] user-stakeholder (4th narrator/user veto voice) ON"
fi

# Data split: default 1000 train / 500 test (OEQ/PAS). AITA/FLIP wrappers
# override TRAIN_N/TEST_N to match their held-out baselines (FLIP counts PAIRS).
echo "[marlowe] split: train_n=${TRAIN_N:-1000} test_n=${TEST_N:-500}"
echo "[marlowe] num_iterations=${NUM_ITERATIONS:-3} (with --allow-preempt, resumes from highest finished adapter_iter_<k>)"

# Run the trainer in the background so the trap can fire while it runs.
"$PY" training_mitigations/train_dpo_lora.py \
    --env "$RUN_ENV" \
    --output-dir "$OUT_DIR" \
    --allow-preempt \
    --use-final-data \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --veto-prompt "${VETO_PROMPT:-P5}" \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --sim-backend vllm \
    --veto-concurrency 16 \
    --precompute-ref-batch-size 8 \
    --dpo-save-steps "${DPO_SAVE_STEPS:-25}" \
    --num-iterations "${NUM_ITERATIONS:-3}" \
    --train-n "${TRAIN_N:-1000}" \
    --test-n "${TEST_N:-500}" \
    --n-samples 10 \
    --mixed-sampling \
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
