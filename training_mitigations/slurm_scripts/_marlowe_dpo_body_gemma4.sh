# GEMMA-4 variant of _marlowe_dpo_body_final.sh. Same frozen final-data recipe
# (--use-final-data on final_training_eval_data/<env>_train, held out on
# <env>_test) but runs the Gemma-4-12B-it policy+sim instead of Qwen3-8B. It
# differs from _marlowe_dpo_body_final.sh in EXACTLY these ways:
#   - ENV_PREFIX -> the gemma4vllm env (cu13 torch/vLLM/transformers 5.14).
#   - LD_LIBRARY_PATH prepends the staged CUDA-13 forward-compat libcuda so the
#     cu13 stack runs on Marlowe's r565 / CUDA-12.7 kernel driver (H100 is a
#     datacenter GPU; forward compat is supported and GPU-verified 2026-07-23).
#   - VLLM_USE_FLASHINFER_SAMPLER=0 (flashinfer's JIT sampler can't find curand.h
#     against the node's system CUDA 12.6; native Torch sampler is used instead).
#   - HF_TOKEN is OPTIONAL (google/gemma-4-12B-it is public/ungated).
#   - passes --model-id google/gemma-4-12B-it and the text-decoder-only
#     --lora-target-modules list (Gemma-4's modality-input projections must not
#     be LoRA-adapted; verified dead on a text-only forward).
# The Qwen final wrappers/body are untouched. If you change the training recipe,
# mirror the edit in _marlowe_dpo_body_final.sh. See the memory
# gemma4-marlowe-forward-compat-env for the full rationale.
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
ENV_PREFIX=/projects/$PROJECT/envs/gemma4vllm    # cu13 env (torch2.11/vllm0.25/transformers5.14) for gemma4_unified
COMPAT_LIB=/projects/$PROJECT/cuda-compat-13/lib # staged CUDA-13 forward-compat libcuda (runs cu13 on the r565 driver)
MODEL_ID=google/gemma-4-12B-it                   # instruction-tuned Gemma-4 (ships the chat template; public/ungated)
# Text-decoder-only LoRA targets: "all-linear" minus Gemma-4's 3 modality-input
# projections (patch_dense / *embedding_projection), which are dead on a
# text-only forward (grad-norm verified 2026-07-23).
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
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

# google/gemma-4-12B-it is PUBLIC/ungated, so HF_TOKEN is optional (unlike the
# Qwen body which hard-fails without it). Forward the token if present.
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ENV_PREFIX (gemma4vllm) is self-contained; ignore ~/.local so imports are
# deterministic and never shadowed by user-site packages.
export PYTHONNOUSERSITE=1

# --- Gemma-4 / CUDA-13 forward-compat runtime env (see header) ---
# Prepend the staged forward-compat libcuda so the cu13 torch/vLLM stack finds
# the 580-branch libcuda.so.1 BEFORE the node's system r565 one — this is what
# lets CUDA 13 run on Marlowe's CUDA-12.7 driver.
export LD_LIBRARY_PATH="$COMPAT_LIB:${LD_LIBRARY_PATH:-}"
# vLLM's flashinfer sampler JIT-compiles against the node's system CUDA 12.6
# (missing curand.h) and fails; fall back to the native Torch sampler. The
# trainer's vLLM sim/policy subprocess inherits this env.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Compute nodes are air-gapped but the model is cached — force offline so HF/vLLM
# never block on a network fetch.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Use the env's interpreter DIRECTLY. `conda activate "$ENV_PREFIX"` does not
# reliably switch `python` on the compute nodes (the original runs silently fell
# back to base Mambaforge python + ~/.local user-site packages). The gemma4vllm
# env is self-contained (PYTHONNOUSERSITE=1), so invoking its python directly is
# deterministic. Verified on a GPU node.
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

# MID-EPOCH CHECKPOINTS ARE UNSAFE TO RESUME FROM: a mid-DPO-epoch resume
# restores the LR schedule but NOT the LoRA weights, silently zeroing them and
# under-training the adapter. That bug invalidated the OEQ and AITA final arms.
# The Gemini body pins --dpo-save-steps 0 for exactly this reason; mirror that
# for the assistant-role arms so a walltime overrun restarts from the last
# COMPLETED iteration (durable via responses_iter_*.jsonl) instead of resuming
# mid-epoch. In the happy path this changes nothing: with no resume, the
# checkpoints are written and never read, so training is identical.
if [[ -n "${ASSISTANT_ROLE_VETO:-}" && -z "${DPO_SAVE_STEPS:-}" ]]; then
  DPO_SAVE_STEPS=0
  echo "[marlowe] ASSISTANT_ROLE_VETO set -> DPO_SAVE_STEPS=0 (mid-epoch resume is unsafe)"
fi

# Force DPO_SAVE_STEPS=0 for the Gemma arms. A mid-DPO-epoch checkpoint resume
# restores the LR schedule but NOT the LoRA weights, silently zeroing them and
# under-training the adapter (see the memory marlowe-dpo-resume-corrupts-adapter,
# which invalidated the Qwen OEQ/AITA final arms). The 12B policy is slower than
# Qwen-8B and a run may exceed the 24h window, so a walltime overrun must restart
# from the last COMPLETED iteration (durable via responses_iter_*.jsonl), never
# mid-epoch. In the happy single-window path this changes nothing (checkpoints
# written and never read). Mirrors the Gemini body's pin.
if [[ -z "${DPO_SAVE_STEPS:-}" ]]; then
  DPO_SAVE_STEPS=0
  echo "[marlowe] gemma4: DPO_SAVE_STEPS=0 (mid-epoch resume corrupts the LoRA adapter)"
fi

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
  # REWRITE_CONTRACT=1 requires a single clean rewrite. Gemma-4 otherwise
  # answers the App G.2 instruction with a menu of three labelled rewrites on
  # 31.5% of OEQ posts, and the answering turn grades them instead of engaging
  # the situation. Off by default so the existing gemma arms are unchanged.
  [[ -n "${REWRITE_CONTRACT:-}" ]] && EXTRA_ARGS+=(--rewrite-output-contract)
  echo "[marlowe] perspective-shift sampling ON (assess=${PERSPSHIFT_ASSESS:-0} contract=${REWRITE_CONTRACT:-0})"
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
# ASSISTANT_ROLE_VETO=1 auto-REJECTS candidates that fail to answer the user as an
# AI assistant (message-as-text critique, in-story voice, third-person case
# study). Unlike ADDRESSEE_FILTER these become rejected exemplars, <= 3 pairs per
# prompt, off the normal cross-product. Mutually exclusive with FUTURE_USER.
# ADDITIVE and gated: no frozen `finaldata` wrapper sets this, so every paper-set
# arm is behaviorally unchanged. Mirrors the block in _marlowe_dpo_body.sh.
if [[ -n "${ASSISTANT_ROLE_VETO:-}" ]]; then
  EXTRA_ARGS+=(--assistant-role-veto)
  echo "[marlowe] assistant-role veto ON (auto-reject role-abandoning candidates, <=3 pairs/prompt)"
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
# USE_FINAL_DATA=0 trains on the ORIGINAL sampled split (--train-n/--test-n/
# --seed off the env CSV) instead of the frozen final_training_eval_data ids.
# Defaults to 1 so every existing gemma wrapper is byte-identical.
FINAL_DATA_ARGS=()
if [[ "${USE_FINAL_DATA:-1}" == "1" ]]; then
  FINAL_DATA_ARGS+=(--use-final-data)
  echo "[marlowe] data: FROZEN final_training_eval_data split (--use-final-data)"
else
  echo "[marlowe] data: ORIGINAL sampled split (--train-n/--test-n/--seed ARE used)"
fi
echo "[marlowe] split: train_n=${TRAIN_N:-1000} test_n=${TEST_N:-500}"
echo "[marlowe] num_iterations=${NUM_ITERATIONS:-3} (with --allow-preempt, resumes from highest finished adapter_iter_<k>)"

# Run the trainer in the background so the trap can fire while it runs.
"$PY" training_mitigations/train_dpo_lora.py \
    --env "$RUN_ENV" \
    --output-dir "$OUT_DIR" \
    --model-id "$MODEL_ID" \
    --cache-dir "$HF_HOME/hub" \
    --lora-target-modules "$LORA_TARGET_MODULES" \
    --allow-preempt \
    ${FINAL_DATA_ARGS[@]+"${FINAL_DATA_ARGS[@]}"} \
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
    --grad-accum 16 \
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
