# Shared body for the next7 GEMINI-SIM DPO wrappers
# (next7_train_{oeq,pas,aita}_P6_perspshift_gemini.sbatch source this).
#
# WHAT THIS IS: the P6_perspshift_distill_gp_rpo0p5_final recipe with ONE thing
# changed — the frozen sim is Gemini 3.1 Flash-Lite on the direct Google API
# instead of a local Qwen3-8B. Everything else (candidate mix, veto panel,
# RPO alpha, LoRA, frozen final data split) is identical to
# _marlowe_dpo_body_final.sh, so the arm is comparable to those runs in every
# respect except the teacher.
#
# NOT a drop-in comparison to the existing P6 numbers: the sim IS the training
# signal, so these runs need their own baseline. See FINAL_RUNS.md.
#
# DELTAS vs the Marlowe body, and why:
#   - sim on the API => NO sim model on the GPU. --sim-device cuda:1 still names
#     the GPU the vLLM PROCESS uses, because --policy-on-vllm serves the POLICY
#     there. Two GPUs are still required: cuda:0 trains, cuda:1 samples.
#   - DPO_SAVE_STEPS=0 (was 25). Mid-epoch resume restores the LR schedule but
#     NOT the LoRA weights, yielding an under-trained adapter. The next
#     partition has a 21-day limit and no preemption, so mid-epoch checkpoints
#     buy nothing here.
#   - --precompute-ref-batch-size 2 (was 8). 8 OOMs at the iter-2 reference
#     log-prob precompute on a 48GB A6000.
#   - NO self-resubmitting chain. That exists for Marlowe's 12h preempt
#     partition; next has neither constraint.
#   - --sim-concurrency 32: every sim fan-out (identify/filter/concerns and the
#     veto panel) is capped here, overriding --stage-concurrency (128, tuned for
#     a local vLLM server, which would rate-limit the API).
#
# COST: ~$39 OEQ / ~$30 PAS / ~$41 AITA (~$110 total) at $0.25/$1.50 per 1M with
# thinking OFF. SIM_MAX_COST defaults to ~1.5x the per-env estimate and aborts
# the run on breach; the last finished iteration's adapter stays loadable.

set -uo pipefail

# ============================ next7 SITE PATHS ============================
REPO=/next/u/stephhk/sycophancy
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
export HF_HOME=/next/u/stephhk/.cache/huggingface
RUNS_ROOT=$REPO/training_mitigations/runs
# =========================================================================

OUT_DIR="$RUNS_ROOT/$OUT_NAME"
mkdir -p "$HF_HOME" "$RUNS_ROOT" "$OUT_DIR"

if [[ -f "$OUT_DIR/.complete" ]]; then
  echo "[next7] $OUT_DIR/.complete exists — run already finished. Exiting."
  exit 0
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set; export it before sbatch (Qwen3-8B is gated)." >&2
  exit 1
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY not set; the frozen sim runs on the direct Gemini API." >&2
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
# Qwen3-8B is pre-cached; the sim needs the network but HF does not.
export HF_HUB_OFFLINE=1

cd "$REPO"
echo "[next7] env=$RUN_ENV jobid=${SLURM_JOB_ID:-?} node=$(hostname)"
echo "[next7] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[next7] out=$OUT_DIR  sim=gemini:${SIM_MODEL:-gemini-3.1-flash-lite}"

# Candidate-mix + veto knobs. Wrappers set these; defaults mirror the P6
# perspshift_distill_gp recipe so a wrapper only states its env-specific deltas.
EXTRA_ARGS=()
[[ -n "${RPO_ALPHA:-}" ]] && EXTRA_ARGS+=(--rpo-alpha "$RPO_ALPHA")
EXTRA_ARGS+=(--n-source-model "${N_SOURCE_MODEL:-2}" --n-source-critical "${N_SOURCE_CRITICAL:-2}")
[[ -n "${STAKEHOLDER_VOICE:-}" ]] && EXTRA_ARGS+=(--stakeholder-voice)
if [[ -n "${PERSPSHIFT:-}" ]]; then
  EXTRA_ARGS+=(--perspective-shift-sampling --rewrite-max-tokens "${REWRITE_MAX_TOKENS:-1536}")
  [[ -n "${PERSPSHIFT_ASSESS:-}" ]] && EXTRA_ARGS+=(--perspshift-assess)
fi
if [[ -n "${USER_STAKEHOLDER:-}" ]]; then
  [[ "${VETO_PROMPT:-P5}" == "P6" ]] || { echo "ERROR: USER_STAKEHOLDER=1 requires VETO_PROMPT=P6." >&2; exit 1; }
  EXTRA_ARGS+=(--user-stakeholder)
fi

echo "[next7] mix: model=${N_SOURCE_MODEL:-2} critical=${N_SOURCE_CRITICAL:-2} veto=${VETO_PROMPT:-P6} rpo=${RPO_ALPHA:-none}"
echo "[next7] sim: concurrency=${SIM_CONCURRENCY:-32} retries=${SIM_MAX_RETRIES:-5} backoff<=${SIM_BACKOFF_MAX:-300}s max_cost=\$${SIM_MAX_COST:?set SIM_MAX_COST}"

"$PY" training_mitigations/train_dpo_lora.py \
    --env "$RUN_ENV" \
    --output-dir "$OUT_DIR" \
    --allow-preempt \
    --use-final-data \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --veto-prompt "${VETO_PROMPT:-P6}" \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --sim-backend gemini \
    --sim-model "${SIM_MODEL:-gemini-3.1-flash-lite}" \
    --prompt-variant gemini \
    --sim-concurrency "${SIM_CONCURRENCY:-32}" \
    --sim-thinking-budget 0 \
    --sim-max-retries "${SIM_MAX_RETRIES:-5}" \
    --sim-backoff-max "${SIM_BACKOFF_MAX:-300}" \
    --sim-price-in 0.25 \
    --sim-price-out 1.50 \
    --sim-max-cost "$SIM_MAX_COST" \
    --veto-concurrency 32 \
    --precompute-ref-batch-size 2 \
    --dpo-save-steps 0 \
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
    --lora-dropout 0.05
rc=$?
echo "[next7] trainer exited rc=$rc"
[[ -f "$OUT_DIR/.complete" ]] && echo "[next7] run complete." || echo "[next7] NOT complete (rc=$rc)."
exit "$rc"
