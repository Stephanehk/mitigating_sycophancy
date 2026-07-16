# NEXT-CLUSTER port of _marlowe_dpo_body_final.sh (the FINAL-DATA training body).
#
# Same recipe, same trainer flags, same --use-final-data frozen split. The ONLY
# things that differ are site config and two cluster-specific corrections that do
# not change what gets trained:
#
#   1. SITE PATHS. repo root, conda env, HF cache, runs/ and logs/ roots all
#      repointed from /projects/m000254/... to the next cluster. HF_HUB_OFFLINE=1
#      because next7 has flaky outbound DNS (Qwen3-8B is fully pre-cached under
#      $HF_HOME). Nothing in this config needs egress: the P6 veto panel, the
#      stakeholder voices and the perspective-shift rewrite all run on the LOCAL
#      vLLM sim model on cuda:1 (the gpt-5-mini judge is an EVAL-time tool only).
#
#   2. NO SELF-RESUBMIT CHAIN. The Marlowe body chains <=12h jobs because Marlowe
#      Basic Access runs on a preemptible partition with a 12h cap. The `next`
#      partition has a 21-DAY limit and does not preempt, so the whole run fits in
#      one job and the chain is dead weight here. Worse, it was a live hazard: the
#      successor inherits the environment via --export=ALL, so DPO_SAVE_STEPS had
#      to survive every hop of the chain to stay 0 (see below). We drop the chain
#      and keep --allow-preempt, so if a job DOES die (node failure), simply
#      resubmitting the same script resumes from the highest finished
#      adapter_iter_<k> plus any partial sampling. The .complete guard makes
#      resubmission idempotent.
#
#   3. DPO_SAVE_STEPS DEFAULTS TO 0 HERE, not 25. This is the correctness fix the
#      whole re-run exists for. With save_steps>0 the HF Trainer writes mid-epoch
#      checkpoint-<step>/ dirs, and a job killed mid-DPO-epoch resumes from one via
#      a path that restores the LR schedule but NOT the LoRA weights — silently
#      producing an under-trained adapter. That is what invalidated the previous
#      OEQ and AITA runs. At 0, trainer.py sets save_strategy="no" and gates
#      resume_ckpt to None (core/trainer.py:1418, 1535), so the broken path is
#      UNREACHABLE rather than merely unused. Defaulting to 0 here means the run is
#      correct even if someone forgets the --export flag; keep passing
#      --export=ALL,DPO_SAVE_STEPS=0 anyway so both layers agree.
#      Cost of 0: a killed job redoes that iteration's DPO epoch from scratch
#      (sampling is still durable — it flushes per prompt to responses_iter_*.jsonl).
#      That is the correct trade: recompute is cheap, a silently under-trained
#      adapter is not.
#
#   4. --precompute-ref-batch-size 2, not 8. Purely a no-grad micro-batching knob
#      for TRL's reference-logp precompute (same reference logps, smaller chunks —
#      semantics-preserving, does NOT change the trained model). 8 is safe on
#      Marlowe's larger cards but reproducibly OOMs at ITERATION 2 on next7's 47 GB
#      A6000s: after iter-1 the policy emits longer responses, so the full-vocab
#      logits tensor (batch x 3072 x ~152k vocab, bf16, doubled by a contiguous
#      shift-copy) tips over. We run 3 iterations, so at 8 this run would die at
#      iter-2. batch 2 ~= 4 GiB/batch.
#
# The wrapper sets RUN_ENV / OUT_NAME (+ recipe knobs) before sourcing this.

set -uo pipefail

# ===================== NEXT-CLUSTER SITE CONFIG =====================
REPO=/next/u/stephhk/sycophancy
ENV_PREFIX=/next/u/stephhk/miniconda3/envs/llm
export HF_HOME=/next/u/stephhk/.cache/huggingface
RUNS_ROOT=$REPO/training_mitigations/runs
LOGS_ROOT=$REPO/training_mitigations/logs
# ====================================================================

OUT_DIR="$RUNS_ROOT/$OUT_NAME"
mkdir -p "$HF_HOME" "$RUNS_ROOT" "$LOGS_ROOT" "$OUT_DIR"

# Already finished? ($OUT_DIR/.complete is written by trainer.train() after the
# final adapter is saved.) Makes a resubmit a no-op instead of a clobber.
if [[ -f "$OUT_DIR/.complete" ]]; then
  echo "[next] $OUT_DIR/.complete exists — run already finished. Exiting."
  exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set; export it before sbatch (Qwen3-8B is gated)." >&2
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# next7 has flaky outbound DNS; every model this run touches is pre-cached.
export HF_HUB_OFFLINE=1
# Don't let vLLM's usage-stats ping block startup on a node with no egress.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# Use the env's interpreter DIRECTLY (conda activate does not reliably switch
# `python` on compute nodes); ignore ~/.local so imports can't be shadowed.
export PYTHONNOUSERSITE=1

cd "$REPO"
PY="$ENV_PREFIX/bin/python"

echo "[next] env=$RUN_ENV jobid=${SLURM_JOB_ID:-?} node=$(hostname)"
echo "[next] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[next] out=$OUT_DIR python=$PY"

# Stop the trainer cleanly if SLURM signals the wall limit (--signal=B:USR1@300).
# Progress is durable regardless: sampling flushes per prompt to
# responses_iter_*.jsonl. Leave .complete absent so a resubmit knows to resume.
on_signal() {
  echo "[next] grace signal (USR1/TERM) — stopping trainer."
  kill -TERM "${PY_PID:-}" 2>/dev/null || true
  wait "${PY_PID:-}" 2>/dev/null || true
  exit 0
}
trap on_signal USR1 TERM

# ---- recipe knobs (identical to the Marlowe body) ----
EXTRA_ARGS=()
if [[ -n "${RPO_ALPHA:-}" ]]; then
  EXTRA_ARGS+=(--rpo-alpha "$RPO_ALPHA")
  echo "[next] RPO/distill mode: --rpo-alpha $RPO_ALPHA"
fi
if [[ -n "${DISTILL_SELF_ONLY:-}" ]]; then
  EXTRA_ARGS+=(--distill-self-only --distill-max-per-prompt "${DISTILL_MAX_PER_PROMPT:-4}")
  echo "[next] targeted distill (self-source, <= ${DISTILL_MAX_PER_PROMPT:-4}/prompt) ON"
fi

N_SOURCE_MODEL="${N_SOURCE_MODEL:-4}"
N_SOURCE_CRITICAL="${N_SOURCE_CRITICAL:-0}"
EXTRA_ARGS+=(--n-source-model "$N_SOURCE_MODEL" --n-source-critical "$N_SOURCE_CRITICAL")
echo "[next] mix: n_source_model=$N_SOURCE_MODEL n_source_critical=$N_SOURCE_CRITICAL"
if [[ -n "${STAKEHOLDER_VOICE:-}" ]]; then
  EXTRA_ARGS+=(--stakeholder-voice)
  echo "[next] stakeholder-voice (gp) ON"
fi
if [[ -n "${FROZEN_STAKEHOLDER:-}" ]]; then
  EXTRA_ARGS+=(--frozen-stakeholder)
  echo "[next] frozen stakeholder (base teacher) ON"
fi
if [[ -n "${PERSPSHIFT:-}" ]]; then
  EXTRA_ARGS+=(--perspective-shift-sampling --rewrite-max-tokens "${REWRITE_MAX_TOKENS:-1536}")
  [[ -n "${PERSPSHIFT_ASSESS:-}" ]] && EXTRA_ARGS+=(--perspshift-assess)
  echo "[next] perspective-shift sampling ON (assess=${PERSPSHIFT_ASSESS:-0})"
fi
if [[ -n "${FUTURE_USER:-}" ]]; then
  EXTRA_ARGS+=(--future-user-sampling)
  echo "[next] future-user voice sampling ON (fills n_source_critical slots)"
fi
if [[ -n "${ADDRESSEE_FILTER:-}" ]]; then
  EXTRA_ARGS+=(--addressee-filter)
  echo "[next] addressee filter ON (drop message-critique candidates)"
fi

# USER_STAKEHOLDER=1 adds the 4th "user"/narrator veto voice. train_dpo_lora.py
# asserts this is P6-only; guard here for a clearer failure.
if [[ -n "${USER_STAKEHOLDER:-}" ]]; then
  [[ "${VETO_PROMPT:-P5}" == "P6" ]] || { echo "ERROR: USER_STAKEHOLDER=1 requires VETO_PROMPT=P6 (the 4th narrator voice is P6-only)." >&2; exit 1; }
  EXTRA_ARGS+=(--user-stakeholder)
  echo "[next] user-stakeholder (4th narrator/user veto voice) ON"
fi

# POLICY_DEVICE_MAP pipeline-shards the trainable policy across the listed CUDA
# indices (e.g. "0,2"), so the ~7GB full-vocab logits/CE spike sits on only one
# card instead of OOMing a single 47GB A6000 at --max-length 3072. Requires a
# 3-GPU alloc: the two policy shards plus the --sim-device (cuda:1) vLLM card,
# which must NOT be in the list. Unset -> single-device training on --device
# (the 2-GPU default). See train_dpo_lora.py --policy-device-map.
if [[ -n "${POLICY_DEVICE_MAP:-}" ]]; then
  EXTRA_ARGS+=(--policy-device-map "$POLICY_DEVICE_MAP")
  echo "[next] policy pipeline-MP across GPUs: $POLICY_DEVICE_MAP (sim stays on cuda:1)"
fi

# Guard the correctness invariant this whole re-run exists for. Anything other
# than 0 re-enables the mid-epoch checkpoint resume that silently under-trains the
# adapter. Fail loudly at submit time rather than discover it in the adapter.
DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-0}"
if [[ "$DPO_SAVE_STEPS" != "0" ]]; then
  echo "ERROR: DPO_SAVE_STEPS=$DPO_SAVE_STEPS but this final-run body requires 0." >&2
  echo "       save_steps>0 lets a job killed mid-DPO-epoch resume via a path that" >&2
  echo "       restores the LR schedule but NOT the LoRA weights -> under-trained" >&2
  echo "       adapter. Submit with --export=ALL,DPO_SAVE_STEPS=0." >&2
  exit 1
fi

echo "[next] split: --use-final-data (frozen final_training_eval_data/ split; --train-n/--test-n/--seed ignored)"
echo "[next] num_iterations=${NUM_ITERATIONS:-3}  dpo_save_steps=$DPO_SAVE_STEPS (no mid-epoch checkpoints)"

# Run in the background so the trap can fire while it runs.
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
    --precompute-ref-batch-size 2 \
    --dpo-save-steps "$DPO_SAVE_STEPS" \
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
echo "[next] trainer exited rc=$rc"

if [[ -f "$OUT_DIR/.complete" ]]; then
  echo "[next] run complete: $OUT_DIR"
else
  echo "[next] NOT complete (rc=$rc). Resubmit the same script to resume from the"
  echo "[next] highest finished adapter_iter_<k> (--allow-preempt)."
fi
exit "$rc"
