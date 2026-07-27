#!/bin/bash
# FINAL-RUN held-out GENERATION on Marlowe over the FROZEN
# final_training_eval_data/<env>_TEST split (--use-final-data --final-split test).
# ONE parametrized job covers both trained adapters and the base+prompt-steer
# baselines, for all four envs. GPU generation only; scoring is deferred to the
# login-node judge/classify stages (marlowe_final_endorse_all.sh /
# marlowe_final_classify_all.sh) because Marlowe compute nodes are air-gapped.
#
# Two eval paths + shard regimes (kept consistent with each path's downstream):
#   OEQ / PAS -> eval_endorsement_heldout.py --no-judge, NUM_SHARDS=2 (array 0-1).
#                The endorse eval appends _shard{i}of{N} only when N>1, and
#                stage_heldout_inputs.py reads responses_<tag>_shard{i}of2.csv, so
#                OEQ/PAS MUST run with 2 shards. Writes responses_<tag>_shard*.csv.
#   AITA / FLIP -> eval_aita_label_heldout.py, NUM_SHARDS=1 (single task). The AITA
#                eval ALWAYS tags by shard (…_shard0of1), matching classify
#                --num-shards 1. Needs --sample-max-tokens 1536 so verbose answers
#                reach the verdict. Writes aita_responses_<tag>_shard0of1.csv.
#
# Parametrized via --export (set by marlowe_final_gen_submit.sh):
#   GEN_ENV    : OEQ | PAS | AITA | AITA-NTA-FLIP
#   GEN_RUNDIR : abs run dir. trained -> the adapter dir (adapter_config.json at
#                root); base -> a writable baseline container (outputs go to eval/).
#   GEN_KIND   : trained (default) | base
#   GEN_STEER  : base only -> none|becritical|explicit|sharma|dontbesyco|perspshift
#                (ignored for trained). Selects the prompt-only steer + tag suffix.
# The array is set per env by the submit driver (--array=0-1 for OEQ/PAS,
# --array=0 for AITA/FLIP). Idempotent: skips a shard whose responses csv exists.
#SBATCH --job-name=fgen
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/fgen_%A_%a.out
#SBATCH --error=/projects/m000254/logs/fgen_%A_%a.err

# Site paths + $PY + HF offline + cd $REPO (shared with the eval baselines).
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_eval_env.sh
# eval_env sets -uo pipefail; add -e so a failed driver reports FAILED (no silent
# success echo with missing output).
set -e

# GEN_BACKEND=gemma4 swaps the cu126/Qwen interpreter for the cu13 gemma4vllm env
# + CUDA-13 forward-compat libcuda, exactly as marlowe_voice2_stage_gemma4.sbatch
# does. Unset (the default) leaves every existing Qwen submission untouched.
CACHE_ARGS=()
if [[ "${GEN_BACKEND:-}" == "gemma4" ]]; then
  PY=/projects/m000254/envs/gemma4vllm/bin/python
  export LD_LIBRARY_PATH="/projects/m000254/cuda-compat-13/lib:${LD_LIBRARY_PATH:-}"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export TRANSFORMERS_OFFLINE=1
  # The gemma-4 weights are cached under $HF_HOME/hub, but the eval's --cache-dir
  # defaults to $HF_HOME itself, where an offline lookup fails (same bug fixed in
  # _marlowe_dpo_body_gemma4.sh and the voice2 train stage).
  CACHE_ARGS=(--cache-dir "$HF_HOME/hub")
fi

: "${GEN_ENV:?set GEN_ENV via --export}"
: "${GEN_RUNDIR:?set GEN_RUNDIR via --export}"
GEN_KIND="${GEN_KIND:-trained}"
GEN_STEER="${GEN_STEER:-none}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

# --- env -> driver / path-args / shard count / output prefix ------------------
case "$GEN_ENV" in
  OEQ|PAS)
    DRIVER=training_mitigations/eval/eval_endorsement_heldout.py
    PATH_ARGS=(--no-judge --sample-max-tokens 512)
    NUM_SHARDS="${GEN_NUM_SHARDS:-2}"
    OUT_PREFIX=responses ;;
  AITA)
    DRIVER=training_mitigations/eval/eval_aita_label_heldout.py
    PATH_ARGS=(--sample-max-tokens 1536)
    NUM_SHARDS="${GEN_NUM_SHARDS:-1}"
    OUT_PREFIX=aita_responses ;;
  AITA-NTA-FLIP)
    DRIVER=training_mitigations/eval/eval_aita_label_heldout.py
    PATH_ARGS=(--sample-max-tokens 1536)
    NUM_SHARDS="${GEN_NUM_SHARDS:-2}"
    OUT_PREFIX=aita_responses ;;
  *) echo "ERROR: unknown GEN_ENV=$GEN_ENV" >&2; exit 1 ;;
esac

# --- kind + steer -> model args + adapter_tag (must mirror the eval's tag) ----
MODEL_ARGS=()
if [[ "$GEN_KIND" == "trained" ]]; then
  ADAPTER_TAG=final
  [[ -f "$GEN_RUNDIR/adapter_config.json" ]] || { echo "ERROR: no final adapter in $GEN_RUNDIR" >&2; exit 1; }
elif [[ "$GEN_KIND" == "base" ]]; then
  mkdir -p "$GEN_RUNDIR/eval"
  MODEL_ARGS=(--base)
  case "$GEN_STEER" in
    none)       ADAPTER_TAG=base ;;
    becritical) MODEL_ARGS+=(--be-critical);        ADAPTER_TAG=base-becritical ;;
    explicit)   MODEL_ARGS+=(--explicit);           ADAPTER_TAG=base-explicit ;;
    sharma)     MODEL_ARGS+=(--sharma-honesty);     ADAPTER_TAG=base-sharma ;;
    dontbesyco) MODEL_ARGS+=(--gen-suffix "Don't be sycophantic." --gen-tag dontbesyco); ADAPTER_TAG=base-dontbesyco ;;
    perspshift) MODEL_ARGS+=(--perspective-shift --assess-narrative); ADAPTER_TAG=base-perspshiftassess ;;
    besyco)     MODEL_ARGS+=(--be-sycophantic);     ADAPTER_TAG=base-besycophantic ;;
    *) echo "ERROR: unknown GEN_STEER=$GEN_STEER" >&2; exit 1 ;;
  esac
else
  echo "ERROR: GEN_KIND must be trained|base (got $GEN_KIND)" >&2; exit 1
fi

TAG="${ADAPTER_TAG}_finaldata_temp1.0_seed0"

# --- gate resample mode (GEN_ATTEMPT>=2 + GEN_WORKLIST) -----------------------
# Regenerates ONLY the ids the assistant-mode gate rejected, reusing the very
# same steer flags resolved above. This is why resampling goes through this
# script instead of a hand-written command: marlowe_gate_round.sh warns that "a
# resample drawn from a different arm is not a resample", and the arm flags live
# here. The eval forces unsharded output into eval/gate/ and seeds per
# (prompt, attempt), so rounds are reproducible.
GATE_ARGS=()
if [[ -n "${GEN_ATTEMPT:-}" ]]; then
  : "${GEN_WORKLIST:?GEN_ATTEMPT requires GEN_WORKLIST (resample_<tag>_a<N>.json)}"
  [[ "$GEN_ATTEMPT" -ge 2 ]] || { echo "ERROR: GEN_ATTEMPT must be >= 2" >&2; exit 1; }
  [[ -f "$GEN_WORKLIST" ]] || { echo "ERROR: no worklist at $GEN_WORKLIST" >&2; exit 1; }
  NUM_SHARDS=1; SHARD=0
  GATE_ARGS=(--attempt "$GEN_ATTEMPT" --resample-ids-file "$GEN_WORKLIST")
  DONE_CSV="$GEN_RUNDIR/eval/gate/${OUT_PREFIX}_${TAG}_gateattempt${GEN_ATTEMPT}.csv"
else
  DONE_CSV="$GEN_RUNDIR/eval/${OUT_PREFIX}_${TAG}_shard${SHARD}of${NUM_SHARDS}.csv"
fi
if [[ -f "$DONE_CSV" ]]; then
  echo "[fgen] output already present ($DONE_CSV) — skipping."
  exit 0
fi

# GEN_MODEL overrides the policy model-id (default: the harness's 8B default, so
# existing 8B submissions are byte-for-byte unchanged). Set GEN_MODEL=Qwen/Qwen3-32B
# for the 32B sweep. Only added to the command line when set.
MODELID_ARGS=()
[[ -n "${GEN_MODEL:-}" ]] && MODELID_ARGS=(--model-id "$GEN_MODEL")
echo "[fgen] env=$GEN_ENV kind=$GEN_KIND steer=$GEN_STEER tag=$TAG shard=$SHARD/$NUM_SHARDS run=$GEN_RUNDIR model=${GEN_MODEL:-default8b} backend=${GEN_BACKEND:-qwen} py=$PY"
"$PY" "$DRIVER" \
    --run-dir "$GEN_RUNDIR" --env "$GEN_ENV" \
    --use-final-data --final-split test --tag-extra finaldata \
    --device cuda:0 --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$SHARD" \
    ${MODELID_ARGS[@]+"${MODELID_ARGS[@]}"} \
    ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} \
    ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} \
    "${PATH_ARGS[@]}" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
echo "[fgen] shard $SHARD done -> $DONE_CSV"
