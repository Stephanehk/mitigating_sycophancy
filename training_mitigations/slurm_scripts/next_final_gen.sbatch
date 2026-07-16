#!/bin/bash
# NEXT-CLUSTER port of marlowe_final_gen.sbatch.
#
# FINAL-RUN held-out GENERATION over the FROZEN final_training_eval_data/<env>_test
# split (--use-final-data --final-split test -> the eval driver replaces
# <run_dir>/test_ids.json with final_training_eval_data/<env>_test_ids.json;
# 1000 ids for both OEQ and AITA). GPU generation ONLY — scoring is deferred to
# the login-node judge/classify stages, because the gpt-5-mini judge needs egress
# and next7's DNS is flaky.
#
# Shard regimes are NOT free parameters — each must match what the downstream
# scoring stage globs for:
#   OEQ / PAS     -> eval_endorsement_heldout.py --no-judge, NUM_SHARDS=2 (array 0-1).
#                    The endorse eval only appends _shard{i}of{N} when N>1, and
#                    stage_heldout_inputs.py reads responses_<tag>_shard{i}of2.csv.
#                    Writes eval/responses_<tag>_shard{i}of2.csv.
#   AITA / FLIP   -> eval_aita_label_heldout.py, NUM_SHARDS=1 (array 0). The AITA
#                    eval ALWAYS tags by shard (…_shard0of1), matching classify
#                    --num-shards 1. Needs --sample-max-tokens 1536 so verbose
#                    answers reach the verdict. Writes
#                    eval/aita_responses_<tag>_shard0of1.csv.
#
# Parametrized via --export:
#   GEN_ENV    : OEQ | PAS | AITA | AITA-NTA-FLIP
#   GEN_RUNDIR : abs run dir. trained -> the adapter dir (the eval driver defaults
#                --adapter to <run_dir>, i.e. the FINAL saved adapter, tag "final").
#                base -> a writable baseline container (outputs go to eval/).
#   GEN_KIND   : trained (default) | base
#   GEN_STEER  : base only -> none|becritical|explicit|sharma|dontbesyco|perspshift
#
# Idempotent: skips a shard whose responses csv already exists.
#
#SBATCH --job-name=fgen
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/next/u/stephhk/sycophancy/training_mitigations/logs/fgen_%A_%a.out
#SBATCH --error=/next/u/stephhk/sycophancy/training_mitigations/logs/fgen_%A_%a.err

source /next/u/stephhk/sycophancy/training_mitigations/slurm_scripts/_next_eval_env.sh
# eval_env sets -uo pipefail; add -e so a failed driver reports FAILED (no silent
# success echo with missing output).
set -e

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
    *) echo "ERROR: unknown GEN_STEER=$GEN_STEER" >&2; exit 1 ;;
  esac
else
  echo "ERROR: GEN_KIND must be trained|base (got $GEN_KIND)" >&2; exit 1
fi

TAG="${ADAPTER_TAG}_finaldata_temp1.0_seed0"
DONE_CSV="$GEN_RUNDIR/eval/${OUT_PREFIX}_${TAG}_shard${SHARD}of${NUM_SHARDS}.csv"
if [[ -f "$DONE_CSV" ]]; then
  echo "[fgen] shard $SHARD already done ($DONE_CSV) — skipping."
  exit 0
fi

echo "[fgen] env=$GEN_ENV kind=$GEN_KIND steer=$GEN_STEER tag=$TAG shard=$SHARD/$NUM_SHARDS run=$GEN_RUNDIR"
"$PY" "$DRIVER" \
    --run-dir "$GEN_RUNDIR" --env "$GEN_ENV" \
    --use-final-data --final-split test --tag-extra finaldata \
    --device cuda:0 --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$SHARD" \
    "${PATH_ARGS[@]}" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
echo "[fgen] shard $SHARD done -> $DONE_CSV"
