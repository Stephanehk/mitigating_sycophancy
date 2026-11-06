#!/bin/bash
# Cross-dataset held-out GENERATION for the four Qwen3-8B "Our Model (3+1)"
# adapters (rebuttal, 2026-11-06): each adapter is evaluated on the three envs
# it was NOT trained on -> 12 (train_env, eval_env) conditions.
#
# Generation only (attempt 1). No API calls: the assistant-mode gate, the gate
# loop resample, and the endorse judge / verdict classifier are deferred (see
# REBUTTAL_XENV_MMLU.md).
#
# WHY A SEPARATE CONTAINER PER CONDITION. The gen sbatch writes to
# <GEN_RUNDIR>/eval/<prefix>_final_finaldata_temp1.0_seed0_shard*.csv -- the tag
# does not encode the eval env, and the sbatch skips a shard whose csv exists.
# Pointing it at the adapter's own run dir would therefore collide with (and
# silently skip on) that run's in-domain outputs. Each condition instead gets
#   <RUNS>/xenv_qwen3-8b_<train>_on_<eval>/
# holding SYMLINKS to the source adapter's adapter_config.json +
# adapter_model.safetensors (all the eval reads from the run dir; the tokenizer
# comes from the base model id). The adapter tag stays "final", so downstream
# tools see the same tag layout as the in-domain arms. Source runs are untouched.
# NOTE: container names contain two env tokens, so always pass --env explicitly
# to any downstream tool (name-based env inference fails loud on these).
#
# CLUSTER (env var, default marlowe):
#   marlowe -> marlowe_final_gen.sbatch on the PAID association + preempt
#              (the free account's CPU-minute allocation was 0 on 2026-11-06).
#              Preempt is PreemptMode=REQUEUE (a preempted task restarts from
#              scratch), so shards are small (~40 min on an H100, measured from
#              past fgen runs: 500 OEQ prompts ~2.5h, 250 FLIP pairs ~3.3h).
#   next    -> next_final_gen.sbatch (its own #SBATCH header: account/partition
#              next, next7, A6000; expect roughly 2x the H100 time per shard).
# Override paths with XENV_RUNS (where containers are created) and
# XENV_ADAPTER_ROOT (where the four source adapter dirs live; default XENV_RUNS).
#
# Shards: OEQ/PAS 8, AITA 10, FLIP 20 (AITA/FLIP match the in-domain arms'
# layout) -> 138 array tasks. Downstream: pass the matching --num-shards
# (stage_heldout_inputs.py, classify_aita_verdicts.py,
# build_gated_finaldata_tables.py all take it).
#
# Usage (login node):  CLUSTER=next bash submit_xenv_qwen3_gen.sh [--dry-run]
set -euo pipefail

CLUSTER="${CLUSTER:-marlowe}"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
case "$CLUSTER" in
  marlowe)
    REPO=/projects/m000254/mitigating_sycophancy
    RUNS="${XENV_RUNS:-/projects/m000254/runs}"
    GENJOB=$REPO/training_mitigations/slurm_scripts/marlowe_final_gen.sbatch
    SB_ARGS=(--account=marlowe-m000254 --qos=normal --partition=preempt --time=03:00:00) ;;
  next)
    REPO=/next/u/stephhk/sycophancy
    RUNS="${XENV_RUNS:-$REPO/training_mitigations/runs}"
    GENJOB=$REPO/training_mitigations/slurm_scripts/next_final_gen.sbatch
    mkdir -p "$REPO/training_mitigations/logs"   # the sbatch's --output dir
    SB_ARGS=() ;;                                # header already sets account/partition/time
  *) echo "ERROR: CLUSTER must be marlowe|next (got $CLUSTER)" >&2; exit 1 ;;
esac
ADAPTER_ROOT="${XENV_ADAPTER_ROOT:-$RUNS}"

declare -A SRC=(
  [oeq]=dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5
  [pas]=dpo_lora_pas_qwen3-8b_P6_voice2_rpo0p5
  [aita]=dpo_lora_aita_qwen3-8b_P6_voice2R2_rpo0p5
  [flip]=dpo_lora_flip_qwen3-8b_P6_voice2R2_rpo0p5
)
declare -A ENVNAME=([oeq]=OEQ [pas]=PAS [aita]=AITA [flip]=AITA-NTA-FLIP)
declare -A NSH=([oeq]=8 [pas]=8 [aita]=10 [flip]=20)
declare -A PREFIX=([oeq]=responses [pas]=responses [aita]=aita_responses [flip]=aita_responses)
TAG=final_finaldata_temp1.0_seed0

echo "[xenv] cluster=$CLUSTER runs=$RUNS adapters=$ADAPTER_ROOT genjob=$GENJOB"
for tr in oeq pas aita flip; do
  src="$ADAPTER_ROOT/${SRC[$tr]}"
  [[ -f "$src/adapter_config.json" && -f "$src/adapter_model.safetensors" ]] \
    || { echo "ERROR: missing adapter in $src" >&2; exit 1; }
  for ev in oeq pas aita flip; do
    [[ "$tr" == "$ev" ]] && continue
    name="xenv_qwen3-8b_${tr}_on_${ev}"
    dir="$RUNS/$name"; n=${NSH[$ev]}

    # skip if every shard csv already exists
    done=1
    for ((s=0; s<n; s++)); do
      [[ -f "$dir/eval/${PREFIX[$ev]}_${TAG}_shard${s}of${n}.csv" ]] || done=0
    done
    if (( done )); then echo "[skip] $name: all $n shards on disk"; continue; fi
    if squeue -u "$USER" -h -o "%j" | grep -qxF "xg_${tr}_on_${ev}"; then
      echo "[skip] $name: already queued"; continue
    fi

    if (( DRY )); then
      echo "[dry] $name  env=${ENVNAME[$ev]}  shards=$n  adapter=$src"; continue
    fi
    mkdir -p "$dir/eval"
    ln -sfn "$src/adapter_config.json"       "$dir/adapter_config.json"
    ln -sfn "$src/adapter_model.safetensors" "$dir/adapter_model.safetensors"
    printf '%s\n' "train_env=$tr" "eval_env=${ENVNAME[$ev]}" "source_adapter=$src" \
      "cluster=$CLUSTER" "created=$(date -Is)" > "$dir/XENV_SOURCE.txt"

    sbatch "${SB_ARGS[@]}" --job-name="xg_${tr}_on_${ev}" --array="0-$((n-1))" \
      --export=ALL,GEN_ENV="${ENVNAME[$ev]}",GEN_RUNDIR="$dir",GEN_KIND=trained,GEN_NUM_SHARDS="$n" \
      "$GENJOB"
  done
done
