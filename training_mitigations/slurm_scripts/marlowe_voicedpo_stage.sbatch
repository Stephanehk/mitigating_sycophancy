#!/bin/bash
# GPU stages of the filtered-pair DPO round (voice-DPO arm).
#
# ONE parametrized job covering the three GPU stages, selected by VD_STAGE:
#   sample   -> eval/sample_train_candidates.py : N=10 candidates per TRAIN
#               prompt from the trained P6 adapter, via vLLM. ~1-2 h/env.
#   classify -> eval/classify_failure_mode.py   : label each 0/1/2 with the
#               Qwen3-8B BASE model (digit-likelihood scoring). ~30-60 min/env.
#   train    -> train_dpo_offline_lora.py       : one vanilla DPO epoch on the
#               precomputed pairs, continuing the P6 adapter into a NEW dir.
#
# Stage 3 (build_dpo_pairs.py) is pure CPU/IO and runs inline in the driver.
#
# Compute nodes are AIR-GAPPED, which is why the classifier is a local Qwen3-8B
# rather than the house gpt-5-mini judge, and why every gpt-5-mini judge stage
# is deferred to the login-node driver (run_voicedpo_all.sh).
#
# Parametrized via --export (set by run_voicedpo_all.sh):
#   VD_STAGE   : sample | classify | train
#   VD_ENV     : OEQ | PAS
#   VD_ADAPTER : abs path to the trained P6 adapter to sample from / continue
#   VD_OUTDIR  : abs path to the NEW run dir (never the adapter dir)
#
# Idempotent: sample and classify resume from their partial output files; train
# skips if the output adapter already carries its .complete sentinel.
#SBATCH --job-name=voicedpo
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/m000254/logs/voicedpo_%x_%j.out
#SBATCH --error=/projects/m000254/logs/voicedpo_%x_%j.out

set -euo pipefail

: "${VD_STAGE:?set VD_STAGE=sample|classify|train}"
: "${VD_ENV:?set VD_ENV=OEQ|PAS}"
: "${VD_ADAPTER:?set VD_ADAPTER=/abs/path/to/p6/adapter}"
: "${VD_OUTDIR:?set VD_OUTDIR=/abs/path/to/new/run/dir}"

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
ENV_PREFIX=/projects/$PROJECT/envs/llm126
export HF_HOME=/projects/$PROJECT/hf_cache
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
PY="$ENV_PREFIX/bin/python"

echo "[voicedpo] stage=$VD_STAGE env=$VD_ENV jobid=${SLURM_JOB_ID:-?} node=$(hostname) start=$(date -Is)"
echo "[voicedpo] beta=${VD_BETA:-0.1}"
echo "[voicedpo] adapter=$VD_ADAPTER"
echo "[voicedpo] outdir=$VD_OUTDIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# PREFLIGHT: the batch partition shares nodes, and an orphaned server from
# another job can sit on the GPU we were allocated (observed: a 70 GB sglang
# process on n06 surviving its job by 36 h). vLLM would then take minutes to
# fail deep in engine startup. Fail in seconds instead, so the driver's retry
# can resubmit with this node excluded.
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [[ -n "${FREE_MIB:-}" ]] && (( FREE_MIB < MIN_FREE_MIB )); then
  echo "[voicedpo] PREFLIGHT FAIL on $(hostname): only ${FREE_MIB} MiB free on the" >&2
  echo "[voicedpo] allocated GPU (need >= ${MIN_FREE_MIB} MiB for Qwen3-8B)." >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2 || true
  exit 75
fi
echo "[voicedpo] preflight OK: ${FREE_MIB:-?} MiB free"

mkdir -p "$VD_OUTDIR"

case "$VD_STAGE" in
  sample)
    "$PY" training_mitigations/eval/sample_train_candidates.py \
        --env "$VD_ENV" \
        --adapter "$VD_ADAPTER" \
        --out-dir "$VD_OUTDIR" \
        --n 10 --temperature 1.0 --top-p 0.95 \
        --max-tokens 1536 --device cuda:0 --seed 0
    ;;
  classify)
    "$PY" training_mitigations/eval/classify_failure_mode.py \
        --out-dir "$VD_OUTDIR" --device cuda:0
    ;;
  train)
    if [[ -f "$VD_OUTDIR/.complete" ]]; then
      echo "[voicedpo] adapter already complete, skipping train"
      exit 0
    fi
    # Vanilla DPO: no --rpo-alpha, and no save_strategy=steps anywhere, so the
    # mid-epoch checkpoint-resume path that silently zeroes LoRA weights is
    # unreachable rather than merely unused (_next_dpo_body_final.sh:25-34).
    # VD_BETA: DPO KL strength. Default 0.1 matches the trainer default and the
    # iterative P6 runs. A second-stage round continues an ALREADY-CONVERGED
    # adapter, so 0.1 is loose -- the beta=0.1 PAS arm drove rewards/chosen to
    # -8.0 and emitted degenerate "you're not the one who" loops in 9.6% of
    # held-out responses (absent from its training pairs at 1.2%, so it was an
    # optimisation artifact, not data). Raise beta to constrain drift.
    "$PY" training_mitigations/train_dpo_offline_lora.py \
        --adapter "$VD_ADAPTER" \
        --out-dir "$VD_OUTDIR" \
        --beta "${VD_BETA:-0.1}" \
        --device cuda:0 --seed 0
    ;;
  *)
    echo "ERROR: unknown VD_STAGE=$VD_STAGE" >&2; exit 1 ;;
esac

echo "[voicedpo] OK stage=$VD_STAGE env=$VD_ENV end=$(date -Is)"
