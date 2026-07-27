#!/bin/bash
#
# Offline RPO+LoRA DPO of a FRESH Qwen3-32B on the reconstructed P6 pair set.
#
# Same objective/hyperparameters as the P6 rpo0p5 runs, offline path:
#   beta=0.1, rpo-alpha=0.5, 1 epoch, eff-batch 8 (1x grad-accum 8),
#   max_length 3072, LoRA r16/alpha32 on all 7 proj modules.
# Two differences from an in-repo P6 run, both intentional/necessary:
#   - base is Qwen/Qwen3-32B (not 8B); a fresh zero-init LoRA is created first
#     (make_fresh_lora.py) because train_dpo_offline_lora.py continues an adapter.
#   - pairs come from a static file (the reconstructed all-epochs pairs), not the
#     online sampler/veto loop.
# NOTE: the RPO/SFT term here is trl's UNMASKED `sft` over every chosen; the P6
# online run masked it to on-policy model-source samples. These pairs are
# mixed-source, so this anchors toward all chosen (a known deviation).
#
# Env in:
#   DD_ENV    : oeq | pas   (label only)
#   DD_PAIRS  : abs path to the cleaned {prompt,chosen,rejected} pair file
#   DD_OUTDIR : abs path to the NEW run dir (fresh adapter goes to <DD_OUTDIR>_init)
#
#SBATCH --job-name=dpo32b
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo32b_%x_%j.out
#SBATCH --error=/projects/m000254/logs/dpo32b_%x_%j.err

set -euo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
ENV_PREFIX=/projects/$PROJECT/envs/llm126
export HF_HOME=/projects/$PROJECT/hf_cache
export HF_HUB_OFFLINE=1                 # compute node is air-gapped; use cache only
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

: "${DD_ENV:?set DD_ENV=oeq|pas}"
: "${DD_PAIRS:?set DD_PAIRS=abs path to pair file}"
: "${DD_OUTDIR:?set DD_OUTDIR=abs path to new run dir}"

MODEL_ID=Qwen/Qwen3-32B
INIT_ADAPTER="${DD_OUTDIR}_init"

cd "$REPO"
PY="$ENV_PREFIX/bin/python"
echo "[dpo32b] env=$DD_ENV job=${SLURM_JOB_ID:-?} node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd, -

if [[ -f "$DD_OUTDIR/.complete" ]]; then
  echo "[dpo32b] $DD_OUTDIR already complete, skipping"; exit 0
fi
assert_pairs() { [[ -f "$DD_PAIRS" ]] || { echo "ERROR: no pair file $DD_PAIRS" >&2; exit 1; }; }
assert_pairs

# 1) Fresh zero-init LoRA on the 32B base (idempotent).
if [[ ! -f "$INIT_ADAPTER/adapter_config.json" ]]; then
  echo "[dpo32b] creating fresh LoRA adapter -> $INIT_ADAPTER"
  "$PY" training_mitigations/make_fresh_lora.py \
      --model-id "$MODEL_ID" --out-dir "$INIT_ADAPTER" --device cuda:0
else
  echo "[dpo32b] fresh adapter already present: $INIT_ADAPTER"
fi

# 2) Offline RPO DPO, continuing the fresh adapter (== training it from base).
# DD_DEVICE_MAP=auto shards the 32B across all GPUs in the job (needs --gpus>=2);
# unset => single-GPU. Model-parallel changes nothing about the objective.
"$PY" training_mitigations/train_dpo_offline_lora.py \
    --adapter "$INIT_ADAPTER" \
    --out-dir "$DD_OUTDIR" \
    --pairs-file "$DD_PAIRS" \
    --model-id "$MODEL_ID" \
    --beta 0.1 \
    --rpo-alpha 0.5 \
    --num-train-epochs 1 \
    ${DD_DEVICE_MAP:+--device-map "$DD_DEVICE_MAP"} \
    --device cuda:0 --seed 0

echo "[dpo32b] OK env=$DD_ENV end=$(date -Is)"
