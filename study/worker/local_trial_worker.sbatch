#!/bin/bash
#SBATCH --job-name=syco_trial_worker
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --time=8-08:00:00
#SBATCH --output=/next/u/stephhk/sycophancy/study/logs/trial_worker_%j.out
#SBATCH --error=/next/u/stephhk/sycophancy/study/logs/trial_worker_%j.out
#
# Long-running worker for a LOCAL TRIAL of the study (not a Prolific launch).
# Runs until the wall clock expires or you scancel it.
#
#   export STUDY_WORKER_TOKEN=<same secret as the server>
#   sbatch --export=ALL,STUDY_WORKER_TOKEN \
#          study/worker/local_trial_worker.sbatch http://10.79.12.73:8750
#
# ONE a6000 ON next7, changed 2026-07-28 from `--gres=gpu:a4000:2`.
#
# 2x a4000 + `--device-map auto` IS a valid configuration and is this repo's
# established path — the smoke test passed on it, and it was serving this study
# correctly at 10:44 on 2026-07-28. Do not read this as "a4000s don't work".
#
# But from ~10:49 that day the inter-card path on next5 AND next6 degraded, and
# the failure is vicious: generation drops to 0.36 tok/s and emits "!!!!!!!!"
# or token-id garbage instead of text, so every job dies on the GEN_MAX_SECONDS
# cap reporting "the node is too contended" — which is NOT the cause and will
# send you chasing the wrong thing. On next6 it also raised
# `indexSelectSmallIndex: srcIndex < srcSelectDimSize` (out-of-range token ids).
#
# Measured with preallocated destinations and per-device sync, same script, same
# torch 2.11.0+cu130, on otherwise-idle cards:
#
#                        next5 (2x a4000)      next7 (2x a6000)
#   tiny-op latency      0.065 ms              0.055 ms      <- both fine
#   copy  8MB cross-GPU  78.95 ms (0.10 GB/s)  0.33 ms (23.37 GB/s)
#   copy 64MB cross-GPU  98.56 ms (0.63 GB/s)  2.79 ms (22.42 GB/s)
#
# Single-GPU compute is healthy on both; only the crossing is broken, and the
# cost is a flat ~85ms per copy regardless of size (a per-operation stall, not a
# bandwidth limit). Reproduced across four different physical a4000s on two
# nodes. `--device-map auto` shards BY LAYER, so every token pays it.
#
# Qwen3-8B bf16 is ~16.4GB and an a6000 has 48GB, so ONE card holds the whole
# model and there is no cross-GPU traffic to be slow. Verified on next7, all
# four arms: ours 18.2s, untrained 4.0s, neutral 8.1s, syco 4.0s.
#
# BEFORE going back to 2x a4000, re-measure — the degradation may well have been
# cleared. `study/logs/p2p_*.out` and `compute_*.out` have the method; healthy is
# >10 GB/s cross-GPU. If it is still ~0.1 GB/s, stay on a single 40GB+ card and
# report the node.

set -euo pipefail

SERVER_URL="${1:?usage: sbatch local_trial_worker.sbatch <server-url>}"

REPO=/next/u/stephhk/sycophancy
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
mkdir -p "$REPO/study/logs"

export HF_HOME=/next/u/stephhk/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

: "${STUDY_WORKER_TOKEN:?STUDY_WORKER_TOKEN must be exported (must match the server)}"

echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-unset} server=$SERVER_URL"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# No --max-jobs: this worker stays up and serves the browsing session.
exec "$PY" "$REPO/study/worker/run_worker.py" \
    --server-url "$SERVER_URL" \
    --backend hf \
    --device-map auto
