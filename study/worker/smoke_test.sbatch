#!/bin/bash
#SBATCH --job-name=syco_study_smoke
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodes=1
#SBATCH --nodelist=next7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/next/u/stephhk/sycophancy/study/logs/smoke_%j.out
#SBATCH --error=/next/u/stephhk/sycophancy/study/logs/smoke_%j.out
#
# Self-contained GPU smoke test for the study's generation path.
#
# Starts its own study server on localhost, seeds one job per arm, runs the
# real worker against a real vLLM engine, and prints all four responses.
# Nothing outside this node is required, so it can sit in the queue and run
# unattended whenever a GPU frees.
#
#   sbatch study/worker/smoke_test.sbatch
#   # results land in study/logs/smoke_<jobid>.out
#
# ONE A6000 (48GB) and NO tensor parallelism is the intended configuration.
# Qwen3-8B in bf16 is ~16.4GB, which does not fit a 16GB a4000 (next5/next6);
# TP=2 across two a4000s was tried and FAILS on next6 — NCCL's first
# cross-rank collective (_ALLGATHER_BASE) hangs until the 600s watchdog kills
# the worker, both with and without NCCL_P2P_DISABLE. Treat multi-a4000 TP as
# unavailable on these nodes until someone fixes NCCL there.

set -euo pipefail

REPO=/next/u/stephhk/sycophancy
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
mkdir -p "$REPO/study/logs"

# Cluster node DNS is unreliable; serve strictly from the pre-populated cache.
export HF_HOME=/next/u/stephhk/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

exec "$PY" "$REPO/study/worker/smoke_test.py" \
    --device cuda:0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --enforce-eager \
    --vllm-log "$REPO/study/logs/smoke_vllm_${SLURM_JOB_ID}.log"
