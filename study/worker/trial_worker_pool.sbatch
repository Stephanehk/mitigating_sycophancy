#!/bin/bash
#SBATCH --job-name=syco_worker_pool
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodes=1
#SBATCH --nodelist=next7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --time=8-08:00:00
#SBATCH --output=/next/u/stephhk/sycophancy/study/logs/worker_pool_%j.out
#SBATCH --error=/next/u/stephhk/sycophancy/study/logs/worker_pool_%j.out
#
# ONE vLLM engine plus four worker processes, for burst capacity.
#
#   source study/.supabase_env                 # URL + PUBLIC anon key
#   export STUDY_WORKER_TOKEN="$(cat study/.trial_token)"
#   sbatch --export=ALL,STUDY_WORKER_TOKEN,STUDY_SUPABASE_URL,STUDY_SUPABASE_ANON_KEY \
#          study/worker/trial_worker_pool.sbatch
#
# The workers claim jobs from Supabase directly — there is no study server any
# more. They carry the PUBLIC anon key plus the shared worker token; the anon key
# alone reaches no table and no raw queue function, so a compromise of this node
# leaks no participant data.
#
# WHY FOUR WORKERS AND NOT ONE. run_worker.py is serial — claim, generate,
# repeat — so a single worker never has more than one request in flight and
# vLLM's continuous batching has nothing to batch. Switching to vLLM without
# also running several workers changes nothing. Measured with one HF worker: a
# 30-job burst takes ~4.3 minutes at mean arm timings, and jobs past roughly
# #23 exceed JOB_DEADLINE_SECONDS outright when the arm mix skews to `ours`
# (18.2s each).
#
# WHY ONE ENGINE AND NOT FOUR. vLLM reserves its share of the card up front via
# gpu_memory_utilization. Four engines would each hold a quarter of the KV
# cache and batch nothing; one engine holding the card, fed by four clients, is
# what actually batches.
#
# A6000, NOT 2x A4000. Qwen3-8B bf16 is ~16.4GB and does not fit a 16GB a4000,
# and multi-a4000 is doubly unavailable here: vLLM's NCCL collectives hang on
# next5/next6, and as of 2026-07-28 plain cross-card copies there run at
# 0.10 GB/s against 23.4 GB/s on next7. See local_trial_worker.sbatch for the
# measurements and how to re-check.

set -euo pipefail

WORKERS="${STUDY_WORKER_POOL_SIZE:-4}"

REPO=/next/u/stephhk/sycophancy
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
mkdir -p "$REPO/study/logs"

# Cluster node DNS is unreliable; serve strictly from the pre-populated cache.
export HF_HOME=/next/u/stephhk/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

: "${STUDY_WORKER_TOKEN:?STUDY_WORKER_TOKEN must be exported (must match study_config.worker_token)}"
: "${STUDY_SUPABASE_URL:?STUDY_SUPABASE_URL must be exported}"
: "${STUDY_SUPABASE_ANON_KEY:?STUDY_SUPABASE_ANON_KEY must be exported (the PUBLIC key)}"
case "$STUDY_SUPABASE_ANON_KEY" in
    *secret*|*service_role*)
        echo "refusing to run: STUDY_SUPABASE_ANON_KEY looks like a SECRET key" >&2
        exit 1 ;;
esac

echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-unset} workers=$WORKERS queue=$STUDY_SUPABASE_URL"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

exec "$PY" "$REPO/study/worker/run_worker_pool.py" \
    --workers "$WORKERS" \
    --device cuda:0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --vllm-log "$REPO/study/logs/pool_vllm_${SLURM_JOB_ID}.log"
