#!/bin/bash
#SBATCH --job-name=syco_study_worker
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/next/u/stephhk/sycophancy/study/logs/worker_%j.out
#SBATCH --error=/next/u/stephhk/sycophancy/study/logs/worker_%j.err
#
# GPU worker for the sycophancy user study.
#
# Launches one vLLM engine (Qwen3-8B + our OEQ LoRA) and polls the public
# study server for generation jobs. The node needs OUTBOUND https only.
#
# Submit with sbatch (NOT srun inside an salloc): an interactive allocation is
# torn down when the ssh session drops, which would kill the study mid-run.
#
#   export STUDY_WORKER_TOKEN=<the same secret the server has>
#   sbatch --export=ALL study/worker/study_worker.sbatch https://<study-host>
#
# --nodes=1 with no --nodelist on purpose: pinning to a node list forces
# multi-node allocations for a single-process task and wastes GPUs.

set -euo pipefail

SERVER_URL="${1:?usage: sbatch study_worker.sbatch <server-url>}"

REPO=/next/u/stephhk/sycophancy
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

mkdir -p "$REPO/study/logs"

# The cluster's node DNS is unreliable; serve strictly from the pre-populated
# hub cache so a lookup failure cannot stall the engine at load time.
export HF_HOME="${HF_HOME:-/next/u/stephhk/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export STUDY_ENV="${STUDY_ENV:-production}"

: "${STUDY_WORKER_TOKEN:?STUDY_WORKER_TOKEN must be exported (must match the server)}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} server=$SERVER_URL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

exec "$PY" "$REPO/study/worker/run_worker.py" \
    --server-url "$SERVER_URL" \
    --device cuda:0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --vllm-log "$REPO/study/logs/vllm_${SLURM_JOB_ID:-local}.log"
