#!/bin/bash
#SBATCH --job-name=rmscore
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/m000254/logs/rmscore_%x_%j.out
#SBATCH --error=/projects/m000254/logs/rmscore_%x_%j.out
#
# Score one env's arms with one Skywork RM.
#   sbatch --export=ALL,RM=llama,SCORE_ENV=OEQ marlowe_final_rm_score.sbatch
#
# Compute nodes are AIR-GAPPED: the RM weights must already be in
# /projects/m000254/hf_cache (fetched on the login node). HF_HUB_OFFLINE=1 makes
# a cache miss fail loudly instead of hanging on a network retry.
set -euo pipefail

: "${RM:?set RM=llama|qwen}"
: "${SCORE_ENV:?set SCORE_ENV=OEQ|PAS|AITA}"

REPO=/projects/m000254/sycophancy
PY=/projects/m000254/envs/llm126/bin/python
export HF_HOME=/projects/m000254/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# ~/.local/lib/python3.10/site-packages shadows the env's transformers (same
# version today, but pin the env's copy so a stray `pip install --user` can't
# silently change what these jobs run).
export PYTHONNOUSERSITE=1

cd "$REPO"
echo "[rmscore] rm=$RM env=$SCORE_ENV node=$(hostname) start=$(date -Is)"
"$PY" training_mitigations/eval/score_reward_models.py \
    --rm "$RM" --env "$SCORE_ENV" --device cuda:0
echo "[rmscore] OK rm=$RM env=$SCORE_ENV end=$(date -Is)"
