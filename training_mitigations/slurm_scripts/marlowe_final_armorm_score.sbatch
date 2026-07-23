#!/bin/bash
#SBATCH --job-name=armorm
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/m000254/logs/armorm_%x_%j.out
#SBATCH --error=/projects/m000254/logs/armorm_%x_%j.out
#
# Score one env's arms with ArmoRM-Llama3-8B-v0.1 (multi-objective RM).
#   sbatch --export=ALL,SCORE_ENV=OEQ marlowe_final_armorm_score.sbatch
#
# Compute nodes are AIR-GAPPED: the RM weights AND the trust_remote_code
# modeling_custom.py must already be in /projects/m000254/hf_cache (fetched on the
# login node). HF_HUB_OFFLINE=1 makes a cache miss fail loudly. The dynamic-module
# cache under $HF_HOME/modules is regenerated from the local snapshot, no network.
set -euo pipefail

: "${SCORE_ENV:?set SCORE_ENV=OEQ|PAS|AITA}"

REPO=/projects/m000254/sycophancy
# Dedicated env: ArmoRM's custom code needs transformers 4.43.x (LLAMA_INPUTS_DOCSTRING),
# which the newer llm126/ramp envs no longer provide.
PY=/projects/m000254/envs/armorm/bin/python
export HF_HOME=/projects/m000254/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Do NOT let ~/.local/site-packages shadow the pinned env transformers.
export PYTHONNOUSERSITE=1

cd "$REPO"
echo "[armorm] env=$SCORE_ENV node=$(hostname) start=$(date -Is)"
"$PY" training_mitigations/eval/score_armorm.py --env "$SCORE_ENV" --device cuda:0
echo "[armorm] OK env=$SCORE_ENV end=$(date -Is)"
