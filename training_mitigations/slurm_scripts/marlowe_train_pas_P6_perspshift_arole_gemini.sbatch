#!/bin/bash
#
# dpo_pas_P6_perspshift_arole_gemini (MARLOWE): the P6_perspshift_distill_gp
# rpo0p5 recipe with the frozen sim served by Gemini 3.1 Flash-Lite (direct API)
# instead of a local Qwen3-8B, PLUS the assistant-role veto.
#
# The veto auto-REJECTS candidates that fail to answer the user AS an AI
# assistant (message-as-text critique, in-story voice, third-person case study),
# <= 3 such pairs per prompt, off the veto cross-product. On the Gemini judge the
# BASE assistant-role prompt is used: validated 2026-07-20 at recall 1.000 /
# FPR 0.125 on a held-out labelled set (the _QWEN variant is for the local judge
# only and costs Gemini recall).
#
# NOT comparable to the non-arole numbers, and not to the Qwen-sim arms: the sim
# IS the training signal, so this arm needs its own baseline.
#SBATCH --job-name=dpo_pas_P6_perspshift_arole_gemini
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_pas_P6_perspshift_arole_gemini_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_pas_P6_perspshift_arole_gemini_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_arole_gemini
RPO_ALPHA=0.5
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
ASSISTANT_ROLE_VETO=1
USER_STAKEHOLDER=1
TRAIN_N=1000
TEST_N=500
# Estimated sim spend $30 (base recipe) + ~$13 for the assistant-role veto
# (one extra classify call per candidate: 10 samples x 1000 prompts x 3 iters)
# = ~$43; cap set to ~1.5x that. The cap aborts the run on breach
# (fail-fast); the last finished iteration's adapter stays loadable.
SIM_MAX_COST=64
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body_gemini.sh
