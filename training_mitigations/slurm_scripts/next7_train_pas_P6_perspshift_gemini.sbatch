#!/bin/bash
#
# dpo_pas_P6_perspshift_gemini: P6_perspshift_distill_gp (alpha=0.5) on the FROZEN
# final_training_eval_data/pas_train split, with the frozen sim served by
# Gemini 3.1 Flash-Lite (direct API) instead of a local Qwen3-8B, running the
# retuned core/gemini_pipeline.py identify/filter prompts.
#
# Recipe is otherwise identical to
# marlowe_final_train_pas_P6_perspshift_distill_gp_rpo0p5.sbatch.
# NOT comparable to those runs' numbers: the sim IS the training signal, so this
# arm needs its own baseline.
#SBATCH --job-name=dpo_pas_P6_perspshift_gemini
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/next/u/stephhk/sycophancy/training_mitigations/logs/dpo_pas_P6_perspshift_gemini_%j.out
#SBATCH --error=/next/u/stephhk/sycophancy/training_mitigations/logs/dpo_pas_P6_perspshift_gemini_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_gemini
RPO_ALPHA=0.5
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
# PAS intentionally does NOT set PERSPSHIFT_ASSESS (matches the Marlowe arm)
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
TRAIN_N=1000
TEST_N=500
# Estimated sim spend $30; cap set to ~1.5x that. The cap aborts the run on
# breach (fail-fast); the last finished iteration's adapter stays loadable.
SIM_MAX_COST=46
source /next/u/stephhk/sycophancy/training_mitigations/slurm_scripts/_next7_dpo_body_gemini.sh
