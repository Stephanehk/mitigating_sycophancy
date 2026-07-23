#!/bin/bash
#
# dpo_oeq_P6_perspshift_arole_gemini: P6_perspshift_distill_gp (alpha=0.5) on the FROZEN
# final_training_eval_data/oeq_train split, with the frozen sim served by
# Gemini 3.1 Flash-Lite (direct API) instead of a local Qwen3-8B, running the
# retuned core/gemini_pipeline.py identify/filter prompts.
#
# DELTA vs next7_train_oeq_P6_perspshift_gemini.sbatch: adds the
# assistant-role veto (ASSISTANT_ROLE_VETO=1) -- candidates that fail to answer
# the user AS an AI assistant (message-as-text critique, in-story voice,
# third-person case study) are auto-REJECTED into the DPO pairs, <= 3 such
# pairs per prompt, off the veto cross-product.
#
# Recipe is otherwise identical to
# marlowe_final_train_oeq_P6_perspshift_distill_gp_rpo0p5.sbatch.
# NOT comparable to those runs' numbers: the sim IS the training signal, so this
# arm needs its own baseline.
#SBATCH --job-name=dpo_oeq_P6_perspshift_arole_gemini
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
#SBATCH --output=/next/u/stephhk/sycophancy/training_mitigations/logs/dpo_oeq_P6_perspshift_arole_gemini_%j.out
#SBATCH --error=/next/u/stephhk/sycophancy/training_mitigations/logs/dpo_oeq_P6_perspshift_arole_gemini_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_arole_gemini
RPO_ALPHA=0.5
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
ASSISTANT_ROLE_VETO=1
USER_STAKEHOLDER=1
TRAIN_N=1000
TEST_N=500
# Estimated sim spend $39 (base recipe) + ~$17 for the assistant-role
# veto (10 samples x 1000 prompts x 3 iterations of extra classify calls)
# = ~$56; cap set to ~1.5x that. The cap aborts the run on
# breach (fail-fast); the last finished iteration's adapter stays loadable.
SIM_MAX_COST=84
source /next/u/stephhk/sycophancy/training_mitigations/slurm_scripts/_next7_dpo_body_gemini.sh
