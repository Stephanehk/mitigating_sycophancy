#!/bin/bash
#
# GEMMA-4 OEQ P6_persp (alpha=0.5), RERUN FROM SCRATCH 2026-07-27, differing from
# marlowe_final_train_oeq_P6_perspshift_distill_gp_rpo0p5_gemma4.sbatch in exactly
# two ways:
#
#   1. REWRITE_CONTRACT=1 -- the perspective-shift rewriter must return ONE clean
#      rewrite. Without it Gemma-4 answers the App G.2 instruction with a menu
#      ("Option 1: Natural/Narrative ... Option 2: Concise ...") on 31.5% of OEQ
#      posts (measured, job 396378; Qwen3-8B: 2.5%), which flows into the
#      answering turn and makes the policy grade its own rewrites. With the
#      contract: 0.0% menus, rewrite/post length ratio p90 4.19 -> 1.13.
#   2. USE_FINAL_DATA=0 -- trains on the ORIGINAL sampled split
#      (--train-n/--test-n/--seed off the OEQ CSV), not the frozen
#      final_training_eval_data ids. NOTE: held-out results from this run are
#      therefore NOT comparable to any *_final arm; it needs its own baseline.
#
# Everything else (RPO alpha, source mix, veto prompt, user stakeholder,
# stakeholder voice, LoRA config) is identical to the final gemma OEQ wrapper.
# OUT_NAME is distinct so it can never clobber the existing run dir.
#
#SBATCH --job-name=dpo_oeq_P6_perspshift_gemma4_origdata
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_oeq_P6_perspshift_gemma4_origdata_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_P6_perspshift_gemma4_origdata_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_gemma4-12b-it_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_origdata_contract
SELF_BASENAME=marlowe_train_oeq_P6_perspshift_distill_gp_rpo0p5_gemma4_origdata.sbatch
RPO_ALPHA=0.5
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
REWRITE_CONTRACT=1
USE_FINAL_DATA=0
TRAIN_N=1000
TEST_N=500
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body_gemma4.sh
