#!/bin/bash
#
# FINAL-RUN wrapper: AITA P5_mixed trained on the FROZEN
# final_training_eval_data/aita_train split, held out on aita_test.
#
# Recipe = the plain P5_mixed arm (run_train_aita_n1000_mixed_P5.sbatch): the P5
# TWO-STAGE veto (Stage 1 = stakeholder-INDEPENDENT endorsement classifier
# ENDORSE_CLASSIFY_PROMPT_P5 run once per response; Stage 2 = each cached
# stakeholder vetoes only the responses Stage 1 flagged as endorsing), with
# MIXED sampling of the 10 candidates: 2 bare policy / 2 "Be critical." / 6
# round-robin over identified stakeholders. No perspective-shift, no RPO, no
# distillation, no frozen-stake, no user-stakeholder -- those are the P6/other
# arms. VETO_PROMPT=P5 is the body default but is set explicitly so the recipe
# is readable without cross-referencing the body.
#
# Differs from the tuning arm ONLY in:
#   - sources _marlowe_dpo_body_final.sh (adds --use-final-data)
#   - OUT_NAME suffixed _final so it never clobbers a tuning run dir
# --train-n/--test-n/--seed for the split are IGNORED under --use-final-data;
# they are set here only to mirror the P6 final wrappers.
#
# P5's veto is the FROZEN LOCAL stakeholder model (no OpenRouter egress), so
# unlike the neutral arms this trains fine on air-gapped Marlowe compute.
#
# SUBMIT WITH DPO_SAVE_STEPS=0 -- NOT OPTIONAL:
#   sbatch --export=ALL,DPO_SAVE_STEPS=0 <this script>
# A mid-DPO-epoch checkpoint resume silently returns the LoRA weights at ~init
# while the LR schedule/dataloader resume, under-training the run (it is what
# produced the *_final_UNDERTRAINED OEQ/AITA arms). DPO_SAVE_STEPS=0 makes the
# DPO step atomic, so no mid-epoch checkpoint exists and the broken resume path
# is never taken; an interruption restarts the iteration cleanly instead.
# Health check after iter 1: LoRA ||B||_F per adapter_iter_1/ should match the
# method's tuning run (a large negative gap = the bug). For AITA the reference
# EXISTS: compare adapter_iter_*/ against runs/dpo_lora_aita_qwen3-8b_n1000_mixed_P5
# (the tuning arm, 3 iters on disk). ||B|| inits to zero so it is the sensitive
# statistic; a clean final matches its tuning run within ~1% at every iteration.
#SBATCH --job-name=dpo_aita_mixed_P5_final
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
#SBATCH --output=/projects/m000254/logs/dpo_aita_mixed_P5_final_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_aita_mixed_P5_final_%j.err

RUN_ENV=AITA
OUT_NAME=dpo_lora_aita_qwen3-8b_n1000_mixed_P5_final
SELF_BASENAME=marlowe_final_train_aita_mixed_P5.sbatch
TRAIN_N=1000
TEST_N=1000
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
VETO_PROMPT=P5
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body_final.sh
