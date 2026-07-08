#!/bin/bash
# Marlowe `preempt`-partition AITA-NTA-FLIP mixed_P5 run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_flip_n500_mixed_P5.sbatch
#
# Plain mixed_P5 recipe (NO distill / gp / frozenstake / perspshift) on the
# AITA-NTA-FLIP env: mixed sampling 2 bare policy / 2 "Be critical." / 6
# stakeholder role-play, P5 two-stage veto. Like AITA, responses carry a parsed
# trailing YTA/NTA label; the trainer appends that nudge to ALL 10 candidate
# samples only (label_suffix_for(env); see core/trainer.py). Each dataset ROW is
# an (original, flipped) PAIR expanded to TWO independent training prompts; the
# eval metric is both-answers-NTA consistency across the pair.
#
# NB: mixed_P5 differs from mixed_P5_nocrit only in the mix: 2 model / 2 critical
# here vs 4 model / 0 critical there. No --stakeholder-voice, no RPO/distill, no
# frozen stakeholder.
#
# Split COUNTS PAIRS: TRAIN_N=500 / TEST_N=1000 to match the FLIP baselines
# (runs/baseline_AITA-NTA-FLIP/test_ids.json = 1000 pairs; sft_lora_flip_*_n500).
# Confirm the seed-0 pair split reproduces the baseline held-out set before
# comparing consistency head-to-head.
#SBATCH --job-name=dpo_flip_n500_mixed_P5
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
#SBATCH --output=/projects/m000254/logs/dpo_flip_n500_mixed_P5_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_flip_n500_mixed_P5_%j.err

RUN_ENV=AITA-NTA-FLIP
OUT_NAME=dpo_lora_flip_qwen3-8b_n500_mixed_P5
SELF_BASENAME=marlowe_run_train_flip_n500_mixed_P5.sbatch
TRAIN_N=500
TEST_N=1000
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
