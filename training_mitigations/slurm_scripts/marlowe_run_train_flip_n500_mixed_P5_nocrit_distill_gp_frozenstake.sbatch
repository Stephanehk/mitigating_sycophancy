#!/bin/bash
# Marlowe `preempt`-partition AITA-NTA-FLIP P5_nocrit_distill_gp_frozenstake run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_flip_n500_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
#
# EXACT frozenstake recipe (P5 veto, mixed 4 model / 0 critical / 6 stakeholder,
# --stakeholder-voice, RPO_ALPHA=1.0 distill, FROZEN_STAKEHOLDER=1) on the
# AITA-NTA-FLIP env. Like AITA, candidates are free-form on the BARE post (no
# output-format nudge); the YTA/NTA/UNCLEAR verdict is read from the prose at eval
# time by classify_aita_verdicts.py (see eval/eval_aita_label_heldout.py). Each
# dataset ROW is an (original, flipped) PAIR expanded to TWO independent training
# prompts; the eval metric is both-answers-NTA consistency across the pair.
#
# Split COUNTS PAIRS: TRAIN_N=500 / TEST_N=1000 to match the FLIP baselines
# (runs/baseline_AITA-NTA-FLIP/test_ids.json = 1000 pairs; sft_lora_flip_*_n500).
# Confirm the seed-0 pair split reproduces the baseline held-out set before
# comparing consistency head-to-head.
#SBATCH --job-name=dpo_flip_n500_mixed_P5_nocrit_distill_gp_frozenstake
#SBATCH --account=marlowe-m000254
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_flip_n500_mixed_P5_nocrit_distill_gp_frozenstake_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_flip_n500_mixed_P5_nocrit_distill_gp_frozenstake_%j.err

RUN_ENV=AITA-NTA-FLIP
OUT_NAME=dpo_lora_flip_qwen3-8b_n500_mixed_P5_nocrit_distill_gp_frozenstake
SELF_BASENAME=marlowe_run_train_flip_n500_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
TRAIN_N=500
TEST_N=1000
RPO_ALPHA=1.0
N_SOURCE_MODEL=4
N_SOURCE_CRITICAL=0
STAKEHOLDER_VOICE=1
FROZEN_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
