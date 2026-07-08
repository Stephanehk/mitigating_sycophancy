#!/bin/bash
# Marlowe `preempt`-partition AITA mixed_P5 run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_aita_n1000_mixed_P5.sbatch
#
# Plain mixed_P5 recipe (NO distill / gp / frozenstake / perspshift) on the AITA
# binary-verdict env: mixed sampling 2 bare policy / 2 "Be critical." / 6
# stakeholder role-play, P5 two-stage veto (stakeholder-independent endorsement
# classifier -> per-stakeholder veto). AITA responses are free-form and parsed
# for a trailing YTA/NTA label; the trainer appends that nudge (label_suffix) to
# ALL 10 candidate samples but keeps it OUT of the identify/concerns/veto stages
# automatically (label_suffix_for(env); see core/trainer.py). There is no
# perspshift source here, so every candidate carries the nudge natively.
#
# NB: mixed_P5 differs from mixed_P5_nocrit only in the mix: 2 model / 2 critical
# here vs 4 model / 0 critical there. No --stakeholder-voice (that is the "gp"
# second-person variant), no RPO/distill, no frozen stakeholder.
#
# Split: TRAIN_N=1000 / TEST_N=1000 to match the AITA baselines
# (runs/baseline_AITA/test_ids.json = 1000; sft_lora_aita_*_n1000). AITA splits
# 50/50 balanced on is_asshole at --seed 0; confirm this reproduces the baseline
# held-out set before comparing FNR/FPR head-to-head.
#SBATCH --job-name=dpo_aita_n1000_mixed_P5
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
#SBATCH --output=/projects/m000254/logs/dpo_aita_n1000_mixed_P5_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_aita_n1000_mixed_P5_%j.err

RUN_ENV=AITA
OUT_NAME=dpo_lora_aita_qwen3-8b_n1000_mixed_P5
SELF_BASENAME=marlowe_run_train_aita_n1000_mixed_P5.sbatch
TRAIN_N=1000
TEST_N=1000
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
