#!/bin/bash
# Marlowe port of run_train_pas_n1000_mixed_P5_nocrit.sbatch.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_pas_n1000_mixed_P5_nocrit.sbatch
#
# Basic Access => `preempt` partition: 12h cap, preemptible with a ~30-min grace
# signal. This is a SELF-RESUBMITTING 12h chain; each (re)start auto-resumes via
# --allow-preempt. All site paths live in _marlowe_dpo_body.sh.
#SBATCH --job-name=dpo_pas_n1000_mixed_P5_nocrit
#SBATCH --account=marlowe-m000254
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=11:55:00
#SBATCH --signal=B:USR1@1800
#SBATCH --open-mode=append
#SBATCH --output=/scratch/m000254/logs/dpo_pas_n1000_mixed_P5_nocrit_%j.out
#SBATCH --error=/scratch/m000254/logs/dpo_pas_n1000_mixed_P5_nocrit_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit
SELF_BASENAME=marlowe_run_train_pas_n1000_mixed_P5_nocrit.sbatch
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
