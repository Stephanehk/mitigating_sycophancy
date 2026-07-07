#!/bin/bash
# Marlowe `preempt`-partition PAS P5_nocrit_distill_gp_frozenstake run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
#
# EXACT frozenstake recipe (P5 veto, mixed 4 model / 0 critical / 6 stakeholder,
# --stakeholder-voice second-person prompt, RPO_ALPHA=1.0 distill, and
# FROZEN_STAKEHOLDER=1 so the 6 stakeholder candidates come from the FROZEN BASE
# Qwen3-8B), on the PAS env. PAS is open-ended advice like OEQ, so the label
# machinery is inert here (label_suffix="" for PAS) — this is the OEQ frozenstake
# wrapper with RUN_ENV=PAS. Split: default 1000 train / 500 test.
#SBATCH --job-name=dpo_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake
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
#SBATCH --output=/projects/m000254/logs/dpo_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake
SELF_BASENAME=marlowe_run_train_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
RPO_ALPHA=1.0
N_SOURCE_MODEL=4
N_SOURCE_CRITICAL=0
STAKEHOLDER_VOICE=1
FROZEN_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
