#!/bin/bash
# BATCH-partition retrain of OEQ P5_nocrit_distill_gp_frozenstake (the alpha=1
# frozenstake arm). Byte-identical recipe to
# marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
# EXCEPT it runs on the non-preemptible `batch` partition (24h) instead of
# `preempt` (4h). WHY: the original preempt run was silently ZEROED by the
# mid-epoch DPO-resume bug (adapter came back at ~init after a preempt restart
# at iters 2 and 3). batch never preempts mid-epoch, and this job is submitted
# with --export=ALL,DPO_SAVE_STEPS=0 so the DPO step is atomic (no mid-epoch
# checkpoint => the broken resume path is never taken) as a second safeguard.
# Submit:
#   export HF_TOKEN=...
#   sbatch --export=ALL,DPO_SAVE_STEPS=0 <this file>
#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake
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
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_BATCH.sbatch
RPO_ALPHA=1.0
N_SOURCE_MODEL=4
N_SOURCE_CRITICAL=0
STAKEHOLDER_VOICE=1
FROZEN_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
