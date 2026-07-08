#!/bin/bash
# Marlowe `preempt`-partition PAS P6_perspshift_distill_gp run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_pas_n1000_mixed_P6_perspshift_distill_gp.sbatch
#
# P6_perspshift_distill_gp recipe (mixed 2 model / 2 perspshift / 6 stakeholder,
# --stakeholder-voice, RPO_ALPHA=1.0 distill) with the P6 BIDIRECTIONAL veto +
# USER_STAKEHOLDER (4th narrator voice), on the PAS env. PAS is open-ended advice
# (label machinery inert, label_suffix=""). PAS does NOT set PERSPSHIFT_ASSESS
# (that OEQ-only addendum is for long narratives). Split: default 1000 / 500.
#SBATCH --job-name=dpo_pas_n1000_mixed_P6_perspshift_distill_gp
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
#SBATCH --output=/projects/m000254/logs/dpo_pas_n1000_mixed_P6_perspshift_distill_gp_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_pas_n1000_mixed_P6_perspshift_distill_gp_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp
SELF_BASENAME=marlowe_run_train_pas_n1000_mixed_P6_perspshift_distill_gp.sbatch
RPO_ALPHA=1.0
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
