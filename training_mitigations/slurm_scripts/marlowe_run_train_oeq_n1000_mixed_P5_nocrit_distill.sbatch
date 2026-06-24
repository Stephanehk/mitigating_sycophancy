#!/bin/bash
# Marlowe port of run_train_oeq_n1000_mixed_P5_nocrit_distill.sbatch.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill.sbatch
#
# DISTILL variant of marlowe_run_train_oeq_n1000_mixed_P5_nocrit.sbatch: identical
# in every way (P5 veto, mixed sampling 4 model / 0 critical / 6 stakeholder,
# seed, decode, DPO hyperparameters) EXCEPT it sets RPO_ALPHA, which makes the
# shared body pass --rpo-alpha to the trainer. That adds an SFT/NLL term on the
# CHOSEN (non-vetoed) responses to the DPO loss (total = DPO + RPO_ALPHA *
# mean-token CE(chosen)), anchoring chosen log-probs up instead of only the
# chosen-rejected margin. 1.0 is the canonical RPO value.
#
# `preempt` partition on Marlowe: MaxTime=04:00:00 (4h), GraceTime=900 (15min),
# preemptible. SELF-RESUBMITTING 4h chain; each (re)start auto-resumes via
# --allow-preempt. NOTE: a full DPO iteration may not fit in 4h, and the train
# step is not mid-checkpointed — prefer the `hero` partition variant
# (marlowe_hero_run_train_oeq_n1000_mixed_P5_nocrit_distill.sbatch) if you have
# access. All site paths live in _marlowe_dpo_body.sh.
#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_nocrit_distill
#SBATCH --account=marlowe-m000254
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill.sbatch
RPO_ALPHA=1.0
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
