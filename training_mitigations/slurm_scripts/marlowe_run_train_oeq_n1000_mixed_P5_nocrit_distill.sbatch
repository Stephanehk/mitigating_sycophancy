#!/bin/bash
# Marlowe `preempt`-partition OEQ P5_nocrit_distill run.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/mitigating_sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill.sbatch
#
# Training config: P5 veto, mixed sampling 4 model / 0 critical / 6 stakeholder,
# RPO_ALPHA=1.0 SFT-on-chosen term (total = DPO + 1.0 * mean-token CE(chosen)).
#
# preempt partition: MaxTime=04:00:00 (4h), GraceTime=900 (15min), preemptible.
# SELF-RESUBMITTING 4h chain (the body queues an afterany successor each window).
# This is viable for the multi-hour run ONLY because EVERY phase of an iteration
# is now resumable: per-prompt sampling/veto flush to responses_iter_*.jsonl, and
# the DPO train phase checkpoints every --dpo-save-steps (25) and resumes from the
# latest checkpoint (set in _marlowe_dpo_body.sh). So a 4h window that ends
# mid-train continues where it left off next window instead of restarting the
# iteration. All site paths live in _marlowe_dpo_body.sh.
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
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill.sbatch
RPO_ALPHA=1.0
source /projects/m000254/mitigating_sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
