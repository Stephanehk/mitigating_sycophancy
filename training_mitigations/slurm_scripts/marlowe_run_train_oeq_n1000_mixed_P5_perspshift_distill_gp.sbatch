#!/bin/bash
# Marlowe `preempt`-partition OEQ P5_perspshift_distill_gp run.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_oeq_n1000_mixed_P5_perspshift_distill_gp.sbatch
#
# Training config: P5 veto, mixed sampling 2 model / 2 perspshift / 6 stakeholder,
# with THREE interventions stacked (all wired via _marlowe_dpo_body.sh env knobs):
#   RPO_ALPHA=1.0        distill/SFT-on-chosen term (DPO + 1.0*mean-token CE(chosen))
#   PERSPSHIFT=1         the 2 "critical" steer candidates come from the ELEPHANT
#                        third-person rewrite of the post, not "Be critical."
#   PERSPSHIFT_ASSESS=1  OEQ-only: append "Assess this narrative." to the rewrite
#   STAKEHOLDER_VOICE=1  the 6 stakeholder candidates use the 2nd-person voice
#                        prompt instead of first-person role-play (the "_gp" change)
# The DPO prompt column stays the bare first-person post throughout. NOT "nocrit":
# n_source_critical stays 2 (now perspshift). See core/trainer.py.
#
# preempt partition: MaxTime=04:00:00 (4h), GraceTime=900 (15min), preemptible.
# SELF-RESUBMITTING 4h chain (the body queues an afterany successor each window),
# resumable via responses_iter_*.jsonl + --dpo-save-steps checkpoints. All site
# paths + the recipe knobs live in _marlowe_dpo_body.sh.
#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_perspshift_distill_gp
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
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_perspshift_distill_gp_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_perspshift_distill_gp_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_perspshift_distill_gp
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P5_perspshift_distill_gp.sbatch
RPO_ALPHA=1.0
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
