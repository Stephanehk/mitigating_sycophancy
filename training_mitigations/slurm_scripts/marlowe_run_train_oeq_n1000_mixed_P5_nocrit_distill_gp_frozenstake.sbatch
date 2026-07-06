#!/bin/bash
# Marlowe `preempt`-partition OEQ P5_nocrit_distill_gp_frozenstake run.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/mitigating_sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
#
# EXACTLY the nocrit_distill_gp recipe (P5 veto, mixed 4 model / 0 critical / 6
# stakeholder, --stakeholder-voice second-person prompt, RPO_ALPHA=1.0 distill
# SFT-on-chosen term) with ONE change: FROZEN_STAKEHOLDER=1, so the 6 stakeholder
# candidates are generated from the FROZEN BASE Qwen3-8B (a fixed teacher, the
# base already served on the sim vLLM engine) instead of the evolving policy.
# This removes the on-policy feedback loop in the stakeholder exemplars while the
# bare-post "model" candidates stay on-policy. The stakeholder candidates are
# recorded as source "stakeholder-base:<name>" in responses_*.jsonl so the freeze
# is auditable. See core/trainer.py (_generate_n use_base / _sample_responses_mixed).
#
# preempt partition: MaxTime=04:00:00 (4h), GraceTime=900 (15min), preemptible.
# SELF-RESUBMITTING 4h chain (the body queues an afterany successor each window),
# resumable via responses_iter_*.jsonl + --dpo-save-steps checkpoints. All site
# paths + the recipe knobs live in _marlowe_dpo_body.sh.
#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake
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
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
RPO_ALPHA=1.0
N_SOURCE_MODEL=4
N_SOURCE_CRITICAL=0
STAKEHOLDER_VOICE=1
FROZEN_STAKEHOLDER=1
source /projects/m000254/mitigating_sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
