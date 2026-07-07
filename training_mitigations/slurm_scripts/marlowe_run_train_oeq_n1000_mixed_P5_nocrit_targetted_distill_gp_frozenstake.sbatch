#!/bin/bash
# Marlowe `preempt`-partition OEQ P5_nocrit_targetted_distill_gp_frozenstake run.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_oeq_n1000_mixed_P5_nocrit_targetted_distill_gp_frozenstake.sbatch
#
# EXACTLY the nocrit_distill_gp_frozenstake recipe (P5 veto, mixed 4 model / 0
# critical / 6 stakeholder, --stakeholder-voice, FROZEN_STAKEHOLDER=1,
# RPO_ALPHA=1.0 distill SFT-on-chosen term) with ONE change: DISTILL_SELF_ONLY=1.
# That masks the RPO/distill SFT term so it distills ONLY the responses sampled
# on-policy from the model itself (source "model") that survived the veto, at
# most 4 distinct per prompt — instead of the whole veto-surviving pool (which
# includes the frozen-base stakeholder-voice exemplars). The DPO cross-pairs are
# UNCHANGED (all sources still become chosen/rejected), so only the distill
# target narrows. Motivation: the over-criticality appears with SFT-on-chosen,
# not plain DPO; distilling only the model's own accepted responses removes the
# critical-skewed imitation target. See core/self_distill_trainer.py.
# Writes to a distinct output dir so the parent frozenstake run is not clobbered.
#
# preempt partition: MaxTime=04:00:00 (4h), GraceTime=900 (15min), preemptible.
# SELF-RESUBMITTING 4h chain (the body queues an afterany successor each window),
# resumable via responses_iter_*.jsonl + --dpo-save-steps checkpoints. All site
# paths + the recipe knobs live in _marlowe_dpo_body.sh.
#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_nocrit_targetted_distill_gp_frozenstake
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
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_targetted_distill_gp_frozenstake_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_targetted_distill_gp_frozenstake_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_targetted_distill_gp_frozenstake
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P5_nocrit_targetted_distill_gp_frozenstake.sbatch
RPO_ALPHA=1.0
N_SOURCE_MODEL=4
N_SOURCE_CRITICAL=0
STAKEHOLDER_VOICE=1
FROZEN_STAKEHOLDER=1
DISTILL_SELF_ONLY=1
DISTILL_MAX_PER_PROMPT=4
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
