#!/bin/bash
# Marlowe `preempt`-partition AITA P6_perspshift_distill_gp run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_aita_n1000_mixed_P6_perspshift_distill_gp.sbatch
#
# P6_perspshift_distill_gp recipe (mixed 2 model / 2 perspshift / 6 stakeholder,
# --stakeholder-voice, RPO_ALPHA=1.0 distill, P6 bidirectional veto +
# USER_STAKEHOLDER) on the AITA binary-verdict env. The trainer appends the
# YTA/NTA nudge (label_suffix) to ALL 10 candidate samples INCLUDING the 2
# perspshift candidates — whose third-person rewrite otherwise drops the nudge —
# and keeps it OUT of the rewrite/identify/concerns/veto/preference stages (see
# core/trainer.py + test_aita_label_suffix.py).
#
# PERSPSHIFT_ASSESS=1: AITA posts are long narratives (like OEQ), so the
# 'Assess this narrative.' addendum applies — UNLIKE PAS. Remove it if you want
# candidate generation byte-identical to the PAS perspshift arm.
#
# Split: TRAIN_N=1000 / TEST_N=1000 to match the AITA baselines
# (runs/baseline_AITA/test_ids.json = 1000). Balanced 50/50 on is_asshole at
# --seed 0; confirm it reproduces the baseline held-out set before comparing.
#SBATCH --job-name=dpo_aita_n1000_mixed_P6_perspshift_distill_gp
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
#SBATCH --output=/projects/m000254/logs/dpo_aita_n1000_mixed_P6_perspshift_distill_gp_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_aita_n1000_mixed_P6_perspshift_distill_gp_%j.err

RUN_ENV=AITA
OUT_NAME=dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp
SELF_BASENAME=marlowe_run_train_aita_n1000_mixed_P6_perspshift_distill_gp.sbatch
TRAIN_N=1000
TEST_N=1000
RPO_ALPHA=1.0
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
