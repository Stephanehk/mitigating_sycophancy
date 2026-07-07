#!/bin/bash
# Marlowe `preempt`-partition AITA-NTA-FLIP P6_perspshift_distill_gp run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/mitigating_sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_flip_n500_mixed_P6_perspshift_distill_gp.sbatch
#
# P6_perspshift_distill_gp recipe (mixed 2 model / 2 perspshift / 6 stakeholder,
# --stakeholder-voice, RPO_ALPHA=1.0 distill, P6 bidirectional veto +
# USER_STAKEHOLDER) on the AITA-NTA-FLIP env. The trainer appends the YTA/NTA
# nudge to ALL 10 candidate samples including the 2 perspshift candidates, and
# keeps it out of the rewrite/identify/concerns/veto/preference stages (see
# core/trainer.py + test_aita_label_suffix.py). Each row is an (original,
# flipped) PAIR -> two independent training prompts; eval metric = both-NTA
# consistency across the pair.
#
# PERSPSHIFT_ASSESS=1: FLIP posts are long narratives (like OEQ/AITA). Remove to
# match the PAS perspshift arm's candidate generation.
#
# Split COUNTS PAIRS: TRAIN_N=500 / TEST_N=1000 to match the FLIP baselines
# (runs/baseline_AITA-NTA-FLIP/test_ids.json = 1000 pairs; sft_lora_flip_*_n500).
# Confirm the seed-0 pair split reproduces the baseline held-out set.
#SBATCH --job-name=dpo_flip_n500_mixed_P6_perspshift_distill_gp
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
#SBATCH --output=/projects/m000254/logs/dpo_flip_n500_mixed_P6_perspshift_distill_gp_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_flip_n500_mixed_P6_perspshift_distill_gp_%j.err

RUN_ENV=AITA-NTA-FLIP
OUT_NAME=dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp
SELF_BASENAME=marlowe_run_train_flip_n500_mixed_P6_perspshift_distill_gp.sbatch
TRAIN_N=500
TEST_N=1000
RPO_ALPHA=1.0
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
source /projects/m000254/mitigating_sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
