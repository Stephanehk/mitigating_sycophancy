#!/bin/bash
# Marlowe `hero`-partition variant of the PAS P5_nocrit_distill run.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_hero_run_train_pas_n1000_mixed_P5_nocrit_distill.sbatch
#
# Same training config as the preempt variant (P5 veto, mixed 4 model / 0
# critical / 6 stakeholder, RPO_ALPHA=1.0 SFT-on-chosen term) but targets the
# `hero` partition (MaxTime=30 days, NOT preemptible). The whole 3-iteration run
# fits in one --time=1-00:00:00 window, so there is no 4h-window / unfinished-
# iteration risk that the preempt variant has.
#
# Reuses _marlowe_dpo_body.sh, so it still passes --allow-preempt (a harmless
# cold start on first run) and queues an afterany successor as a safety net; the
# successor is cancelled when the run writes .complete, which it will well within
# 24h. OUT_NAME matches the preempt variant, so this can also RESUME a run that
# was started under preempt.
#SBATCH --job-name=dpo_pas_n1000_mixed_P5_nocrit_distill_hero
#SBATCH --account=marlowe-m000254
#SBATCH --partition=hero
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_pas_n1000_mixed_P5_nocrit_distill_hero_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_pas_n1000_mixed_P5_nocrit_distill_hero_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit_distill
SELF_BASENAME=marlowe_hero_run_train_pas_n1000_mixed_P5_nocrit_distill.sbatch
RPO_ALPHA=1.0
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
