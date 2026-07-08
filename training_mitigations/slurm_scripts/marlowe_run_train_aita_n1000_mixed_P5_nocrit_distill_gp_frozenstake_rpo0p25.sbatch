#!/bin/bash
#
# AUTO-PROMOTED ARM. Generated as the AITA counterpart of the OEQ sweep arm
# P5_nocrit_distill_gp_frozenstake_rpo0p25, so a recipe that clears the OEQ gates
# (action endorsement in (0.4,0.6) AND stakeholder role-play < 10%) can be trained
# on this env too. Derived from marlowe_run_train_aita_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch -- the
# per-env knobs (TRAIN_N/TEST_N) come from that parent unchanged; only
# --rpo-alpha and the veto prompt differ.
#
# alpha = 0.25. --rpo-alpha is the weight of the SFT/NLL term added to the DPO
# loss: how hard the policy is pushed to IMITATE the chosen responses. See
# core/self_distill_trainer.py and the OEQ sweep wrappers.
# Marlowe `preempt`-partition AITA P5_nocrit_distill_gp_frozenstake run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_aita_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25.sbatch
#
# EXACT frozenstake recipe (P5 veto, mixed 4 model / 0 critical / 6 stakeholder,
# --stakeholder-voice, RPO_ALPHA=1.0 distill, FROZEN_STAKEHOLDER=1) on the AITA
# binary-verdict env. AITA responses are free-form on the BARE post (no
# output-format nudge). The YTA/NTA/UNCLEAR verdict is read from the response
# prose at eval time by classify_aita_verdicts.py, not self-emitted (see
# eval/eval_aita_label_heldout.py + test_aita_bare_prompt.py).
#
# Split: TRAIN_N=1000 / TEST_N=1000 to match the AITA baselines
# (runs/baseline_AITA/test_ids.json = 1000; sft_lora_aita_*_n1000). AITA splits
# 50/50 balanced on is_asshole at --seed 0; confirm this reproduces the baseline
# held-out set before comparing FNR/FPR head-to-head.
#SBATCH --job-name=dpo_aita_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25
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
#SBATCH --output=/projects/m000254/logs/dpo_aita_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_aita_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25_%j.err

RUN_ENV=AITA
OUT_NAME=dpo_lora_aita_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25
SELF_BASENAME=marlowe_run_train_aita_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25.sbatch
TRAIN_N=1000
TEST_N=1000
RPO_ALPHA=0.25
N_SOURCE_MODEL=4
N_SOURCE_CRITICAL=0
STAKEHOLDER_VOICE=1
FROZEN_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
