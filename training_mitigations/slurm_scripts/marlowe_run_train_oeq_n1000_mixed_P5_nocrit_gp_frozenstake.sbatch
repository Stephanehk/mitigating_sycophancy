#!/bin/bash
# Marlowe `preempt`-partition OEQ P5_nocrit_gp_frozenstake run — NO DISTILL (alpha=0).
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_oeq_n1000_mixed_P5_nocrit_gp_frozenstake.sbatch
#
# ABLATION: identical to marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake
# except RPO_ALPHA is UNSET -> plain DPO, no SFT/NLL term on the chosen responses.
#
# WHY THIS ARM EXISTS: it is the alpha=0 endpoint of the frozenstake dose-response
# curve. Without it the frozenstake row is alpha in {0.25, 0.5, 1.0} and the
# hypothesis "the distill (SFT-on-chosen) term causes the stakeholder-voice collapse"
# cannot be FALSIFIED for this recipe -- only its slope measured. The perspshift rows
# have their alpha=0 arms (*_perspshift_gp); this gives frozenstake the same.
#
# Together with the sweep arms the OEQ curves become:
#   P6_perspshift : alpha 0 / 0.25 / 0.5 / 1.0
#   frozenstake   : alpha 0 / 0.25 / 0.5 / 1.0   <- this file supplies alpha=0
#   P5_perspshift : alpha 0 / 1.0
#
# CAVEAT: dropping the SFT term does NOT remove exposure to stakeholder exemplars.
# With N_SOURCE_MODEL=4 / N_SOURCE_CRITICAL=0, 6 of the 10 candidates per prompt are
# still stakeholder-voiced (from the FROZEN base teacher), and any can be selected as
# `chosen` in a DPO pair. DPO makes the model PREFER them; the SFT term makes it
# IMITATE them. If role-play persists at alpha=0, the cause is the candidate pool --
# shrink it with N_SOURCE_MODEL / N_SOURCE_CRITICAL (see _marlowe_dpo_body.sh).
#
# Training config: P5 veto, mixed sampling 4 model / 0 critical / 6 stakeholder.
#   (RPO_ALPHA unset)    plain DPO -- no distill/SFT-on-chosen term
#   STAKEHOLDER_VOICE=1  the 6 stakeholder candidates use the 2nd-person voice
#                        prompt instead of first-person role-play (the "_gp" change)
#   FROZEN_STAKEHOLDER=1 those candidates come from the FROZEN base model, not the
#                        current policy (the "_frozenstake" change)
# "nocrit": n_source_critical=0, no "Be critical." steer source.
#
# preempt partition: MaxTime=04:00:00 (4h), GraceTime=900 (15min), preemptible.
# SELF-RESUBMITTING 4h chain (the body queues an afterany successor each window),
# resumable via responses_iter_*.jsonl + --dpo-save-steps checkpoints. All site
# paths + the recipe knobs live in _marlowe_dpo_body.sh.
#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_nocrit_gp_frozenstake
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
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_gp_frozenstake_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_gp_frozenstake_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_gp_frozenstake
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P5_nocrit_gp_frozenstake.sbatch
# RPO_ALPHA intentionally UNSET -> plain DPO (no distill). Do not add it here.
N_SOURCE_MODEL=4
N_SOURCE_CRITICAL=0
STAKEHOLDER_VOICE=1
FROZEN_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
