#!/bin/bash
#
# RPO-ALPHA SWEEP ARM (alpha=0.5, baseline arm uses 1.0).
# Identical to marlowe_run_train_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch in every respect except
# RPO_ALPHA. `--rpo-alpha` is the WEIGHT of the SFT/NLL term added to the DPO loss
# (trl multi-loss `loss_weights=[1.0, rpo_alpha]`), i.e. how hard the model is pushed
# to IMITATE the `chosen` responses. Lowering it partially withdraws that pressure.
#
# WHY: the distill (SFT-on-chosen) term imitates the whole veto-surviving pool,
# stakeholder-voice and perspshift exemplars included -- see core/self_distill_trainer.py:
# "empirically the over-criticality only appears once that SFT term is on." The trained
# adapters role-play a stakeholder instead of answering as an assistant on 30.7% (AITA)
# and 37.8% (PAS) of the held-out 1000 (training_mitigations/eval/classify_response_voice.py).
# With the alpha=0 arms (*_perspshift_gp) this gives a dose-response curve
# alpha in {0, 0.25, 0.5, 1.0} rather than a single on/off ablation.
#
# NOTE the collapse may not be the SFT term's doing alone: 6 of the 10 candidates per
# prompt are stakeholder-voiced regardless of alpha, and any can be selected as `chosen`
# in a DPO pair. If role-play persists at alpha=0, shrink the pool via N_SOURCE_MODEL /
# N_SOURCE_CRITICAL instead. See _marlowe_dpo_body.sh.
# Marlowe `preempt`-partition PAS P5_nocrit_distill_gp_frozenstake run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p5.sbatch
#
# EXACT frozenstake recipe (P5 veto, mixed 4 model / 0 critical / 6 stakeholder,
# --stakeholder-voice second-person prompt, RPO_ALPHA=1.0 distill, and
# FROZEN_STAKEHOLDER=1 so the 6 stakeholder candidates come from the FROZEN BASE
# Qwen3-8B), on the PAS env. PAS is open-ended advice like OEQ, so the label
# machinery is inert here (label_suffix="" for PAS) — this is the OEQ frozenstake
# wrapper with RUN_ENV=PAS. Split: default 1000 train / 500 test.
#SBATCH --job-name=dpo_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p5
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
#SBATCH --output=/projects/m000254/logs/dpo_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p5_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p5_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p5
SELF_BASENAME=marlowe_run_train_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p5.sbatch
RPO_ALPHA=0.5
N_SOURCE_MODEL=4
N_SOURCE_CRITICAL=0
STAKEHOLDER_VOICE=1
FROZEN_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
