#!/bin/bash
#
# RPO-ALPHA SWEEP ARM (alpha=0.25, baseline arm uses 1.0).
# Identical to marlowe_run_train_pas_n1000_mixed_P6_perspshift_distill_gp.sbatch in every respect except
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
# Marlowe `preempt`-partition PAS P6_perspshift_distill_gp run.
# Submit from a Marlowe login node (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_pas_n1000_mixed_P6_perspshift_distill_gp_rpo0p25.sbatch
#
# P6_perspshift_distill_gp recipe (mixed 2 model / 2 perspshift / 6 stakeholder,
# --stakeholder-voice, RPO_ALPHA=1.0 distill) with the P6 BIDIRECTIONAL veto +
# USER_STAKEHOLDER (4th narrator voice), on the PAS env. PAS is open-ended advice
# (label machinery inert, label_suffix=""). PAS does NOT set PERSPSHIFT_ASSESS
# (that OEQ-only addendum is for long narratives). Split: default 1000 / 500.
#SBATCH --job-name=dpo_pas_n1000_mixed_P6_perspshift_distill_gp_rpo0p25
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
#SBATCH --output=/projects/m000254/logs/dpo_pas_n1000_mixed_P6_perspshift_distill_gp_rpo0p25_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_pas_n1000_mixed_P6_perspshift_distill_gp_rpo0p25_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p25
SELF_BASENAME=marlowe_run_train_pas_n1000_mixed_P6_perspshift_distill_gp_rpo0p25.sbatch
RPO_ALPHA=0.25
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
