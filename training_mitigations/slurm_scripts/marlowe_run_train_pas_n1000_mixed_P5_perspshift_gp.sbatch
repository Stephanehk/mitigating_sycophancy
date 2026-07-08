#!/bin/bash
# Marlowe `preempt`-partition PAS P5_perspshift_gp run — NO DISTILL.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_pas_n1000_mixed_P5_perspshift_gp.sbatch
#
# ABLATION: identical to marlowe_run_train_pas_n1000_mixed_P5_perspshift_distill_gp
# except RPO_ALPHA is UNSET -> plain DPO, no SFT/NLL term on the chosen responses.
# Everything else (veto, candidate mix, perspshift, stakeholder voice) held fixed.
#
# WHY: the perspshift-trained adapters collapse into role-playing a stakeholder in
# the user's story instead of answering as an assistant (held-out 1000, measured by
# training_mitigations/eval/classify_response_voice.py: PAS 37.8%, AITA 30.7%).
# core/self_distill_trainer.py records the mechanism: trl's `sft` term pools EVERY
# `chosen` completion into one cross-entropy, so it imitates the whole veto-surviving
# pool -- including the stakeholder-voice / perspshift exemplars -- and "empirically
# the over-criticality only appears once that SFT term is on." This arm removes that
# term to test whether the VOICE collapse shares the cause.
#
# CAVEAT: dropping the SFT term does NOT remove exposure to stakeholder exemplars.
# 6 of the 10 candidates per prompt are still stakeholder-voiced, and any can be
# selected as `chosen` in a DPO pair. If the collapse persists, the cause is the
# CANDIDATE POOL, not the distill term -- shrink it with N_SOURCE_MODEL /
# N_SOURCE_CRITICAL (see _marlowe_dpo_body.sh).
#
# Training config: P5 veto, mixed sampling 2 model / 2 perspshift / 6 stakeholder.
#   (RPO_ALPHA unset)    plain DPO -- no distill/SFT-on-chosen term
#   PERSPSHIFT=1         the 2 "critical" steer candidates come from the ELEPHANT
#                        third-person rewrite of the sentence, not "Be critical."
#   STAKEHOLDER_VOICE=1  the 6 stakeholder candidates use the 2nd-person voice
#                        prompt instead of first-person role-play (the "_gp" change)
# NO PERSPSHIFT_ASSESS: "Assess this narrative." is OEQ-only, and the existing PAS
# distill_gp wrapper omits it too. Kept identical so this is a clean 1-knob ablation.
# The DPO prompt column stays the bare first-person sentence throughout.
#
# preempt partition: MaxTime=04:00:00 (4h), GraceTime=900 (15min), preemptible.
# SELF-RESUBMITTING 4h chain (the body queues an afterany successor each window),
# resumable via responses_iter_*.jsonl + --dpo-save-steps checkpoints. All site
# paths + the recipe knobs live in _marlowe_dpo_body.sh.
#SBATCH --job-name=dpo_pas_n1000_mixed_P5_perspshift_gp
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
#SBATCH --output=/projects/m000254/logs/dpo_pas_n1000_mixed_P5_perspshift_gp_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_pas_n1000_mixed_P5_perspshift_gp_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P5_perspshift_gp
SELF_BASENAME=marlowe_run_train_pas_n1000_mixed_P5_perspshift_gp.sbatch
# RPO_ALPHA intentionally UNSET -> plain DPO (no distill). Do not add it here.
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
STAKEHOLDER_VOICE=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
