#!/bin/bash
# Marlowe `preempt`-partition PAS P6_perspshift_gp run — NO DISTILL.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_pas_n1000_mixed_P6_perspshift_gp.sbatch
#
# ABLATION: identical to marlowe_run_train_pas_n1000_mixed_P6_perspshift_distill_gp
# except RPO_ALPHA is UNSET -> plain DPO, no SFT/NLL term on the chosen responses.
# Everything else (veto panel, candidate mix, perspshift, stakeholder voice) fixed.
#
# WHY: this exact adapter's PAS responses role-play a stakeholder 37.8% of the time
# on the held-out 1000 (378/1000, Wilson 95% CI [0.348, 0.408]) instead of answering
# as an assistant -- e.g. to "Haven't shown up to class for the last 6 weeks" it
# replies "We were counting on your presence... for the community we've built".
# Measured by training_mitigations/eval/classify_response_voice.py. The AITA sibling
# is 30.7%. core/self_distill_trainer.py records the mechanism: trl's `sft` term pools
# EVERY `chosen` completion into one cross-entropy, so it imitates the whole
# veto-surviving pool -- including the stakeholder-voice / perspshift exemplars --
# and "empirically the over-criticality only appears once that SFT term is on."
# This arm removes that term to test whether the VOICE collapse shares the cause.
#
# CAVEAT: dropping the SFT term does NOT remove exposure to stakeholder exemplars.
# 6 of the 10 candidates per prompt are still stakeholder-voiced, and any can be
# selected as `chosen` in a DPO pair. If the collapse persists, the cause is the
# CANDIDATE POOL, not the distill term -- shrink it with N_SOURCE_MODEL /
# N_SOURCE_CRITICAL (see _marlowe_dpo_body.sh).
#
# Training config: P6 veto (4-voice panel incl. the narrator/user stakeholder),
# mixed sampling 2 model / 2 perspshift / 6 stakeholder.
#   (RPO_ALPHA unset)    plain DPO -- no distill/SFT-on-chosen term
#   PERSPSHIFT=1         the 2 "critical" steer candidates come from the ELEPHANT
#                        third-person rewrite of the sentence, not "Be critical."
#   STAKEHOLDER_VOICE=1  the 6 stakeholder candidates use the 2nd-person voice
#                        prompt instead of first-person role-play (the "_gp" change)
#   VETO_PROMPT=P6 + USER_STAKEHOLDER=1
#                        the 4th "user"/narrator voice joins the VETO PANEL only;
#                        it is never sampled for candidates. The body asserts the
#                        USER_STAKEHOLDER=1 => VETO_PROMPT=P6 invariant.
# NO PERSPSHIFT_ASSESS: OEQ-only; the existing PAS distill_gp wrapper omits it too.
# The DPO prompt column stays the bare first-person sentence throughout.
#
# free/batch partition: MaxTime=2-00:00:00, PreemptMode=OFF -- not preemptible.
# The self-resubmit chain in _marlowe_dpo_body.sh stays wired but should never fire
# within the 24h limit; if it does, it resumes from responses_iter_*.jsonl +
# --dpo-save-steps checkpoints and no-ops once .complete exists.
#SBATCH --job-name=dpo_pas_n1000_mixed_P6_perspshift_gp
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
#SBATCH --output=/projects/m000254/logs/dpo_pas_n1000_mixed_P6_perspshift_gp_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_pas_n1000_mixed_P6_perspshift_gp_%j.err

RUN_ENV=PAS
OUT_NAME=dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_gp
SELF_BASENAME=marlowe_run_train_pas_n1000_mixed_P6_perspshift_gp.sbatch
# RPO_ALPHA intentionally UNSET -> plain DPO (no distill). Do not add it here.
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
