#!/bin/bash
# Marlowe free/batch OEQ P5_futureuser run — NO DISTILL (plain DPO).
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_oeq_n1000_mixed_P5_futureuser.sbatch
#
# METHOD (new): base P5_mixed (P5 veto, 2 model / 6 stakeholder-voice) but the 2
# "Be critical." steer candidates are REPLACED by 2 FUTURE-USER-voice candidates —
# the policy answers in the voice of the narrator's own future self, who must live
# with the long-term consequences of the present decision
# (FUTURE_USER_RESPONSE_PROMPT, on-policy). The DPO prompt column stays the bare
# first-person post. See core/trainer.py.
#
# Training config: P5 veto, mixed sampling 2 model / 2 future-user / 6 stakeholder.
#   (RPO_ALPHA unset)    plain DPO — no distill/SFT-on-chosen term
#   FUTURE_USER=1        the 2 steer candidates come from the future-self voice
#                        (fills the N_SOURCE_CRITICAL slots; replaces "Be critical.")
#   N_SOURCE_CRITICAL=2  the future-user source count (mutually exclusive w/ PERSPSHIFT)
#   STAKEHOLDER_VOICE=1  the 6 stakeholder candidates use the 2nd-person voice
# The DPO prompt column stays the bare first-person post throughout.
#
# free/batch partition: PreemptMode=OFF, 24h. Self-resubmit chain in
# _marlowe_dpo_body.sh resumes from responses_iter_*.jsonl + checkpoints if a
# window ever dies; no-ops once .complete exists.
#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_futureuser
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_futureuser_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_futureuser_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_futureuser
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P5_futureuser.sbatch
# RPO_ALPHA intentionally UNSET -> plain DPO (no distill). Do not add it here.
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
FUTURE_USER=1
STAKEHOLDER_VOICE=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
