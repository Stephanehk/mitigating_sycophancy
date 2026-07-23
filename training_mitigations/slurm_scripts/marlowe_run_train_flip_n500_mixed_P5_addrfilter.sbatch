#!/bin/bash
# Marlowe free/batch AITA-NTA-FLIP P5_addrfilter run — NO DISTILL (plain DPO).
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_flip_n500_mixed_P5_addrfilter.sbatch
#
# METHOD (new): base P5_mixed (2 model / 2 "Be critical." / 6 stakeholder-voice,
# P5 veto) PLUS the ADDRESSEE FILTER — a pre-veto stage that removes any candidate
# which critiques the FORM of the user's message (grammar, wording, structure, a
# "more polished version") instead of engaging the narrator's situation. A
# filtered candidate becomes NEITHER chosen NOR rejected. Runs on the frozen
# sim/veto backend (ADDRESSEE_FILTER_PROMPT). See core/simulated_stakeholders.py.
# Env-only clone of marlowe_run_train_oeq_n1000_mixed_P5_addrfilter.sbatch (RUN_ENV
# + split changed; recipe identical).
#
# Training config: P5 veto, mixed sampling 2 model / 2 critical / 6 stakeholder.
#   (RPO_ALPHA unset)    plain DPO — no distill/SFT-on-chosen term
#   N_SOURCE_CRITICAL=2  the 2 steer candidates are the "Be critical." source
#   STAKEHOLDER_VOICE=1  the 6 stakeholder candidates use the 2nd-person voice
#   ADDRESSEE_FILTER=1   drop message-critique candidates before the veto panel
# The DPO prompt column stays the bare post. FLIP counts PAIRS: TRAIN_N=500 pairs
# / TEST_N=1000 (matches the held-out baseline).
#
# free/batch partition: PreemptMode=OFF, 24h. Self-resubmit chain in
# _marlowe_dpo_body.sh resumes from responses_iter_*.jsonl + checkpoints if a
# window ever dies; no-ops once .complete exists.
#SBATCH --job-name=dpo_flip_n500_mixed_P5_addrfilter
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
#SBATCH --output=/projects/m000254/logs/dpo_flip_n500_mixed_P5_addrfilter_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_flip_n500_mixed_P5_addrfilter_%j.err

RUN_ENV=AITA-NTA-FLIP
OUT_NAME=dpo_lora_flip_qwen3-8b_n500_mixed_P5_addrfilter
SELF_BASENAME=marlowe_run_train_flip_n500_mixed_P5_addrfilter.sbatch
TRAIN_N=500
TEST_N=1000
# RPO_ALPHA intentionally UNSET -> plain DPO (no distill). Do not add it here.
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
STAKEHOLDER_VOICE=1
ADDRESSEE_FILTER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
