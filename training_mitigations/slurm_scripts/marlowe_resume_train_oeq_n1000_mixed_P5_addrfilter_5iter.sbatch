#!/bin/bash
# RESUME wrapper — continue the finished OEQ P5_addrfilter run for 2 MORE
# iterations (epochs), 3 -> 5, on the FREE allocation.
#
# Context: dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_addrfilter finished its original
# 3-iteration recipe (adapter_iter_1..3 all .complete; eval gave endorse~0.630 /
# roleplay~0.009). This wrapper re-runs the SAME recipe with NUM_ITERATIONS=5 so
# --allow-preempt's detect_resume_point picks up at k_max=3 and runs iterations 4
# and 5 — each re-samples on-policy from the previous adapter (iter-4 from
# adapter_iter_3), so this is 2 more rounds of iterative DPO, not extra epochs on
# stale iter-3 pairs. adapter_iter_3 is preserved, so the endorse=0.630
# checkpoint is never lost.
#
# BEFORE submitting: move the top-level completion sentinel aside so the body
# does not no-op (the per-iteration adapter_iter_<k>/.complete sentinels stay):
#   mv /projects/m000254/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_addrfilter/.complete \
#      /projects/m000254/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_addrfilter/.complete.iter3.bak
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_resume_train_oeq_n1000_mixed_P5_addrfilter_5iter.sbatch
#
# Recipe identical to marlowe_run_train_oeq_n1000_mixed_P5_addrfilter.sbatch
# (P5 veto, mixed 2 model / 2 critical / 6 stakeholder-voice, addressee filter,
# plain DPO); only NUM_ITERATIONS is added and SELF_BASENAME points at THIS file
# so the self-resubmit chain continues the 5-iter recipe.
#
# free/batch partition: PreemptMode=OFF, 24h. no-ops once .complete exists again
# (i.e. after iteration 5 saves).
#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_addrfilter_resume5
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
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_addrfilter_resume5_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_addrfilter_resume5_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_addrfilter
SELF_BASENAME=marlowe_resume_train_oeq_n1000_mixed_P5_addrfilter_5iter.sbatch
NUM_ITERATIONS=5
# RPO_ALPHA intentionally UNSET -> plain DPO (no distill). Do not add it here.
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
STAKEHOLDER_VOICE=1
ADDRESSEE_FILTER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
