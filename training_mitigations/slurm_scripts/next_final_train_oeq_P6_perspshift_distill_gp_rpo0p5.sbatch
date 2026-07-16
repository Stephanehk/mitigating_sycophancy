#!/bin/bash
#
# NEXT-CLUSTER port of marlowe_final_train_oeq_P6_perspshift_distill_gp_rpo0p5.sbatch.
#
# FINAL-RUN: OEQ P6_persp (alpha=0.5), trained on the FROZEN
# final_training_eval_data/oeq_train split, held out on oeq_test.
# The recipe knobs below are byte-identical to the Marlowe wrapper; only the
# #SBATCH site lines and the sourced body differ (see _next_dpo_body_final.sh for
# what changes and why: site paths, no resubmit chain, DPO_SAVE_STEPS=0,
# --precompute-ref-batch-size 2).
#
# OUT_NAME is UNCHANGED from the Marlowe run dir name on purpose — the adapter is
# relayed back to Marlowe for generation + scoring, which keys off this name.
#
# Submit from the repo root, with HF_TOKEN exported:
#   sbatch --export=ALL,DPO_SAVE_STEPS=0 \
#     training_mitigations/slurm_scripts/next_final_train_oeq_P6_perspshift_distill_gp_rpo0p5.sbatch
#
# DPO_SAVE_STEPS=0 is mandatory: at the default (25) the HF Trainer writes
# mid-epoch checkpoints, and a job killed mid-DPO-epoch resumes via a path that
# restores the LR schedule but NOT the LoRA weights — silently producing an
# under-trained adapter. That is what invalidated the previous OEQ run. The body
# also defaults it to 0 and hard-fails on any other value, so both layers agree.
#
#SBATCH --job-name=dpo_oeq_P6_perspshift_distill_gp_rpo0p5_final
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/next/u/stephhk/sycophancy/training_mitigations/logs/dpo_oeq_P6_perspshift_distill_gp_rpo0p5_final_%j.out
#SBATCH --error=/next/u/stephhk/sycophancy/training_mitigations/logs/dpo_oeq_P6_perspshift_distill_gp_rpo0p5_final_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final
RPO_ALPHA=0.5
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
# 3xA6000 pipeline-MP: shard the policy across GPU 0+2 so the full-vocab logits
# spike stops OOMing a single card at --max-length 3072 (vLLM sim on GPU 1).
POLICY_DEVICE_MAP=0,2
source /next/u/stephhk/sycophancy/training_mitigations/slurm_scripts/_next_dpo_body_final.sh
