#!/bin/bash
#
# DELTA vs marlowe_final_train_oeq_P6_perspshift_distill_gp_rpo0p5.sbatch:
# adds the assistant-role veto (ASSISTANT_ROLE_VETO=1) -- candidates that fail
# to answer the user AS an AI assistant (message-as-text critique, in-story
# voice, third-person case study) are auto-REJECTED into the DPO pairs, <= 3
# such pairs per prompt, off the veto cross-product. The veto runs on the LOCAL
# frozen Qwen3-8B sim, so it costs GPU time, not API spend.
# NOT part of the frozen `finaldata` paper set -- this is a new arm.
#
# FINAL-RUN wrapper: OEQ P6_persp (alpha=0.5) trained on the FROZEN
# final_training_eval_data/oeq_train split, held out on oeq_test.
# Byte-identical recipe to marlowe_run_train_oeq_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_arole.sbatch
# (the tuning arm) except:
#   - sources _marlowe_dpo_body_final.sh (adds --use-final-data)
#   - OUT_NAME suffixed _final so it never clobbers the tuning run dir
# --train-n/--test-n/--seed for the split are IGNORED under --use-final-data.
#SBATCH --job-name=dpo_oeq_P6_perspshift_distill_gp_rpo0p5_arole
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_oeq_P6_perspshift_distill_gp_rpo0p5_arole_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_P6_perspshift_distill_gp_rpo0p5_arole_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_arole
SELF_BASENAME=marlowe_train_oeq_P6_perspshift_distill_gp_rpo0p5_arole.sbatch
RPO_ALPHA=0.5
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
ASSISTANT_ROLE_VETO=1
USER_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body_final.sh
