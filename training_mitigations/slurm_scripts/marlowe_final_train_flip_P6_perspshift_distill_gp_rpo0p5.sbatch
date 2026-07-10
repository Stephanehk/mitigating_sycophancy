#!/bin/bash
#
# FINAL-RUN wrapper: AITA-NTA-FLIP P6_persp (alpha=0.5) trained on the FROZEN
# final_training_eval_data/flip_train split (500 PAIRS), held out on flip_test
# (1000 pairs). Byte-identical recipe to
# marlowe_run_train_flip_n500_mixed_P6_perspshift_distill_gp_rpo0p5.sbatch except
# it sources _marlowe_dpo_body_final.sh (--use-final-data) and suffixes OUT_NAME
# _final. --train-n/--test-n/--seed are IGNORED under --use-final-data.
#SBATCH --job-name=dpo_flip_P6_perspshift_distill_gp_rpo0p5_final
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
#SBATCH --output=/projects/m000254/logs/dpo_flip_P6_perspshift_distill_gp_rpo0p5_final_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_flip_P6_perspshift_distill_gp_rpo0p5_final_%j.err

RUN_ENV=AITA-NTA-FLIP
OUT_NAME=dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final
SELF_BASENAME=marlowe_final_train_flip_P6_perspshift_distill_gp_rpo0p5.sbatch
TRAIN_N=500
TEST_N=1000
RPO_ALPHA=0.5
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body_final.sh
