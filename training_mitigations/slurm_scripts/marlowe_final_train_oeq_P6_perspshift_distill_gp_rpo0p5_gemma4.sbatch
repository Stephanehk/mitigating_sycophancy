#!/bin/bash
#
# GEMMA-4 FINAL-RUN wrapper: OEQ P6_persp (alpha=0.5) with the Gemma-4-12B-it
# policy+sim (replacing Qwen3-8B), trained on the FROZEN
# final_training_eval_data/oeq_train split, held out on oeq_test.
# Same recipe as marlowe_final_train_oeq_P6_perspshift_distill_gp_rpo0p5.sbatch
# except it sources _marlowe_dpo_body_gemma4.sh (gemma4vllm env + CUDA-13
# forward-compat + --model-id google/gemma-4-12B-it + text-decoder-only LoRA
# targets) and OUT_NAME names the gemma4 policy so it never clobbers the Qwen dir.
# --train-n/--test-n/--seed for the split are IGNORED under --use-final-data.
#SBATCH --job-name=dpo_oeq_P6_perspshift_distill_gp_rpo0p5_gemma4
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
#SBATCH --output=/projects/m000254/logs/dpo_oeq_P6_perspshift_distill_gp_rpo0p5_gemma4_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_P6_perspshift_distill_gp_rpo0p5_gemma4_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_gemma4-12b-it_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final
SELF_BASENAME=marlowe_final_train_oeq_P6_perspshift_distill_gp_rpo0p5_gemma4.sbatch
RPO_ALPHA=0.5
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body_gemma4.sh
