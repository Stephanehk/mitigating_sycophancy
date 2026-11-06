#!/bin/bash
# MMLU (lm-evaluation-harness, 5-shot, loglikelihood multiple choice, all 57
# subjects / 14,042 test questions) for Qwen3-8B base + the four "Our Model
# (3+1)" adapters -- capability-loss check for the rebuttal (2026-11-06).
#
# One array task per model:
#   0 base | 1 oeq | 2 pas | 3 aita | 4 flip
# Env: /projects/m000254/envs/lmeval = venv layered (--system-site-packages) on
# llm126, with lm_eval 0.4.13 installed under pins for torch/transformers/peft/
# accelerate/datasets, so llm126 itself is unchanged. MMLU (cais/mmlu) is
# pre-cached in $HF_HOME; runs fully offline.
# --log_samples keeps per-question correctness for a paired McNemar vs base.
# Preempt is REQUEUE (restart from scratch); a task whose results json already
# exists exits immediately, so a requeue after completion is a no-op.
#
# Submit:  sbatch --array=0-4 run_mmlu_qwen3_preempt.sbatch
#SBATCH --job-name=mmlu_q3
#SBATCH --account=marlowe-m000254
#SBATCH --qos=normal
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/mmlu_q3_%A_%a.out
#SBATCH --error=/projects/m000254/logs/mmlu_q3_%A_%a.err
set -euo pipefail

export HF_HOME=/projects/m000254/hf_cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LMEVAL=/projects/m000254/envs/lmeval/bin/lm_eval
RUNS=/projects/m000254/runs
OUTROOT=$RUNS/mmlu_qwen3-8b

NAMES=(base oeq pas aita flip)
ADAPTERS=("" \
  $RUNS/dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5 \
  $RUNS/dpo_lora_pas_qwen3-8b_P6_voice2_rpo0p5 \
  $RUNS/dpo_lora_aita_qwen3-8b_P6_voice2R2_rpo0p5 \
  $RUNS/dpo_lora_flip_qwen3-8b_P6_voice2R2_rpo0p5)
i=${SLURM_ARRAY_TASK_ID:?run as an array job}
NAME=${NAMES[$i]}; AD=${ADAPTERS[$i]}
OUT=$OUTROOT/$NAME

if compgen -G "$OUT/*/results_*.json" > /dev/null; then
  echo "[mmlu] $NAME already has results in $OUT -- skipping."; exit 0
fi
rm -rf "$OUT"; mkdir -p "$OUT"   # a requeued partial run restarts clean

MA="pretrained=Qwen/Qwen3-8B,dtype=bfloat16"
if [[ -n "$AD" ]]; then
  [[ -f "$AD/adapter_config.json" ]] || { echo "ERROR: no adapter at $AD" >&2; exit 1; }
  MA="$MA,peft=$AD"
fi
echo "[mmlu] job=${SLURM_JOB_ID} task=$i name=$NAME node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,) model_args=$MA"
"$LMEVAL" --model hf --model_args "$MA" \
  --tasks mmlu --num_fewshot 5 --batch_size auto --device cuda:0 --seed 0 \
  --log_samples --output_path "$OUT"
echo "[mmlu] done $NAME"
