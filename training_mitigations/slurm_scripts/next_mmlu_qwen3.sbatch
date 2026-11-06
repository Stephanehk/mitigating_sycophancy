#!/bin/bash
# NEXT-CLUSTER port of run_mmlu_qwen3_preempt.sbatch.
#
# MMLU (lm-evaluation-harness 0.4.13, 5-shot, loglikelihood multiple choice,
# all 57 subjects / 14,042 test questions) for Qwen3-8B base + the four "Our
# Model (3+1)" adapters -- capability-loss check for the rebuttal (2026-11-06).
#
# One array task per model:
#   0 base | 1 oeq | 2 pas | 3 aita | 4 flip
#
# One-time setup on the next login node (see REBUTTAL_XENV_MMLU.md):
#   1. lm-eval venv layered on the llm env (pins its torch/transformers/peft):
#        $LLM/bin/python -m venv --system-site-packages /next/u/stephhk/envs/lmeval
#        $LLM/bin/python -m pip freeze | grep -iE '^(torch|transformers|peft|accelerate|datasets)==' > /tmp/pins.txt
#        PYTHONNOUSERSITE=1 /next/u/stephhk/envs/lmeval/bin/pip install -r requirements-mmlu.txt -c /tmp/pins.txt
#   2. cache MMLU (jobs run offline):
#        HF_HOME=... /next/u/stephhk/envs/lmeval/bin/python training_mitigations/eval/prefetch_mmlu.py
#   3. adapters under $MMLU_ADAPTER_ROOT with their Marlowe run-dir names.
#
# --log_samples keeps per-question correctness for eval/mmlu_mcnemar.py.
# A task whose results json already exists exits immediately (safe to resubmit).
#
# Submit:  sbatch --array=0-4 next_mmlu_qwen3.sbatch
#SBATCH --job-name=mmlu_q3
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/next/u/stephhk/sycophancy/training_mitigations/logs/mmlu_q3_%A_%a.out
#SBATCH --error=/next/u/stephhk/sycophancy/training_mitigations/logs/mmlu_q3_%A_%a.err
set -euo pipefail

export HF_HOME="${HF_HOME:-/next/u/stephhk/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LMEVAL="${LMEVAL:-/next/u/stephhk/envs/lmeval/bin/lm_eval}"
RUNS="${MMLU_ADAPTER_ROOT:-/next/u/stephhk/sycophancy/training_mitigations/runs}"
OUTROOT="${MMLU_OUT_ROOT:-$RUNS/mmlu_qwen3-8b}"

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
rm -rf "$OUT"; mkdir -p "$OUT"   # a partial run restarts clean

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
