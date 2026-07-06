#!/bin/bash
# mixed_P5_perspshift_distill_gp (PAS): the perspective-shift recipe WITH the
# distill SFT term and the voice-constrained stakeholder prompt. Concretely, this
# is run_train_pas_n1000_mixed_P5_perspective_shift.sbatch (P5 two-stage veto,
# seed 0, identical DPO hyperparameters, 2 bare / 2 perspshift / 6 stakeholder
# mix, --perspective-shift-sampling, --rewrite-max-tokens 1536; PAS does NOT use
# --perspshift-assess) with TWO additions stacked on top:
#
#   --rpo-alpha 1.0    : distill / SFT anchor term on the chosen response
#                        (loss = DPO + 1.0 * mean-token CE(chosen)), exactly as
#                        in the *_nocrit_distill* runs.
#   --stakeholder-voice: the 6 stakeholder candidates are generated with the
#                        second-person VOICE-constrained prompt
#                        (STAKEHOLDER_RESPONSE_PROMPT_VOICE) instead of the
#                        first-person role-play prompt — the "_gp" change.
#
# These three interventions hit three independent parts of the candidate mix:
# perspective-shift replaces the 2 "critical" steer candidates (post rewritten to
# third person, then sampled), --stakeholder-voice reshapes the 6 stakeholder
# candidates, and --rpo-alpha adds an SFT term on the chosen exemplar. The DPO
# prompt column stays the bare first-person post throughout. This is NOT a
# "nocrit" run: n_source_critical stays 2 (now perspshift), so the name is
# perspshift_distill_gp, not nocrit_distill_gp.
#
# GPU layout: 2xA6000, matching both the PAS perspective_shift base and the PAS
# _nocrit_distill_gp run (PAS fits the distill term on 2 GPUs; only OEQ needed
# 3-GPU pipeline-MP). Policy on cuda:0, sim judge on cuda:1. Separate
# _perspshift_distill_gp OUT_DIR so the plain perspective_shift run is untouched.
#
# Requires 2 free A6000s on next7 and $HF_TOKEN (next7 runs HF offline).

#SBATCH --job-name=dpo_pas_n1000_mixed_P5_perspshift_distill_gp
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_pas_n1000_mixed_P5_perspshift_distill_gp_%j.out
#SBATCH --error=training_mitigations/logs/dpo_pas_n1000_mixed_P5_perspshift_distill_gp_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it in your shell before sbatch." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT_DIR=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_mixed_P5_perspshift_distill_gp

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env PAS \
    --output-dir "$OUT_DIR" \
    --veto-prompt P5 \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --sim-backend vllm \
    --veto-concurrency 16 \
    --precompute-ref-batch-size 2 \
    --num-iterations 3 \
    --train-n 1000 \
    --test-n 500 \
    --n-samples 10 \
    --mixed-sampling \
    --n-source-model 2 \
    --n-source-critical 2 \
    --perspective-shift-sampling \
    --rewrite-max-tokens 1536 \
    --stakeholder-voice \
    --rpo-alpha 1.0 \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-tokens 1024 \
    --sample-max-tokens 1024 \
    --beta 0.1 \
    --learning-rate 5e-5 \
    --num-train-epochs 1 \
    --per-device-batch-size 1 \
    --grad-accum 8 \
    --max-length 3072 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05
