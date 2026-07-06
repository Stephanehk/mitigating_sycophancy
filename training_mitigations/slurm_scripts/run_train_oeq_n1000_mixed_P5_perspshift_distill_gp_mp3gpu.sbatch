#!/bin/bash
# mixed_P5_perspshift_distill_gp (OEQ): the perspective-shift recipe WITH the
# distill SFT term and the voice-constrained stakeholder prompt. Concretely, this
# is run_train_oeq_n1000_mixed_P5_perspective_shift.sbatch (P5 two-stage veto,
# seed 0, identical DPO hyperparameters, 2 bare / 2 perspshift / 6 stakeholder
# mix, --perspective-shift-sampling + --perspshift-assess, --rewrite-max-tokens
# 1536) with TWO additions stacked on top:
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
# GPU layout: cloned from run_train_oeq_n1000_mixed_P5_nocrit_distill_gp_mp3gpu
# (3xA6000 pipeline-MP, --policy-device-map 0,2). OEQ's long narratives + the
# added distill activation memory OOM'd 2xA6000 in the distill runs, so the
# policy is pipelined across GPUs 0 and 2 with the sim judge on GPU 1. (PAS fits
# on 2xA6000 — see the PAS sibling.) Separate _perspshift_distill_gp OUT_DIR so
# the plain perspective_shift run is untouched.
#
# Requires 3 free A6000s on next7 and $HF_TOKEN (next7 runs HF offline).

#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_perspshift_distill_gp_mp3
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_oeq_n1000_mixed_P5_perspshift_distill_gp_mp3_%j.out
#SBATCH --error=training_mitigations/logs/dpo_oeq_n1000_mixed_P5_perspshift_distill_gp_mp3_%j.err

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
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_perspshift_distill_gp

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ \
    --output-dir "$OUT_DIR" \
    --veto-prompt P5 \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --policy-device-map 0,2 \
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
    --perspshift-assess \
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
