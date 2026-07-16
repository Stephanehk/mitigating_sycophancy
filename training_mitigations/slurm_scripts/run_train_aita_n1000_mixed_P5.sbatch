#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_aita_n1000_mixed_P5.sbatch
# (from the repo root). AITA counterpart of run_train_pas_n1000_mixed_P5.sbatch:
# MIXED-sampling AITA n1000 run (2 bare policy / 2 "Be critical." / 6 round-robin
# over identified stakeholders) using the P5 TWO-STAGE veto: Stage 1 is the
# stakeholder-INDEPENDENT endorsement classifier (ENDORSE_CLASSIFY_PROMPT_P5),
# run once per response; Stage 2 asks each cached stakeholder to VETO/ACCEPT only
# the responses Stage 1 flagged as endorsing (VETO_PROMPT_P5). See
# simulated_stakeholders.py. No self-preference. Requires $HF_TOKEN.
#
# BARE-POST SURFACE (post-248b263). AITA candidates are sampled/trained on the
# bare post exactly like PAS/OEQ -- there is NO self-emitted YTA/NTA label. The
# verdict is read from the response prose by the gpt-5-mini classifier at eval
# time (eval/classify_aita_verdicts.py). This SUPERSEDES the earlier
# dpo_lora_aita_qwen3-8b_n1000_mixed_P5 run, which trained + evaluated on the old
# mandatory-label surface (AITA_LABEL_SUFFIX). Back up / clear that run's stale
# old-format eval/ before re-running to this same OUT_DIR.
#
# Split: TRAIN_N=1000 / TEST_N=1000 to match the AITA baselines
# (runs/baseline_AITA/test_ids.json = 1000). AITA splits 50/50 balanced on
# is_asshole at --seed 0; confirm the trainer's OUT_DIR/test_ids.json reproduces
# the baseline held-out set before comparing FNR/FPR head-to-head.
#
# --precompute-ref-batch-size 2 (not 8): the OEQ P5 runs OOM'd at the iter-2
# reference-logp precompute with batch-size 8. Batch size only affects ref-logp
# throughput (semantics-preserving), so we use 2 from the start.
#
# Pinned to next7 (2x A6000, 48GB): policy+training on cuda:0 needs ~17GB which
# does not fit the idle a4000 (16GB) nodes. next7 has flaky outbound DNS and
# Qwen3-8B is fully pre-cached under $HF_HOME, so we run HF offline.

#SBATCH --job-name=dpo_aita_n1000_mixed_P5
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_aita_n1000_mixed_P5_%j.out
#SBATCH --error=training_mitigations/logs/dpo_aita_n1000_mixed_P5_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# next7 has flaky outbound DNS and Qwen3-8B is fully pre-cached under $HF_HOME.
export HF_HUB_OFFLINE=1

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it in your shell before sbatch." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT_DIR=training_mitigations/runs/dpo_lora_aita_qwen3-8b_n1000_mixed_P5

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] HF_HOME=$HF_HOME"
echo "[sbatch] python=$PY"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env AITA \
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
    --test-n 1000 \
    --n-samples 10 \
    --mixed-sampling \
    --n-source-model 2 \
    --n-source-critical 2 \
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
