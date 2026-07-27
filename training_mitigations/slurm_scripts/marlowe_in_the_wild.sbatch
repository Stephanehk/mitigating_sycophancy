#!/bin/bash
# In-the-wild qualitative dump: run "Our Model (3+1)" (OEQ arm) on 50 first user
# utterances from the Study-3 human-chat transcripts.
# GPU generation only — nothing here is judged, so no network access is needed.
# ~50 prompts x 1024 tokens on one H100 -> well under an hour.
#
#
# Optional --export overrides:
#   ITW_ADAPTER : adapter dir (default: the OEQ voice2 run)
#   ITW_N       : how many utterances (default 50)
#   ITW_BASE    : set to 1 to dump the untrained base model instead
#   ITW_MODEL   : base HF model id (default: the harness's Qwen3-8B). Set to
#                 Qwen/Qwen3-32B for the 32B arms — the 32B loads in bf16 on ONE
#                 H100 80GB (same single-device path the 32B held-out gen uses),
#                 so the --gpus=1 request above still holds.
#   ITW_TAG     : condition tag for the output filename (default: derived)
#   ITW_OUT_DIR : directory for the CSV/JSON/JSONL (default: the adapter's eval/)
#SBATCH --job-name=itw
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/itw_%j.out
#SBATCH --error=/projects/m000254/logs/itw_%j.err

source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_eval_env.sh
set -e

ITW_ADAPTER="${ITW_ADAPTER:-/projects/m000254/runs/dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5}"
ITW_N="${ITW_N:-50}"

ARGS=(--n "$ITW_N" --device cuda:0 --seed 0)
if [[ "${ITW_BASE:-0}" == "1" ]]; then
  ARGS+=(--base)
  [[ -z "${ITW_OUT_DIR:-}" ]] && ARGS+=(--out "/projects/m000254/runs/in_the_wild_study3_base.csv")
else
  ARGS+=(--adapter "$ITW_ADAPTER")
fi
[[ -n "${ITW_MODEL:-}" ]] && ARGS+=(--model-id "$ITW_MODEL")
[[ -n "${ITW_TAG:-}" ]] && ARGS+=(--tag "$ITW_TAG")
[[ -n "${ITW_OUT_DIR:-}" ]] && ARGS+=(--out-dir "$ITW_OUT_DIR")

echo "[itw] adapter=$ITW_ADAPTER n=$ITW_N base=${ITW_BASE:-0} model=${ITW_MODEL:-default8b} tag=${ITW_TAG:-derived}"
"$PY" training_mitigations/eval/in_the_wild_response_analysis.py "${ARGS[@]}"
echo "[itw] done"
