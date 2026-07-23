# Shared env for Marlowe held-out eval GENERATION jobs
# (marlowe_run_eval_*.sbatch source this at the top). This configures the SAME
# Marlowe site paths / cu126 interpreter as _marlowe_dpo_body.sh, so there is a
# single place to edit them.
#
# SCOPE: GPU generation ONLY. Each array task samples free-form responses on the
# BARE post (+ optional prompt steer) and writes
# aita_responses_<tag>_shard{i}of{N}.csv into the run dir. It emits NO verdict.
# The YTA/NTA/UNCLEAR verdict + FNR/FPR / both-NTA metrics come afterward from
# the network-bound classifier stage (classify_aita_verdicts.py, gpt-5-mini),
# which needs OPENROUTER_API_KEY and outbound HTTPS -> run it on a Marlowe LOGIN
# node (marlowe_classify_all.sh), not on a compute node.
#
# The 8B fits on ONE H100 80GB, so the wrappers request --gpus=1 and load on the
# single allocated device (no --device-map). HF_HUB_OFFLINE=1: the base Qwen3-8B
# is already in $HF_HOME from training, so no gated-download / HF_TOKEN needed.

set -uo pipefail

# ===================== EDIT FOR YOUR MARLOWE SITE =====================
PROJECT=m000254                                  # Marlowe project ID
REPO=/projects/$PROJECT/sycophancy               # repo (symlink -> mitigating_sycophancy)
ENV_PREFIX=/projects/$PROJECT/envs/llm126        # cu126 conda env (r565/CUDA-12.7 driver)
export HF_HOME=/projects/$PROJECT/hf_cache       # HF model cache
# =====================================================================

export PYTHONNOUSERSITE=1                         # ignore ~/.local (old CUDA-13 stack)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1                           # base model is cached; stay offline
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

cd "$REPO"
PY="$ENV_PREFIX/bin/python"
EV=training_mitigations/eval/eval_aita_label_heldout.py

echo "[marlowe-eval] jobid=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-?} node=$(hostname)"
echo "[marlowe-eval] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[marlowe-eval] python=$PY repo=$REPO"
