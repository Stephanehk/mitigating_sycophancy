# Shared site env for the next-cluster EVAL/GEN jobs (next_final_gen.sbatch
# sources this). Counterpart of Marlowe's _marlowe_eval_env.sh, which was never
# committed to the repo — it only ever existed on the Marlowe filesystem — so
# this is written from that file's contract (site paths + $PY + HF offline +
# cd $REPO), not copied from it.
#
# Generation is network-free: OEQ/PAS run eval_endorsement_heldout.py with
# --no-judge, and eval_aita_label_heldout.py has no judge at all (AITA verdicts
# are parsed from prose by the separate gpt-5-mini classify stage, which must run
# on a node WITH egress — not here). So the compute node only needs the local
# Qwen3-8B weights, which are pre-cached.

set -uo pipefail

REPO=/next/u/stephhk/sycophancy
ENV_PREFIX=/next/u/stephhk/miniconda3/envs/llm
export HF_HOME=/next/u/stephhk/.cache/huggingface
PY="$ENV_PREFIX/bin/python"

# next7 has flaky outbound DNS; Qwen3-8B is fully pre-cached under $HF_HOME in
# BOTH layouts the code needs (the HF policy load passes cache_dir=$HF_HOME and
# resolves $HF_HOME/models--Qwen--Qwen3-8B/, while vLLM uses the default hub
# layout $HF_HOME/hub/). If HF_HOME ever changes, both must be pre-populated.
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ignore ~/.local so imports can't be shadowed by user-site packages.
export PYTHONNOUSERSITE=1

cd "$REPO"
