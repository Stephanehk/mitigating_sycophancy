#!/bin/bash
# Parametrized single-run version of run_judge_v5_heldout1000_pas_mixed_P5.sbatch:
# stage the 1000-prompt held-out responses for ONE completed run (DPO+LoRA, FINAL
# adapter) and score them with the v5 GPT-5-mini endorse judge (narrow
# action-endorsement rate #1/(#0+#1)). CPU-only, network-bound (OpenRouter); gate
# afterok on the matching gen array job so it runs once the final1000 responses
# are written.
#
# Parametrized via --export:
#   JRUN       : run dir RELATIVE to training_mitigations/ (holds eval/ shards)
#   JNAME      : stage/dataset stem (-> sycoharm_runs/inputs/<JNAME>.csv and
#                endorse_gpt5mini_v5_temp0/scored_<JNAME>.csv)
#   JSHARDS    : number of gen shards to concatenate (default 10)
#   JTAG       : response-file tag preceding _shard{i}of{n} (default
#                final_final1000_temp1.0_seed0). Gen jobs that pass
#                --tag-extra finaldata (next_final_gen.sbatch) write
#                final_finaldata_temp1.0_seed0 instead and MUST set this;
#                otherwise staging fails on a missing shard CSV.
#
# Requires OPENROUTER_API_KEY in the job env (submit with --export=ALL,...).
#
# Outputs: sycoharm_runs/inputs/<JNAME>.csv                              (staged)
#          sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_<JNAME>.csv    (scored)

#SBATCH --job-name=judge_v5_generic
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/judge/%x_%j.out
#SBATCH --error=training_mitigations/logs/judge/%x_%j.err

set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2; exit 1
fi
: "${JRUN:?set JRUN (run dir relative to training_mitigations/)}"
: "${JNAME:?set JNAME (stage/dataset stem)}"
JSHARDS="${JSHARDS:-10}"
JTAG="${JTAG:-final_final1000_temp1.0_seed0}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
cd /next/u/stephhk/sycophancy
mkdir -p training_mitigations/logs/judge

echo "[judge] staging $JRUN -> sycoharm_runs/inputs/$JNAME.csv (FINAL adapter, $JSHARDS shards, tag=$JTAG)"
"$PY" training_mitigations/eval/stage_heldout_inputs.py \
    --run-dir "training_mitigations/$JRUN" \
    --name "$JNAME" --tag "$JTAG" --num-shards "$JSHARDS" --expect 1000

echo "[judge] scoring with v5 gpt-5-mini endorse judge"
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "$JNAME" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost 20.0 \
    --workers 8 --checkpoint 100
echo "[judge] done."
