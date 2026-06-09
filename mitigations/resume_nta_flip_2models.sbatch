#!/bin/bash
#SBATCH --job-name=ntaflip_resume2
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/next/u/stephhk/sycophancy/mitigations/logs/sbatch_ntaflip_resume2_%j.out

# One more --resume pass for the two AITA-NTA-FLIP n=1000 runs that wrote their
# CSV at <3000 records (llama-4-scout 2800, qwen-2.5-7b-instruct 2690). Resume
# only re-does in-flight/missing tasks, so this fills gaps from API/judge
# failures without re-running completed records. Batch job so it survives a
# laptop disconnect. API keys propagate via sbatch --export=ALL.

cd /next/u/stephhk/sycophancy
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

declare -A MODELS=(
  [llama-4-scout]="meta-llama/llama-4-scout"
  [qwen-2.5-7b-instruct]="qwen/qwen-2.5-7b-instruct"
)

for slug in "${!MODELS[@]}"; do
  model="${MODELS[$slug]}"
  out="mitigations/results/explore_mitigations_results_AITA-NTA-FLIP_${slug}_gpt-5.4-mini_n1000.csv"
  log="mitigations/logs/run_AITA-NTA-FLIP_${slug}_n1000.resume2.log"
  "$PY" -u -m mitigations.explore_mitigations \
    --env_name AITA-NTA-FLIP --model "$model" --judge_model openai/gpt-5.4-mini \
    --conditions baseline be_critical stakeholder_concerns_v2 \
    --n 1000 --seed 0 --resume --max-cost 30 \
    --out "$out" >> "$log" 2>&1 &
done
wait
echo "both resume processes exited"
