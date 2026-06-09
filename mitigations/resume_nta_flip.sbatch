#!/bin/bash
#SBATCH --job-name=ntaflip_resume
#SBATCH --partition=next
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/next/u/stephhk/sycophancy/mitigations/logs/sbatch_ntaflip_resume_%j.out

# Resume the AITA-NTA-FLIP n=1000 runs that did not finish, as a BATCH job so
# they are decoupled from the interactive salloc session and survive a laptop
# disconnect. Each model resumes from its checkpoint (--resume), so only
# in-flight tasks are re-done. API keys propagate via sbatch --export=ALL.

cd /next/u/stephhk/sycophancy
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

declare -A MODELS=(
  [llama-3.3-70b-instruct]="meta-llama/llama-3.3-70b-instruct"
  [qwen3-235b-a22b-2507]="qwen/qwen3-235b-a22b-2507"
  [deepseek-v3.2]="deepseek/deepseek-v3.2"
  [deepseek-chat-v3-0324]="deepseek/deepseek-chat-v3-0324"
)

for slug in "${!MODELS[@]}"; do
  model="${MODELS[$slug]}"
  out="mitigations/results/explore_mitigations_results_AITA-NTA-FLIP_${slug}_gpt-5.4-mini_n1000.csv"
  log="mitigations/logs/run_AITA-NTA-FLIP_${slug}_n1000.resume.log"
  "$PY" -u -m mitigations.explore_mitigations \
    --env_name AITA-NTA-FLIP --model "$model" --judge_model openai/gpt-5.4-mini \
    --conditions baseline be_critical stakeholder_concerns_v2 \
    --n 1000 --seed 0 --resume --max-cost 30 \
    --out "$out" >> "$log" 2>&1 &
done
wait
echo "all four resume processes exited"
