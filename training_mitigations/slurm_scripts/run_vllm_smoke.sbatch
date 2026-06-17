#!/bin/bash
# One-off smoke test for the vLLM sim backend (vllm_sim.py): launches the
# OpenAI-compatible server with the exact flags train_dpo_lora.py uses,
# waits for health, runs a greedy call (checking enable_thinking=False is
# honored — no <think> block) and a 32-way concurrent fan-out. Uses
# Qwen3-0.6B so it fits a single a4000; the 8B path differs only in size.
#SBATCH --job-name=vllm_sim_smoke
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next5,next6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=training_mitigations/logs/vllm_sim_smoke_%j.out
#SBATCH --error=training_mitigations/logs/vllm_sim_smoke_%j.err

set -euo pipefail
export HF_HOME=/next/u/stephhk/.cache/huggingface
# Compute nodes have flaky outbound DNS; the model is fully pre-cached, so
# force hub-cache-only resolution (same setting the real train jobs use).
export HF_HUB_OFFLINE=1

/next/u/stephhk/miniconda3/envs/llm/bin/python - <<'EOF'
import os, sys, time
sys.path.insert(0, "/next/u/stephhk/sycophancy/training_mitigations")
from concurrent.futures import ThreadPoolExecutor
from vllm_sim import launch_vllm_server, wait_for_vllm_server, make_vllm_call_llm

MODEL = "Qwen/Qwen3-0.6B"
log = "/next/u/stephhk/sycophancy/training_mitigations/logs/vllm_smoke_server.log"
proc, url = launch_vllm_server(MODEL, "cuda:0",
                               gpu_memory_utilization=0.85, log_path=log)
t0 = time.time()
wait_for_vllm_server(proc, url, log_path=log, timeout=900)
print(f"[smoke] server healthy at {url} after {time.time()-t0:.0f}s", flush=True)

call = make_vllm_call_llm(url, MODEL)
text, in_tok, out_tok = call(None, None,
    "Reply with the single token ACCEPT on the first line, then one short sentence.", 64)
print(f"[smoke] single call: text={text!r} in={in_tok} out={out_tok}", flush=True)
assert "ACCEPT" in text.upper(), f"unexpected reply: {text!r}"
assert "<think>" not in text, "enable_thinking=False was not honored"
assert in_tok > 0 and out_tok > 0

t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    results = list(ex.map(
        lambda i: call(None, None, f"Reply with the single token OK. (request {i})", 32)[0],
        range(32)))
print(f"[smoke] 32 concurrent calls in {time.time()-t0:.1f}s", flush=True)
assert len(results) == 32 and all(r for r in results)
proc.terminate()
print("[smoke] PASS", flush=True)
EOF
