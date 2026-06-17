"""Stage-1 parity check for the single-engine vLLM-policy design.

Verifies, on GPU, that:
  (A) vLLM serving the BASE matches the HF base model's logprobs, and
  (B) vLLM serving BASE + adapter_iter_1 (the policy LoRA) matches the HF
      base model with that PEFT adapter loaded.

Method (TEACHER-FORCED, so there is no greedy-divergence cascade): for each
prompt and each model, HF greedy-decodes a reference completion and reports its
per-token logprobs; we then score that SAME completion under vLLM (echo
logprobs) and compare logprobs position-by-position on identical tokens. (An
earlier version let each engine free-run greedy and compared positionally;
bf16 argmax tie-flips made the sequences diverge and inflated the diff even
though the underlying logprobs agreed to ~0.015 nats. Teacher forcing removes
that artifact.) We also confirm base != adapter greedy text so an unapplied
adapter is caught.

Parity-ensuring choices (mirror llama_local / vllm_sim):
  - vLLM --dtype bfloat16 to match load_llama bf16.
  - Same Qwen3 template, add_generation_prompt=True, enable_thinking=False,
    add_special_tokens=False (the template already inserts specials) on BOTH
    sides, so the tokenization the logprobs are computed over is identical.
  - HF base via PeftModel.disable_adapter(); adapter via the same PeftModel.
  - vLLM base via model=<model_id>; adapter via model="policy" (runtime-loaded).
    --max-lora-rank 16 >= r; alpha/scaling read from adapter_config.json (==PEFT).

Run (2 GPUs): HF on --device, vLLM on --sim-device.
    python training_mitigations/build_vllm_parity_check.py \
        --adapter training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_selfpref/adapter_iter_1
"""

import argparse
import json
import os
import sys
import urllib.request

import torch

_TM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_TM, "core"))
sys.path.insert(0, os.path.join(_TM, "..", "mitigations"))

from llama_local import DEFAULT_HF_MODEL_ID, load_llama, build_chat_input
from vllm_sim import launch_vllm_server, wait_for_vllm_server, load_lora_adapter
from peft import PeftModel

PROMPTS = [
    "My roommate keeps eating my food without asking. I'm thinking of just "
    "labeling everything and putting a lock on the fridge shelf. Is that a "
    "reasonable way to handle it?",
    "I told my friend I couldn't make their birthday dinner because I was "
    "tired, but really I just didn't feel like going. Was that okay?",
    "My sister borrowed money months ago and hasn't paid me back. I posted "
    "about it vaguely on social media. Did I do anything wrong?",
]
K = 32
LOG = "training_mitigations/sycoharm_runs/parity_vllm.log"


def render(tok, prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        tokenize=False, enable_thinking=False)


def hf_greedy(model, tok, prompt, device):
    """Greedy-decode K tokens; return (text, [logprob per generated token])."""
    ids = build_chat_input(tok, prompt, system=None, device=device)
    in_len = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=K, do_sample=False, num_return_sequences=1,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            output_scores=True, return_dict_in_generate=True)
    gen = out.sequences[0, in_len:]
    lps = []
    for t, score in enumerate(out.scores):
        if t >= gen.shape[0]:
            break
        lp = torch.log_softmax(score[0].float(), dim=-1)
        lps.append(lp[gen[t]].item())
    return tok.decode(gen, skip_special_tokens=True), lps


def vllm_teacher_logprobs(base_url, model_name, tok, prompt, completion, timeout=600):
    """Per-token logprobs of `completion` under the vLLM model, teacher-forced
    via /v1/completions echo. Returns the logprob list for the completion tokens
    (sliced off the rendered-prompt prefix)."""
    rendered = render(tok, prompt)
    full = rendered + completion
    n_prompt = len(tok(rendered, add_special_tokens=False).input_ids)
    n_full = len(tok(full, add_special_tokens=False).input_ids)
    body = json.dumps({
        "model": model_name, "prompt": full, "max_tokens": 1, "echo": True,
        "logprobs": 1, "temperature": 0,
        "add_special_tokens": False,  # template already added specials (== HF)
    }).encode()
    req = urllib.request.Request(
        base_url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    tl = data["choices"][0]["logprobs"]["token_logprobs"]  # [null, lp1, ...]
    return [x for x in tl[n_prompt:n_full] if x is not None]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sim-device", default="cuda:1")
    args = ap.parse_args()
    assert os.path.exists(os.path.join(args.adapter, "adapter_model.safetensors"))

    print("[hf] loading base + PEFT adapter...")
    base, tok = load_llama(args.model_id, device=args.device)
    peft = PeftModel.from_pretrained(base, args.adapter)
    peft.eval()

    print("[vllm] launching --enable-lora server (bf16)...")
    proc, base_url = launch_vllm_server(
        args.model_id, device=args.sim_device, enable_lora=True,
        max_lora_rank=16, dtype="bfloat16", log_path=LOG)
    wait_for_vllm_server(proc, base_url, LOG)
    load_lora_adapter(base_url, "policy", os.path.abspath(args.adapter))

    worst_base = worst_adp = 0.0
    n_effect = 0
    for i, p in enumerate(PROMPTS):
        print(f"\nPROMPT {i}: {p[:70]}...")
        # BASE: HF (adapter disabled) reference completion, scored under vLLM base
        with peft.disable_adapter():
            b_text, b_lp = hf_greedy(peft, tok, p, args.device)
        vb_lp = vllm_teacher_logprobs(base_url, args.model_id, tok, p, b_text)
        nb = min(len(b_lp), len(vb_lp))
        db = [abs(b_lp[j] - vb_lp[j]) for j in range(nb)]
        mb = sum(db) / nb if nb else float("nan")
        print(f"  [base   ] mean|Δlogprob|={mb:.4f} max={max(db) if db else 0:.4f} (n={nb})")
        # ADAPTER: HF policy reference completion, scored under vLLM model=policy
        a_text, a_lp = hf_greedy(peft, tok, p, args.device)
        va_lp = vllm_teacher_logprobs(base_url, "policy", tok, p, a_text)
        na = min(len(a_lp), len(va_lp))
        da = [abs(a_lp[j] - va_lp[j]) for j in range(na)]
        ma = sum(da) / na if na else float("nan")
        print(f"  [adapter] mean|Δlogprob|={ma:.4f} max={max(da) if da else 0:.4f} (n={na})")
        worst_base = max(worst_base, mb)
        worst_adp = max(worst_adp, ma)
        n_effect += (b_text.strip() != a_text.strip())

    print("\n==== SUMMARY ====")
    print(f"worst base mean|Δlogprob|    = {worst_base:.4f}")
    print(f"worst adapter mean|Δlogprob| = {worst_adp:.4f}")
    print(f"base != adapter greedy text  : {n_effect}/{len(PROMPTS)} (LoRA has effect)")
    ok = worst_base < 0.05 and worst_adp < 0.05 and n_effect >= 1
    print("PARITY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
