"""Greedy parity check: does vLLM + LoRARequest reproduce HF + PEFT for one
PlurPO adapter? (rebuttal, 2026-11-06; companion to xenv_vllm_gen.py)

Greedy decoding removes sampling randomness, so the only differences left are
numerics. For N test prompts it decodes greedily, `--max-tokens` each, with
  (a) HF + PEFT  -- the reference: eval_endorsement_heldout.generate_one, i.e.
                    the exact HF eval path (load_llama + attach_adapter);
  (b) vLLM + LoRA -- xenv_vllm_gen's prompt ids + LoRARequest;
  (c) vLLM base   -- same engine, no LoRA (negative control).
and reports, per prompt, the shared-prefix length in tokens between (a)/(b)
and (a)/(c). PASS requires (b) to match (a) on a long prefix for most prompts
(bf16 kernels can flip a near-tied argmax late) AND (c) to diverge, which shows
the adapter is actually being applied.

Assumptions: one free GPU with room for HF (~16 GB) then vLLM; HF is freed
before vLLM starts. Prints a table and writes it as json next to --out.
"""

import argparse
import gc
import json
import os
import sys

_EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EVAL)
import xenv_vllm_gen as X  # noqa: E402  (sets the repo sys.path)


def shared_prefix(a, b):
    """Number of leading tokens two id lists share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def hf_token_logprobs(model, prompt_ids, cont_ids):
    """Teacher-forced log p(cont_ids[j] | prompt + cont_ids[:j]) under the HF
    model, one forward pass; returns a list of floats (len(cont_ids))."""
    import torch
    ids = torch.tensor([prompt_ids + cont_ids], device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    n = len(prompt_ids)
    return [float(lp[n - 1 + j, t]) for j, t in enumerate(cont_ids)]


def vllm_token_logprobs(llm, prompt_ids, cont_ids, lora_request):
    """Teacher-forced logprobs of cont_ids under vLLM via prompt_logprobs
    (the logprob of each actual next token is always included)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    o = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids + cont_ids)], sp,
                     lora_request=lora_request)[0]
    n = len(prompt_ids)
    return [float(o.prompt_logprobs[n + j][t].logprob) for j, t in enumerate(cont_ids)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--env", default="pas", choices=list(X.SRC))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    from eval_endorsement_heldout import generate_one, attach_adapter
    from llama_local import load_llama, build_chat_input

    rows = X.load_rows(args.env, args.n)
    texts = [t for _, _, t in X.requests_for(rows, args.env)]

    model, tok = load_llama(model_id=X.MODEL_ID, device="cuda:0",
                            cache_dir=os.environ.get("HF_HOME"))
    model = attach_adapter(model, args.adapter)
    hf = [generate_one(model, tok, t, args.max_tokens, 1.0, 0.95, greedy=True) for t in texts]
    hf_ids = [tok(h, add_special_tokens=False).input_ids for h in hf]
    prompt_ids = [build_chat_input(tok, t, None, "cpu")[0].tolist() for t in texts]
    # Teacher-forced HF logprobs of the HF greedy continuation (LoRA on).
    hf_lp = [hf_token_logprobs(model, pi, ci) for pi, ci in zip(prompt_ids, hf_ids)]
    # HF base greedy (adapter disabled) -- control for backend numerics alone.
    with model.disable_adapter():
        hf_base = [generate_one(model, tok, t, args.max_tokens, 1.0, 0.95, greedy=True)
                   for t in texts]
    hf_base_ids = [tok(h, add_special_tokens=False).input_ids for h in hf_base]
    del model
    gc.collect()
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    prompts = [TokensPrompt(prompt_token_ids=pi) for pi in prompt_ids]
    llm = LLM(model=X.MODEL_ID, dtype="bfloat16", enable_lora=True, max_lora_rank=16,
              max_loras=1, max_model_len=8192, gpu_memory_utilization=0.85,
              generation_config="vllm")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=True)
    lreq = LoRARequest("policy", 1, os.path.abspath(args.adapter))
    lora = llm.generate(prompts, sp, lora_request=lreq)
    v_lp = [vllm_token_logprobs(llm, pi, ci, lreq) for pi, ci in zip(prompt_ids, hf_ids)]
    base = llm.generate(prompts, sp)
    # Compare on re-tokenized decoded text for all three, so every side is in
    # the same (decode -> encode) space as the HF reference.
    enc = lambda o: tok(o.outputs[0].text.strip(), add_special_tokens=False).input_ids
    table = []
    for i in range(len(texts)):
        lo, ba = enc(lora[i]), enc(base[i])
        d = [abs(a - b) for a, b in zip(hf_lp[i], v_lp[i])]
        table.append({"i": i, "hf_len": len(hf_ids[i]), "lora_len": len(lo),
                      "prefix_hf_lora": shared_prefix(hf_ids[i], lo),
                      "prefix_hf_base": shared_prefix(hf_ids[i], ba),
                      "prefix_hfbase_vllmbase": shared_prefix(hf_base_ids[i], ba),
                      "mean_abs_dlogp": sum(d) / len(d), "max_abs_dlogp": max(d),
                      "exact_hf_lora": hf_ids[i] == lo})
    for r in table:
        print(r)
    n_exact = sum(r["exact_hf_lora"] for r in table)
    print(f"[parity] exact HF==vLLM+LoRA: {n_exact}/{len(table)}; "
          f"median prefix HF~LoRA={sorted(r['prefix_hf_lora'] for r in table)[len(table)//2]}, "
          f"HF~base={sorted(r['prefix_hf_base'] for r in table)[len(table)//2]}, "
          f"HFbase~vLLMbase={sorted(r['prefix_hfbase_vllmbase'] for r in table)[len(table)//2]}; "
          f"teacher-forced |dlogp| mean={sum(r['mean_abs_dlogp'] for r in table)/len(table):.4f} "
          f"max={max(r['max_abs_dlogp'] for r in table):.3f}")
    with open(args.out, "w") as f:
        json.dump({"adapter": args.adapter, "env": args.env, "max_tokens": args.max_tokens,
                   "rows": table, "hf_text": hf, "lora_text": [o.outputs[0].text for o in lora],
                   "base_text": [o.outputs[0].text for o in base]}, f, indent=2)


if __name__ == "__main__":
    main()
