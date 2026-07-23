"""Score held-out `finaldata` responses with the Skywork-Reward-V2 RMs.

One job scores every arm of one env with one RM (model loaded once). The user
turn is always the BARE task text (`sentence` / `post_text`) — the prompt-only
baselines inject their steer into the user turn at generation time, but scoring
with the steer included would ask the RM a different question per arm, so the
task is held fixed and only the response varies.

Each RM gets its OWN tokenizer/chat template (Llama-3.1 vs Qwen3); reusing the
policy tokenizer would silently corrupt the Llama RM's scores.

Writes <rundir>/eval/rm_<rm>_<tag>.csv with id, score, n_prompt_tok,
n_resp_tok, rm_input_truncated. n_resp_tok is carried so reward can be checked
against the 512-cap truncation rate downstream.
"""
import argparse, csv, glob, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

csv.field_size_limit(10**9)

# torch 2.10's SDPA dispatches to a cuDNN attention backend that fails to init on
# this cluster (RuntimeError: CUDNN_STATUS_NOT_INITIALIZED). Force the other
# kernels; math is kept as a always-works fallback.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

CACHE = "/projects/m000254/hf_cache"
RUNS = "/projects/m000254/runs"

RMS = {
    "llama": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "qwen": "Skywork/Skywork-Reward-V2-Qwen3-8B",
}

BASE_STEERS = ["base", "base-becritical", "base-explicit",
               "base-sharma", "base-dontbesyco", "base-perspshiftassess",
               "base-besycophantic"]

# env -> (baseline rundir, [trained rundirs], response prefix, nshards, prompt col)
ENVS = {
    "OEQ": ("baseline_final_OEQ", [
        "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
        "dpo_lora_oeq_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo",
        "dpo_lora_oeq_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo_b05",
        "dpo_lora_oeq_qwen3-8b_n1000_neutral_final",
        "dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final",
        "sft_lora_oeq_qwen3-8b_becritical_final",
    ], "responses", 2, "sentence"),
    "PAS": ("baseline_final_PAS", [
        "dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
        "dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo",
        "dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo_b05",
        "dpo_lora_pas_qwen3-8b_n1000_neutral_final",
        "dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final",
        "sft_lora_pas_qwen3-8b_becritical_final",
    ], "responses", 2, "sentence"),
    "AITA": ("baseline_final_AITA", [
        "dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
    ], "aita_responses", 1, "post_text"),
}


def conditions(env):
    """-> [(label, rundir, tag)] for every arm of `env`."""
    base, trained, _, _, _ = ENVS[env]
    out = [(s, base, s) for s in BASE_STEERS]
    out += [(r, r, "final") for r in trained]
    return out


def load_rows(rundir, tag, prefix, nshards, prompt_col):
    rows, seen = [], set()
    for s in range(nshards):
        f = f"{RUNS}/{rundir}/eval/{prefix}_{tag}_finaldata_temp1.0_seed0_shard{s}of{nshards}.csv"
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                rows.append({"id": r["id"], "prompt": r[prompt_col],
                             "response": r["response"]})
    return rows


@torch.no_grad()
def score(rows, tok, model, max_length, batch_tokens, device):
    # Render every conversation, then sort by length so padding is cheap.
    rendered = []
    for i, r in enumerate(rows):
        conv = [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]}]
        text = tok.apply_chat_template(conv, tokenize=False)
        n_p = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
        n_r = len(tok(r["response"], add_special_tokens=False)["input_ids"])
        rendered.append((i, text, n_p, n_r))
    order = sorted(range(len(rendered)), key=lambda k: len(rendered[k][1]))

    out = [None] * len(rows)
    buf = []
    def flush():
        if not buf:
            return
        texts = [rendered[k][1] for k in buf]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, add_special_tokens=False).to(device)
        logits = model(**enc).logits[:, 0].float().cpu().tolist()
        for k, s in zip(buf, logits):
            i, _, n_p, n_r = rendered[k]
            full = len(tok(rendered[k][1], add_special_tokens=False)["input_ids"])
            out[i] = {"id": rows[i]["id"], "score": s, "n_prompt_tok": n_p,
                      "n_resp_tok": n_r,
                      "rm_input_truncated": int(full > max_length)}
        buf.clear()

    cur_max = 0
    for k in order:
        L = min(len(tok(rendered[k][1], add_special_tokens=False)["input_ids"]), max_length)
        cand = max(cur_max, L) * (len(buf) + 1)
        if buf and cand > batch_tokens:
            flush(); cur_max = 0
        buf.append(k); cur_max = max(cur_max, L)
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm", required=True, choices=sorted(RMS))
    ap.add_argument("--env", required=True, choices=sorted(ENVS))
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    _, _, prefix, nshards, prompt_col = ENVS[args.env]
    conds = conditions(args.env)

    todo = [c for c in conds
            if args.overwrite or not os.path.exists(
                f"{RUNS}/{c[1]}/eval/rm_{args.rm}_{c[2]}.csv")]
    if not todo:
        print(f"[rm] {args.rm}/{args.env}: all {len(conds)} arms already scored")
        return

    # Fail before loading 16GB of weights if any arm's responses are missing.
    for label, rundir, tag in todo:
        load_rows(rundir, tag, prefix, nshards, prompt_col)

    name = RMS[args.rm]
    print(f"[rm] loading {name}", flush=True)
    tok = AutoTokenizer.from_pretrained(name, cache_dir=CACHE)
    # sdpa, not flash_attention_2: flash-attn is not installed in envs/llm126.
    model = AutoModelForSequenceClassification.from_pretrained(
        name, cache_dir=CACHE, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(args.device).eval()
    model.config.pad_token_id = tok.pad_token_id

    for label, rundir, tag in todo:
        rows = load_rows(rundir, tag, prefix, nshards, prompt_col)
        res = score(rows, tok, model, args.max_length, args.batch_tokens, args.device)
        out = f"{RUNS}/{rundir}/eval/rm_{args.rm}_{tag}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "score", "n_prompt_tok",
                                               "n_resp_tok", "rm_input_truncated"])
            w.writeheader(); w.writerows(res)
        mean = sum(r["score"] for r in res) / len(res)
        ntr = sum(r["rm_input_truncated"] for r in res)
        print(f"[rm] {args.rm:5s} {args.env:4s} {label:52s} "
              f"n={len(res):4d} mean={mean:+.4f} rm_trunc={ntr}", flush=True)


if __name__ == "__main__":
    main()
