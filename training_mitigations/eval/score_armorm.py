"""Score held-out `finaldata` responses with the ArmoRM-Llama3-8B-v0.1 RM.

Companion to score_reward_models.py (Skywork). Same envs/arms/tasks — only the
reward model differs. The user turn is the BARE task text (`sentence`/`post_text`),
identical to the Skywork scoring, so ArmoRM sees the same (task, response) pairs.

ArmoRM is a multi-objective RM: its forward returns a 19-dim reward vector
(`out.rewards`) plus a MoE-gated scalar (`out.score`, ~0-1). We persist ALL 19
dimensions per response plus the gated scalar. Per-dim column names come from the
model config's `label2id` (helpsteer-*, ultrafeedback-*, code-*, ...).

Loading needs trust_remote_code (custom LlamaForRewardModelWithGating). Two hard
requirements from the custom forward (modeling_custom.py):
  * RIGHT padding — the reward token is picked as (first_pad_index - 1), so pads
    must be on the right; we force tok.padding_side="right".
  * The prompt is rendered with ArmoRM's OWN Llama-3 chat template; the gating
    head locates the assistant header token pattern and raises if it is absent.

max_length is 4096 (ArmoRM's native window), truncation on the right.

Writes <rundir>/eval/rm_armorm_<tag>.csv with:
  id, score, <19 dim columns>, n_prompt_tok, n_resp_tok, rm_input_truncated
"""
import argparse, csv, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the exact env/arm/task wiring from the Skywork scorer.
from score_reward_models import RUNS, ENVS, conditions, load_rows

csv.field_size_limit(10**9)

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

CACHE = "/projects/m000254/hf_cache"
MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
MAX_LENGTH = 4096
CAP = {"OEQ": 512, "PAS": 512, "AITA": 1536}

# Authoritative objective order from the model card's `attributes` list. The
# config's id2label is only {0: LABEL_0} (num_labels==1), so the 19 names are
# NOT recoverable from the config — they are pinned here, index-aligned to the
# regression layer's output columns.
DIMS = [
    "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
    "helpsteer-complexity", "helpsteer-verbosity", "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following", "ultrafeedback-truthfulness",
    "ultrafeedback-honesty", "ultrafeedback-helpfulness", "beavertails-is_safe",
    "prometheus-score", "argilla-overall_quality", "argilla-judge_lm",
    "code-complexity", "code-style", "code-explanation",
    "code-instruction-following", "code-readability",
]


def dim_names(model):
    """19 objective names in index order (see DIMS)."""
    n = getattr(model.config, "num_objectives", 19)
    assert len(DIMS) == n, f"DIMS has {len(DIMS)} names but model has {n} objectives"
    return DIMS


@torch.no_grad()
def score(rows, tok, model, names, device, batch_tokens=8192):
    """Return per-row dict: id, score, {dim: value}, n_prompt_tok, n_resp_tok,
    rm_input_truncated. Batches by token budget; RIGHT padding (ArmoRM req)."""
    rendered = []
    for i, r in enumerate(rows):
        conv = [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]}]
        text = tok.apply_chat_template(conv, tokenize=False)
        n_p = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
        n_r = len(tok(r["response"], add_special_tokens=False)["input_ids"])
        full = len(tok(text, add_special_tokens=False)["input_ids"])
        rendered.append((i, text, n_p, n_r, full))
    order = sorted(range(len(rendered)), key=lambda k: rendered[k][4])

    out = [None] * len(rows)
    buf = []

    def flush():
        if not buf:
            return
        texts = [rendered[k][1] for k in buf]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LENGTH, add_special_tokens=False).to(device)
        res = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        scores = res.score.float().cpu().tolist()
        rewards = res.rewards.float().cpu().tolist()
        for pos, k in enumerate(buf):
            i, _, n_p, n_r, full = rendered[k]
            out[i] = {"id": rows[i]["id"], "score": scores[pos],
                      "_rewards": rewards[pos], "n_prompt_tok": n_p,
                      "n_resp_tok": n_r,
                      "rm_input_truncated": int(full > MAX_LENGTH)}
        buf.clear()

    cur_max = 0
    for k in order:
        L = min(rendered[k][4], MAX_LENGTH)
        cand = max(cur_max, L) * (len(buf) + 1)
        if buf and cand > batch_tokens:
            flush(); cur_max = 0
        buf.append(k); cur_max = max(cur_max, L)
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(ENVS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N rows/arm")
    args = ap.parse_args()

    _, _, prefix, nshards, prompt_col = ENVS[args.env]
    conds = conditions(args.env)
    todo = [c for c in conds
            if args.overwrite or not os.path.exists(
                f"{RUNS}/{c[1]}/eval/rm_armorm_{c[2]}.csv")]
    if not todo:
        print(f"[armorm] {args.env}: all {len(conds)} arms already scored")
        return

    for label, rundir, tag in todo:  # fail fast before loading weights
        load_rows(rundir, tag, prefix, nshards, prompt_col)

    print(f"[armorm] loading {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE, use_fast=True)
    tok.padding_side = "right"  # ArmoRM picks the reward token as first_pad-1
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, cache_dir=CACHE, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    names = dim_names(model)
    print(f"[armorm] {len(names)} dims: {names}", flush=True)

    fields = (["id", "score"] + names + ["n_prompt_tok", "n_resp_tok",
                                         "rm_input_truncated"])
    for label, rundir, tag in todo:
        rows = load_rows(rundir, tag, prefix, nshards, prompt_col)
        if args.limit:
            rows = rows[:args.limit]
        res = score(rows, tok, model, names, args.device, args.batch_tokens)
        outf = f"{RUNS}/{rundir}/eval/rm_armorm_{tag}.csv"
        with open(outf, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in res:
                row = {k: r[k] for k in ("id", "score", "n_prompt_tok",
                                         "n_resp_tok", "rm_input_truncated")}
                for j, nm in enumerate(names):
                    row[nm] = r["_rewards"][j]
                w.writerow(row)
        mean = sum(r["score"] for r in res) / len(res)
        ntr = sum(r["rm_input_truncated"] for r in res)
        print(f"[armorm] {args.env:4s} {label:52s} n={len(res):4d} "
              f"score_mean={mean:+.4f} rm_trunc={ntr}", flush=True)


if __name__ == "__main__":
    main()
