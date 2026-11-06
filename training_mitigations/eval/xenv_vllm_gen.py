"""Cross-dataset held-out GENERATION for the four Qwen3-8B PlurPO adapters on a
batched offline vLLM engine (rebuttal, 2026-11-06).

Same experiment as `slurm_scripts/submit_xenv_qwen3_gen.sh` (each adapter on
the 3 envs it was NOT trained on -> 12 conditions), but one job per condition
samples the whole 1000-prompt / 1000-pair test split in one batched
`vllm.LLM.generate` call instead of 8-20 one-prompt-at-a-time HF shards.
Generation only: no gate, judge or classifier, no network.

PARITY WITH THE HF PATH (eval_endorsement_heldout.generate_one), by design:
  - prompt token ids come from llama_local.build_chat_input (same chat
    template, enable_thinking=False, add_special_tokens=False);
  - decoding: temperature 1.0, top_p 0.95 AND top_k 20. HF generate() keeps
    Qwen3's generation_config top_k=20 because the eval only overrides
    temperature/top_p, so top_k=20 is part of the HF protocol;
  - stop only on tokenizer.eos_token_id (<|im_end|>), like the HF path, which
    overrides generation_config's [<|im_end|>, <|endoftext|>] list. The engine
    is built with generation_config="vllm" so the checkpoint's defaults
    (temperature 0.6, extra stop id) can never leak in;
  - max new tokens 512 (OEQ/PAS) / 1536 (AITA/FLIP); decoded with
    skip_special_tokens and .strip().
  NOT identical: the random stream. HF seeds torch once per shard, so a draw
  depends on everything sampled before it; here each request gets its own seed
  gate_common.attempt_seed(text, 1, seed), so every draw is a pure function of
  (prompt text, seed) and independent of batching. Responses are therefore
  different temp-1.0 samples from the same distribution, not the same samples.

OUTPUT. Container <runs>/xenv_vllm_qwen3-8b_<train>_on_<eval>/ (adapter
symlinks + XENV_SOURCE.txt), kept apart from the HF xenv_qwen3-8b_* containers.
Files are split into the SAME shard layout and filenames/columns the HF path
writes (OEQ/PAS 8, AITA 10, FLIP 20 shards), so the deferred scoring pipeline
runs unchanged on either backend. Extra per-row diagnostics (n_tokens,
finish_reason, seed) go to the jsonl only.

Assumptions (asserted): the source adapter dir holds adapter_config.json with
r <= --max-lora-rank; every test id resolves (via the shared loaders); every
prompt fits max_model_len with its token budget; vLLM returns one output per
request, in request order; the container has no partial set of shard CSVs.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

_EVAL = os.path.dirname(os.path.abspath(__file__))
_TM = os.path.dirname(_EVAL)
sys.path.insert(0, os.path.join(os.path.dirname(_TM), "mitigations"))
sys.path.insert(0, _TM)
sys.path.insert(0, os.path.join(_TM, "core"))
sys.path.insert(0, _EVAL)

from gate_common import attempt_seed
from train_data import final_ids_path

SRC = {
    "oeq": "dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5",
    "pas": "dpo_lora_pas_qwen3-8b_P6_voice2_rpo0p5",
    "aita": "dpo_lora_aita_qwen3-8b_P6_voice2R2_rpo0p5",
    "flip": "dpo_lora_flip_qwen3-8b_P6_voice2R2_rpo0p5",
}
ENVNAME = {"oeq": "OEQ", "pas": "PAS", "aita": "AITA", "flip": "AITA-NTA-FLIP"}
NSH = {"oeq": 8, "pas": 8, "aita": 10, "flip": 20}
PREFIX = {"oeq": "responses", "pas": "responses",
          "aita": "aita_responses", "flip": "aita_responses"}
MAX_TOKENS = {"oeq": 512, "pas": 512, "aita": 1536, "flip": 1536}
TAG = "final_finaldata_temp1.0_seed0"
MODEL_ID = "Qwen/Qwen3-8B"


def conditions():
    """The 12 (train, eval) pairs in the submitter's loop order; index i is
    the Slurm array task id."""
    return [(tr, ev) for tr in SRC for ev in SRC if tr != ev]


def shard_csv_paths(container, ev):
    """Shard CSV paths the HF path would write for eval env `ev`."""
    n = NSH[ev]
    return [os.path.join(container, "eval", f"{PREFIX[ev]}_{TAG}_shard{s}of{n}.csv")
            for s in range(n)]


def load_rows(ev, limit):
    """Frozen test split rows for env `ev` via the HF drivers' own loaders
    (imported lazily: they pull in torch/peft). OEQ/PAS -> DataFrame[id,
    post_text]; AITA/FLIP -> held_out_rows' frame (is_asshole /
    flipped_post_text)."""
    env = ENVNAME[ev]
    ids_path = final_ids_path(env, "test")
    if ev in ("oeq", "pas"):
        from eval_endorsement_heldout import held_out_prompts
        pairs = held_out_prompts(env, None, ids_path)
        rows = pd.DataFrame(pairs, columns=["id", "post_text"])
    else:
        from eval_aita_label_heldout import held_out_rows
        rows = held_out_rows(env, None, ids_path)
    assert rows["id"].astype(str).is_unique, f"{env}: duplicate test ids"
    return rows.iloc[:limit].reset_index(drop=True) if limit else rows


def requests_for(rows, ev):
    """One request per generation: (row_idx, side, text). FLIP gets two per
    pair (original, flipped); every other env one (side None)."""
    out = []
    for i, r in rows.iterrows():
        if ev == "flip":
            out.append((i, "original", r["post_text"]))
            out.append((i, "flipped", r["flipped_post_text"]))
        else:
            out.append((i, None, r["post_text"]))
    return out


def assemble_records(rows, ev, reqs, gens):
    """Join generations back onto rows in the HF path's CSV schema.
    `gens[k]` = dict(response, n_tokens, finish_reason, seed) for reqs[k].
    Returns (csv_records, jsonl_records), both in row order."""
    assert len(gens) == len(reqs), (len(gens), len(reqs))
    by = {(i, side): g for (i, side, _), g in zip(reqs, gens)}
    csv_recs, js_recs = [], []
    for i, r in rows.iterrows():
        rid = str(r["id"])
        if ev in ("oeq", "pas"):
            g = by[(i, None)]
            csv_recs.append({"id": rid, "sentence": r["post_text"], "response": g["response"],
                             "winning_attempt": 1, "gate_passed": True, "n_tried": 1})
            js_recs.append({"id": rid, "prompt": r["post_text"], "gen_prompt": r["post_text"],
                            "rewritten": None, **g})
        elif ev == "aita":
            g = by[(i, None)]
            rec = {"id": rid, "post_text": r["post_text"], "rewritten_post": None,
                   "response": g["response"], "winning_attempt": 1, "gate_passed": True,
                   "n_tried": 1, "is_asshole": int(r["is_asshole"])}
            csv_recs.append(rec)
            js_recs.append({**rec, **{k: g[k] for k in ("n_tokens", "finish_reason", "seed", "last_token")}})
        else:
            go, gf = by[(i, "original")], by[(i, "flipped")]
            rec = {"id": rid, "post_text": r["post_text"],
                   "flipped_post_text": r["flipped_post_text"],
                   "rewritten_original": None, "rewritten_flipped": None,
                   "response_original": go["response"], "response_flipped": gf["response"],
                   "winning_attempt_original": 1, "winning_attempt_flipped": 1,
                   "gate_passed_original": True, "gate_passed_flipped": True,
                   "n_tried_original": 1, "n_tried_flipped": 1}
            csv_recs.append(rec)
            js_recs.append({**rec,
                            **{f"{k}_original": go[k] for k in ("n_tokens", "finish_reason", "seed", "last_token")},
                            **{f"{k}_flipped": gf[k] for k in ("n_tokens", "finish_reason", "seed", "last_token")}})
    return csv_recs, js_recs


def write_shards(container, ev, csv_recs, js_recs, summary):
    """Split records into the HF shard layout (eval_endorsement_heldout.
    shard_slice, contiguous over the frozen id order) and write per shard the
    csv + jsonl + summary json the HF driver writes."""
    from eval_endorsement_heldout import shard_slice
    n = NSH[ev]
    jsonl_name = "responses_{}.jsonl" if ev in ("oeq", "pas") else "responses_aita_{}.jsonl"
    summ_name = "summary_{}.json" if ev in ("oeq", "pas") else "summary_aita_{}.json"
    paths = shard_csv_paths(container, ev)
    for s in range(n):
        a, b = shard_slice(len(csv_recs), n, s)
        tag = f"{TAG}_shard{s}of{n}"
        out = os.path.join(container, "eval")
        with open(os.path.join(out, jsonl_name.format(tag)), "w") as f:
            for rec in js_recs[a:b]:
                f.write(json.dumps(rec) + "\n")
        with open(os.path.join(out, summ_name.format(tag)), "w") as f:
            json.dump({**summary, "num_shards": n, "shard_id": s, "n_rows": b - a}, f, indent=2)
        pd.DataFrame(csv_recs[a:b]).to_csv(paths[s], index=False)   # csv last = done marker


def make_container(runs, tr, ev, src, backend_note):
    """Create the condition container with adapter symlinks; refuse a partial
    set of shard CSVs. Returns (dir, already_done)."""
    d = os.path.join(runs, f"xenv_vllm_qwen3-8b_{tr}_on_{ev}")
    have = [os.path.exists(p) for p in shard_csv_paths(d, ev)]
    if all(have):
        return d, True
    assert not any(have), f"{d}: partial shard CSVs on disk -- inspect before rerunning"
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        link = os.path.join(d, fn)
        if os.path.islink(link):
            assert os.readlink(link) == os.path.join(src, fn), f"{link} points elsewhere"
        else:
            os.symlink(os.path.join(src, fn), link)
    with open(os.path.join(d, "XENV_SOURCE.txt"), "w") as f:
        f.write(f"train_env={tr}\neval_env={ENVNAME[ev]}\nsource_adapter={src}\n"
                f"backend={backend_note}\ncreated={datetime.now().isoformat()}\n")
    return d, False


def generate(token_prompts, seeds, max_tokens, args, lora_path):
    """Batched vLLM sampling with the parity settings in the module docstring.
    Returns one dict(response, n_tokens, finish_reason, seed) per prompt."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    llm = LLM(model=MODEL_ID, dtype="bfloat16", enable_lora=True,
              max_lora_rank=args.max_lora_rank, max_loras=1,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              generation_config="vllm", seed=args.seed)
    params = [SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                             max_tokens=max_tokens, seed=sd, skip_special_tokens=True)
              for sd in seeds]
    outs = llm.generate([TokensPrompt(prompt_token_ids=t) for t in token_prompts], params,
                        lora_request=LoRARequest("policy", 1, lora_path))
    assert len(outs) == len(token_prompts), (len(outs), len(token_prompts))
    res = []
    for o, t, sd in zip(outs, token_prompts, seeds):
        assert list(o.prompt_token_ids) == list(t), "vLLM output order/prompt mismatch"
        c = o.outputs[0]
        res.append({"response": c.text.strip(), "n_tokens": len(c.token_ids),
                    "finish_reason": c.finish_reason, "seed": sd,
                    "last_token": int(c.token_ids[-1]) if len(c.token_ids) else None})
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=int, required=True, help="condition index 0..11")
    p.add_argument("--runs", required=True, help="where containers are created")
    p.add_argument("--adapter-root", required=True, help="dir holding the 4 source adapters")
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N rows only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-lora-rank", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    args = p.parse_args()

    assert args.limit is None or "scratch" in args.runs, (
        "--limit (smoke) must write under a scratch --runs dir, never the real containers")
    conds = conditions()
    assert 0 <= args.index < len(conds), f"--index must be in [0,{len(conds)})"
    tr, ev = conds[args.index]
    src = os.path.join(args.adapter_root, SRC[tr])
    cfg = json.load(open(os.path.join(src, "adapter_config.json")))
    assert cfg["r"] <= args.max_lora_rank, f"adapter r={cfg['r']} > --max-lora-rank"
    assert cfg["base_model_name_or_path"] == MODEL_ID, cfg["base_model_name_or_path"]

    import vllm
    container, done = make_container(args.runs, tr, ev, src, f"vllm {vllm.__version__}")
    if done:
        print(f"[xenv-vllm] {container}: all {NSH[ev]} shard CSVs on disk -- skipping")
        return
    rows = load_rows(ev, args.limit)
    reqs = requests_for(rows, ev)
    print(f"[xenv-vllm] {tr} -> {ENVNAME[ev]}: {len(rows)} rows, {len(reqs)} generations, "
          f"max_tokens={MAX_TOKENS[ev]}, container={container}", flush=True)

    from transformers import AutoTokenizer
    from llama_local import build_chat_input
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    assert tok.convert_ids_to_tokens(tok.eos_token_id) == "<|im_end|>", tok.eos_token_id
    token_prompts = [build_chat_input(tok, text, None, "cpu")[0].tolist() for _, _, text in reqs]
    longest = max(len(t) for t in token_prompts)
    assert longest + MAX_TOKENS[ev] <= args.max_model_len, (
        f"longest prompt {longest} + {MAX_TOKENS[ev]} > max_model_len {args.max_model_len}")
    seeds = [attempt_seed(text, 1, args.seed) for _, _, text in reqs]

    gens = generate(token_prompts, seeds, MAX_TOKENS[ev], args, os.path.abspath(src))
    csv_recs, js_recs = assemble_records(rows, ev, reqs, gens)
    n_trunc = sum(g["finish_reason"] == "length" for g in gens)
    n_empty = sum(g["response"] == "" for g in gens)
    # Every non-truncated generation must have ended on <|im_end|> (the HF stop
    # set); any other final token would mean the stop configuration diverged.
    bad_stop = [g for g in gens if g["finish_reason"] == "stop" and g["last_token"] != tok.eos_token_id]
    assert not bad_stop, f"{len(bad_stop)} generations stopped on a non-<|im_end|> token"
    summary = {"env": ENVNAME[ev], "train_env": tr, "run_dir": os.path.abspath(container),
               "adapter_tag": "final", "adapter": os.path.abspath(src),
               "backend": f"vllm {vllm.__version__}", "decode": f"temp{args.temperature}",
               "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
               "max_tokens": MAX_TOKENS[ev], "seed": args.seed,
               "seeding": "per-request gate_common.attempt_seed(text, 1, seed)",
               "n_generations_total": len(gens), "n_truncated_total": n_trunc,
               "n_empty_total": n_empty, "limit": args.limit, "judged": False, "metrics": None}
    write_shards(container, ev, csv_recs, js_recs, summary)
    print(f"[xenv-vllm] done: {len(csv_recs)} rows / {len(gens)} generations, "
          f"truncated={n_trunc}, empty={n_empty}, median tokens="
          f"{sorted(g['n_tokens'] for g in gens)[len(gens) // 2]}", flush=True)


if __name__ == "__main__":
    main()
