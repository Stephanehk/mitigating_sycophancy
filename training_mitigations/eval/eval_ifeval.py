"""Generate Qwen3-8B responses over the IFEval instruction-following set.

IFEval (google/IFEval, Zhou et al. 2023) is a capability / regression check for
our post-training mitigations: it has 541 prompts carrying *verifiable*
instructions (word counts, casing, JSON, keyword frequency, ...) that are scored
PROGRAMMATICALLY (no LLM judge). This script ONLY generates one response per
prompt for one model configuration; scoring lives in score_ifeval.py.

The model configuration mirrors eval_endorsement_heldout.py exactly, so the
IFEval numbers are comparable to the sycophancy numbers for the same condition:
  - --base                : bare Qwen3-8B (no adapter).
  - --adapter <dir>       : Qwen3-8B + a trained LoRA adapter (e.g. a mixed_P5 run).
  - --be-critical         : bare model, trainer.CRITICAL_SUFFIX appended to the
                            prompt the model sees ("\\n\\nBe critical.").
  - --gen-suffix S --gen-tag T : bare model with "\\n\\n"+S appended (e.g.
                            "Don't be sycophantic." / dontbesyco), matching the
                            prompt-only baselines.
The suffix is added ONLY to what the model is shown; the response file records
the ORIGINAL prompt under "prompt" because the IFEval scorer joins responses to
instructions by the exact original prompt string.

Decoding defaults to GREEDY (the standard IFEval setting); pass --temperature /
--top-p with --sample for stochastic decoding.

Reuses the model-loading and generation helpers from eval_endorsement_heldout
(generate_one / attach_adapter / shard_slice) and llama_local (load_llama) so
there is a single source of truth for how the policy is loaded and sampled.

Output: <out-dir>/responses_ifeval_<tag>[_shard<i>of<n>].jsonl, one JSON object
per line with keys: key, prompt (original), gen_prompt (what the model saw),
response. score_ifeval.py reads these (concatenating shards) to score.
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # training_mitigations/
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, os.path.join(_HERE, "eval"))

from llama_local import DEFAULT_HF_MODEL_ID, load_llama
# Single source of truth for policy load + sampling + sharding.
from eval_endorsement_heldout import generate_one, attach_adapter, shard_slice
from ifeval_data import DEFAULT_DATA, load_records

DEFAULT_OUT_DIR = os.path.join(_HERE, "sycoharm_runs", "ifeval", "responses")


def build_tag(args):
    """Output tag identifying the evaluated configuration (mirrors the
    eval_endorsement_heldout naming so conditions line up across evals)."""
    if args.base:
        tag = "base"
    else:
        adapter_dir = args.adapter
        tag = os.path.basename(os.path.normpath(adapter_dir))
    if args.be_critical:
        tag = f"{tag}-becritical"
    elif args.gen_suffix:
        tag = f"{tag}-{args.gen_tag}"
    return tag


def gen_suffix_for(args):
    """The exact text appended to the prompt the model sees (single source of
    truth: trainer.CRITICAL_SUFFIX for --be-critical)."""
    if args.be_critical:
        from trainer import CRITICAL_SUFFIX
        return CRITICAL_SUFFIX
    if args.gen_suffix:
        return "\n\n" + args.gen_suffix
    return ""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", action="store_true",
                     help="evaluate the bare base model (no adapter)")
    src.add_argument("--adapter", default=None,
                     help="dir with a PEFT LoRA adapter to attach to the base")
    p.add_argument("--be-critical", action="store_true",
                   help="append trainer.CRITICAL_SUFFIX to the model input")
    p.add_argument("--gen-suffix", default=None,
                   help="append '\\n\\n'+this to the model input (prompt-only "
                        "baseline, e.g. \"Don't be sycophantic.\")")
    p.add_argument("--gen-tag", default=None,
                   help="short tag for --gen-suffix (required with it)")
    p.add_argument("--tag", default=None,
                   help="override the output filename stem (default: derived "
                        "from the model source + suffix). Lets a caller use one "
                        "consistent stem for gen and score_ifeval.")
    p.add_argument("--dataset", default=DEFAULT_DATA,
                   help="local IFEval jsonl (default) or an HF dataset name")
    p.add_argument("--split", default="train")
    p.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default=None,
                   help="e.g. 'auto' to split the 8B across visible GPUs")
    p.add_argument("--max-new-tokens", type=int, default=1280,
                   help="IFEval prompts can request 300+ words; keep generous")
    p.add_argument("--sample", action="store_true",
                   help="stochastic decoding (default: greedy)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--limit", type=int, default=None,
                   help="generate only the first N prompts (smoke test)")
    args = p.parse_args()

    assert not (args.be_critical and args.gen_suffix), \
        "--be-critical and --gen-suffix are mutually exclusive"
    if args.gen_suffix:
        assert args.gen_tag, "--gen-suffix requires --gen-tag"
    greedy = not args.sample

    tag = args.tag or build_tag(args)
    suffix = gen_suffix_for(args)
    decode_tag = "greedy" if greedy else f"temp{args.temperature}"

    # 1. Prompts (fixed dataset order; shard AFTER --limit so they compose).
    items = [(r["key"], r["prompt"]) for r in load_records(args.dataset, args.split)]
    if args.limit is not None:
        items = items[:args.limit]
    n_total = len(items)
    shard_tag = ""
    if args.num_shards > 1:
        s, e = shard_slice(n_total, args.num_shards, args.shard_id)
        items = items[s:e]
        shard_tag = f"_shard{args.shard_id}of{args.num_shards}"
        print(f"[ifeval] shard {args.shard_id}/{args.num_shards}: prompts [{s}:{e}) of {n_total}")
    print(f"[ifeval] dataset={args.dataset} prompts={len(items)} tag={tag} decode={decode_tag}")

    # 2. Load policy (mirrors eval_endorsement_heldout).
    placement = f"device_map={args.device_map}" if args.device_map else args.device
    print(f"[load] base {args.model_id} -> {placement}")
    model, tokenizer = load_llama(model_id=args.model_id, device=args.device,
                                  cache_dir=args.cache_dir, device_map=args.device_map)
    if args.base:
        print("[load] no adapter (base model)")
    else:
        print(f"[load] attaching adapter {args.adapter}")
        model = attach_adapter(model, args.adapter)

    # 3. Generate one response per prompt.
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"responses_ifeval_{tag}_{decode_tag}_seed{args.seed}{shard_tag}.jsonl")
    with open(out_path, "w") as f:
        for k, (key, prompt) in enumerate(items):
            gen_prompt = prompt + suffix
            resp = generate_one(model, tokenizer, gen_prompt, args.max_new_tokens,
                                 args.temperature, args.top_p, greedy)
            f.write(json.dumps({"key": key, "prompt": prompt,
                                "gen_prompt": gen_prompt, "response": resp}) + "\n")
            f.flush()
            if (k + 1) % 50 == 0:
                print(f"[gen] {k + 1}/{len(items)}")
    print(f"[gen] wrote {out_path}")


if __name__ == "__main__":
    main()
