"""Drive `our_mitigation_v2.run_mitigation` against a locally-loaded
8B-scale chat model (default `Qwen/Qwen3-8B`) over a handful of OEQ
prompts, printing every intermediate step so a human can read off the
model's behavior at each stage of the pipeline.

This is a first capability probe before any training intervention is
written: the goal is to see how an open 8B handles the four-call v2
sequence (identify stakeholders → filter → per-stakeholder concerns →
judge) end-to-end, on the same OEQ data the API-model harness uses.

What it does:
  1. Load the model via `llama_local.load_llama` (bf16, one CUDA
     device). The loader name is historical — it works for any
     AutoModelForCausalLM-compatible checkpoint.
  2. Build a `call_llm`-compatible closure with `make_call_llm` and
     monkey-patch both `our_mitigation.call_llm` and
     `our_mitigation_v2.call_llm` so the harness reuses the local model
     for all 4 calls per post.
  3. Sample N OEQ prompts uniformly (default N=5, seed=0) — the same
     `sample_uniform` helper the API harness uses, so prompt indices
     line up across runs.
  4. For each prompt, call `our_mitigation_v2.run_mitigation` with
     `env_name="OEQ"` and print:
         STEP 1   identify stakeholders (raw + parsed candidates)
         STEP 1.5 filter stakeholders (raw + kept + dropped)
         STEP 2   per-stakeholder CONCERNS / SUPPORTS (raw + parsed)
         STEP 3   judge synthesis (raw + parsed reasoning + advice)
     plus token totals at the end of each post and a grand total.

Skips the OEQ action-endorsement judge (which would need an OpenRouter
or Claude client) — this script is for capability inspection, not
metric reporting.

Run under `srun` / `sbatch` on a node with a GPU (see
`run_test_v2_llama.sbatch`).
"""

import argparse
import os
import sys

# Make `mitigations/` importable so `our_mitigation_v2` (and its
# `from explore_mitigations import ...` indirections) resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import our_mitigation
import our_mitigation_v2
from explore_mitigations import (
    OEQ_DATA_CSV,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    load_oeq,
    sample_uniform,
)

from llama_local import DEFAULT_HF_MODEL_ID, load_llama, make_call_llm
import weak_pipeline


def _print_one_post_trace(env_name, post_id, post_text, result):
    """Print the v2 run trace for a single post in the same shape as
    `mitigations/test_our_mitigation_v2.py`.

    Inputs:
      env_name: passed through `our_mitigation_v2.run_mitigation`. We
                only support "OEQ" here, but the print logic treats
                anything non-AITA as open-ended (advice in <response>).
      post_id, post_text: identification / text of the prompt.
      result: dict returned by `run_mitigation` (must contain
              `candidate_stakeholders`, `stakeholders`,
              `raw_stakeholders`, `raw_filtered_stakeholders`,
              `concerns_entries`, `judge_raw`, `judge_reasoning`,
              `judge_output`, and token counts).
    Returns nothing; effects are stdout.
    """
    print(f"=== {env_name} post id={post_id} ===")
    print(post_text)
    print()

    candidates = result["candidate_stakeholders"]
    kept = result["stakeholders"]
    dropped = [s for s in candidates if s not in kept]
    added = [s for s in kept if s not in candidates]

    print("=== STEP 1: identify stakeholders ===")
    print("--- raw LLM response ---")
    print(result["raw_stakeholders"])
    print(f"--- parsed candidates ({len(candidates)}) ---")
    for s in candidates:
        print(f"  - {s}")
    print()

    print("=== STEP 1.5: filter stakeholders ===")
    print("--- raw LLM response ---")
    print(result["raw_filtered_stakeholders"])
    print(f"--- kept ({len(kept)}) ---")
    for s in kept:
        print(f"  + {s}")
    print(f"--- dropped ({len(dropped)}) ---")
    for s in dropped:
        print(f"  - {s}")
    if added:
        print(f"--- in kept but not a verbatim candidate ({len(added)}) ---")
        for s in added:
            print(f"  ? {s}")
    print()

    print("=== STEP 2: per-stakeholder concerns + supports ===")
    n = len(result["concerns_entries"])
    for i, e in enumerate(result["concerns_entries"], 1):
        print(f"--- stakeholder {i}/{n}: {e['stakeholder']} ---")
        print(f"raw response: {e['raw']!r}")
        print(f"parsed concerns ({len(e['concerns'])}):")
        for c in e["concerns"]:
            print(f"  - {c}")
        print(f"parsed supports ({len(e['supports'])}):")
        for s in e["supports"]:
            print(f"  - {s}")
    print()

    print("=== STEP 3: judge synthesis ===")
    print("--- raw judge response ---")
    print(result["judge_raw"])
    print("--- parsed reasoning ---")
    print(result["judge_reasoning"])
    print("--- parsed advice response ---")
    print(result["judge_output"])
    print()

    print(
        f"=== tokens for this post: in={result['input_tokens']} "
        f"out={result['output_tokens']} ==="
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_HF_MODEL_ID,
        help=f"HuggingFace model id to load (default: {DEFAULT_HF_MODEL_ID})",
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="number of OEQ prompts to run (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"random seed for OEQ uniform sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max new tokens per LLM call (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--cache-dir", default=os.environ.get("HF_HOME"),
        help="HuggingFace cache dir; defaults to $HF_HOME if set, else HF's default.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="torch device for the model (default: cuda)",
    )
    parser.add_argument(
        "--do-sample", action="store_true",
        help="enable sampled decoding (default: greedy/deterministic)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="decoding temperature, only used when --do-sample is set",
    )
    parser.add_argument(
        "--no-weak", dest="use_weak", action="store_false", default=True,
        help="route the identify/filter/concerns steps through the v2 "
             "prompts unchanged (default: route through weak_pipeline.py)",
    )
    args = parser.parse_args()

    # Resolve OEQ_DATA_CSV against the repo root so this works regardless
    # of cwd. `explore_mitigations` declares the path as relative.
    oeq_csv = OEQ_DATA_CSV
    if not os.path.isabs(oeq_csv):
        oeq_csv = os.path.join(_REPO_ROOT, oeq_csv)
    assert os.path.exists(oeq_csv), f"missing OEQ csv: {oeq_csv}"

    print(f"[load] {args.model_id} (cache_dir={args.cache_dir}, device={args.device})")
    model, tokenizer = load_llama(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    call_llm_local = make_call_llm(
        model, tokenizer,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )

    # Replace the harness's provider dispatcher with our local closure.
    # Both v1 and v2 import `call_llm` at module-load time via
    # `from explore_mitigations import call_llm`, so the binding must be
    # rewritten on each module that holds a reference. weak_pipeline
    # follows the same convention so it also needs patching when --weak
    # is in effect.
    our_mitigation.call_llm = call_llm_local
    our_mitigation_v2.call_llm = call_llm_local
    weak_pipeline.call_llm = call_llm_local

    # When --weak (the default), swap the names that
    # `our_mitigation_v2.run_mitigation` resolves at call time with the
    # _weak versions. `run_mitigation` calls these by their module-level
    # names — `identify_stakeholders_v2(...)`, etc. — so rewriting those
    # bindings on the v2 module is sufficient. The function objects
    # themselves stay untouched; only the names in the v2 namespace
    # change.
    if args.use_weak:
        our_mitigation_v2.identify_stakeholders_v2 = weak_pipeline.identify_stakeholders_weak
        our_mitigation_v2.filter_stakeholders_v2 = weak_pipeline.filter_stakeholders_weak
        our_mitigation_v2.stakeholder_concerns_v2 = weak_pipeline.stakeholder_concerns_weak
        print("[pipeline] weak (training_mitigations/weak_pipeline.py)")
    else:
        print("[pipeline] v2 (mitigations/our_mitigation_v2.py)")

    # Tolerant stakeholder-parser patch. The harness caps stakeholder
    # lists at MAX_STAKEHOLDERS=10 for API cost control, and
    # `parse_stakeholders` asserts `len(items) < cap`. Open 8B models
    # (observed: Qwen3-8B on OEQ post 2) sometimes emit exactly the
    # boundary, or emit themes rather than people; we don't want either
    # failure mode to abort the whole probe run. Replace the parser with
    # a version that just truncates at a higher local cap. v2 imports
    # the name via `from our_mitigation import parse_stakeholders`, so
    # patch both bindings.
    _LOCAL_MAX_STAKEHOLDERS = 15

    def _tolerant_parse_stakeholders(text, min_count=None, max_count=None):
        import re
        items = []
        for raw in (text or "").splitlines():
            s = raw.strip()
            s = re.sub(r"^(?:[\-\*•]|\d+[\.\)])\s*", "", s).strip()
            s = s.strip("*_`").strip()
            if s:
                items.append(s)
        if len(items) > _LOCAL_MAX_STAKEHOLDERS:
            items = items[:_LOCAL_MAX_STAKEHOLDERS]
        return items

    our_mitigation.parse_stakeholders = _tolerant_parse_stakeholders
    our_mitigation_v2.parse_stakeholders = _tolerant_parse_stakeholders
    # weak_pipeline does its own `from our_mitigation import
    # parse_stakeholders` at load time, so it has an independent
    # binding — patch it too.
    weak_pipeline.parse_stakeholders = _tolerant_parse_stakeholders

    print(f"[data] loading OEQ from {oeq_csv}")
    data = load_oeq(oeq_csv)
    sample = sample_uniform(data, args.n, args.seed)
    print(f"[data] sampled {len(sample)} prompts (seed={args.seed})")
    print()

    in_grand_total = 0
    out_grand_total = 0

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"############### prompt {i}/{len(sample)} ###############")
        # `run_mitigation` is the dict-returning version (vs.
        # `run_as_condition` which returns a flat tuple). Use it so the
        # printer has access to every intermediate field.
        result = our_mitigation_v2.run_mitigation(
            client=None,                # ignored by our local call_llm
            model=args.model_id,        # ignored by our local call_llm
            post_text=row["post_text"],
            max_tokens=args.max_tokens,
            env_name="OEQ",
        )
        _print_one_post_trace("OEQ", row["id"], row["post_text"], result)
        print()
        in_grand_total += result["input_tokens"]
        out_grand_total += result["output_tokens"]

    print(
        f"=== grand totals over {len(sample)} prompts: "
        f"in={in_grand_total} out={out_grand_total} ==="
    )


if __name__ == "__main__":
    main()
