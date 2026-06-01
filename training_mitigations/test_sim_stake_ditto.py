"""Probe `SimulatedStakeholders` with a split backend: Qwen3-8B handles
identify + filter + per-(stakeholder, response) VETO/ACCEPT AND the
candidate-response sampling; Ditto-8B (sunweiwei/Ditto-8B,
Qwen3-VL-8B-Instruct base, tuned for first-person stakeholder
simulation) handles ONLY the per-stakeholder concerns
generation.

This is a capability inspection, not a metric run. For each of `--n`
OEQ prompts we:

  1. Run identify + filter through Qwen3-8B.
  2. Run per-stakeholder concerns through Ditto-8B (its
     job is to voice the stakeholder).
  3. Sample candidate responses from Qwen3-8B in four buckets:
       - 2 from the base post,
       - 2 from the post + "Be critical." suffix,
       - 2 from the post + "Don't be sycophantic." suffix,
       - 1 per kept stakeholder, where the prompt asks the model to
         roleplay that stakeholder ("Respond ... roleplaying the
         stakeholder: X. RESPONSE EXACTLY AS THE STAKEHOLDER WOULD.").
     With the filter cap of 3 stakeholders, that's 6 + 3 = 9
     responses per prompt at the typical case. The veto step judges
     every response against the original (unsuffixed) prompt
     regardless of which bucket produced it.
  4. For every (stakeholder, response) pair, ask Qwen3-8B to VETO or
     ACCEPT and print the raw reply (one line + a one-sentence
     justification per the VETO_PROMPT contract).

The two 8B bf16 models default to one GPU each (cuda:0, cuda:1); set
`--primary-device` and `--concerns-device` to the same id to share
one GPU instead (~32 GB total — fits on a single a6000-48G).

This script reuses the same tolerant-stakeholder-parser shim and
weak-pipeline rewiring as `test_v2_llama.py` so an 8B emitting
boundary-or-themed stakeholder lists doesn't abort the run.
"""

import argparse
import os
import re
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import our_mitigation
import our_mitigation_v2
import weak_pipeline

from explore_mitigations import (
    OEQ_DATA_CSV,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    load_oeq,
    sample_uniform,
)
from llama_local import (
    DEFAULT_HF_MODEL_ID,
    build_chat_input,
    load_llama,
    make_call_llm,
)
from simulated_stakeholders import SimulatedStakeholders


def _load_image_text_model(model_id, device, cache_dir):
    """Load a Qwen3-VL-style model (e.g. sunweiwei/Ditto-8B, which is
    Qwen3-VL-8B-Instruct based) via `AutoModelForImageTextToText` so the
    LM head is exposed, then return `(model, tokenizer)` with the same
    interface as `llama_local.load_llama`. Text-only `generate` still
    works on these models — the image branch is dead when no pixel
    inputs are supplied.

    Why not `AutoModelForCausalLM`: that auto class doesn't recognize
    `Qwen3VLConfig`, so passing a VL checkpoint id raises a ValueError
    during loading. `AutoModelForImageTextToText` is the registered
    auto class for VL configs in current transformers.

    Assumptions:
      - The checkpoint ships a chat-template-aware text tokenizer
        (Ditto-8B does — its `tokenizer.apply_chat_template(...)`
        produces the same text format the model card example uses).
      - The caller will not pass image inputs through `make_call_llm`;
        we never construct pixel tensors. If a future variant requires
        image embeddings to be present, this loader will need a real
        processor.
    """
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if device.startswith("cuda"):
        assert torch.cuda.is_available(), (
            f"requested device={device!r} but torch.cuda.is_available() is False"
        )
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        cache_dir=cache_dir,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


_LOCAL_MAX_STAKEHOLDERS = 15


def _tolerant_parse_stakeholders(text, min_count=None, max_count=None):
    """Same shim used in `test_v2_llama.py` / `train_dpo_lora.py`:
    strip bullet/number prefixes and whitespace, keep non-empty lines,
    truncate at a higher local cap to avoid the harness's strict
    `< MAX_STAKEHOLDERS` assert tripping on open 8B emissions.
    """
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


def _patch_pipeline(call_llm_local, use_weak):
    """Point all module-level `call_llm` bindings at the local closure
    and (when --weak) swap the v2 pipeline functions for their _weak
    counterparts. Identical to the patching block in
    `test_v2_llama.py`; factored out so this probe stays small.
    """
    our_mitigation.call_llm = call_llm_local
    our_mitigation_v2.call_llm = call_llm_local
    weak_pipeline.call_llm = call_llm_local
    our_mitigation.parse_stakeholders = _tolerant_parse_stakeholders
    our_mitigation_v2.parse_stakeholders = _tolerant_parse_stakeholders
    weak_pipeline.parse_stakeholders = _tolerant_parse_stakeholders
    if use_weak:
        our_mitigation_v2.identify_stakeholders_v2 = weak_pipeline.identify_stakeholders_weak
        our_mitigation_v2.filter_stakeholders_v2 = weak_pipeline.filter_stakeholders_weak
        our_mitigation_v2.stakeholder_concerns_v2 = weak_pipeline.stakeholder_concerns_weak


def _sample_responses(model, tokenizer, prompt, n, max_new_tokens,
                      temperature, top_p):
    """Draw `n` stochastic responses from `model` for `prompt` in one
    `model.generate(num_return_sequences=n)` call. Returns a list of
    decoded strings (special tokens skipped, leading whitespace
    stripped).
    """
    device = next(model.parameters()).device
    input_ids = build_chat_input(tokenizer, prompt, system=None, device=device)
    in_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(n),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for i in range(n):
        new_ids = output_ids[i, in_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        responses.append(text)
    return responses


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--primary-model-id", default=DEFAULT_HF_MODEL_ID,
        help=f"HF id of the main pipeline + sampler model (default: {DEFAULT_HF_MODEL_ID})",
    )
    parser.add_argument(
        "--concerns-model-id", default="sunweiwei/Ditto-8B",
        help="HF id of the per-stakeholder concerns model (default: sunweiwei/Ditto-8B)",
    )
    parser.add_argument(
        "--n", type=int, default=2,
        help="number of OEQ prompts to probe (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"OEQ uniform-sample seed (default: {DEFAULT_SEED})",
    )
    # HF `generate` requires a finite cap. The defaults below are
    # large enough that natural EOS fires first on Qwen3-8B / Ditto-8B
    # for OEQ-style prompts — bump if you see truncation.
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="max new tokens for pipeline LLM calls (default: 8192)",
    )
    parser.add_argument(
        "--sample-max-tokens", type=int, default=8192,
        help="max new tokens per sampled candidate response (default: 8192)",
    )
    parser.add_argument(
        "--veto-max-tokens", type=int, default=8192,
        help="max new tokens for the per-pair VETO/ACCEPT call (default: 8192)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="sampling temperature for the candidate responses (default: 1.0)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="nucleus sampling top-p for candidate responses (default: 0.95)",
    )
    parser.add_argument(
        "--cache-dir", default=os.environ.get("HF_HOME"),
        help="HF cache dir; defaults to $HF_HOME if set, else HF's default",
    )
    parser.add_argument(
        "--primary-device", default="cuda:0",
        help="torch device for the primary model (default: cuda:0)",
    )
    parser.add_argument(
        "--concerns-device", default="cuda:1",
        help="torch device for the concerns model (default: cuda:1). "
             "Set equal to --primary-device to share one GPU.",
    )
    parser.add_argument(
        "--no-weak", dest="use_weak", action="store_false", default=True,
        help="route identify/filter/concerns through the v2 prompts unchanged "
             "(default: route through weak_pipeline.py)",
    )
    parser.add_argument(
        "--config", choices=("qqqq", "qqdq", "qqdd"), default="qqqq",
        help="4-step routing: each letter is the backend for "
             "(identify, filter, concerns, veto). q=primary (Qwen), "
             "d=concerns model (Ditto). 'qqqq' (default) uses Qwen for "
             "everything (Ditto is loaded but unused). 'qqdq' uses Ditto "
             "only for concerns. 'qqdd' uses Ditto for concerns + veto.",
    )
    args = parser.parse_args()

    oeq_csv = OEQ_DATA_CSV
    if not os.path.isabs(oeq_csv):
        oeq_csv = os.path.join(_REPO_ROOT, oeq_csv)
    assert os.path.exists(oeq_csv), f"missing OEQ csv: {oeq_csv}"

    print(f"[load] primary  {args.primary_model_id} -> {args.primary_device}  (cache_dir={args.cache_dir})")
    primary_model, primary_tokenizer = load_llama(
        model_id=args.primary_model_id,
        device=args.primary_device,
        cache_dir=args.cache_dir,
    )
    primary_call_llm = make_call_llm(primary_model, primary_tokenizer, do_sample=False)

    # Ditto-8B is Qwen3-VL-8B-Instruct based, so its config is
    # `Qwen3VLConfig` and `AutoModelForCausalLM` refuses to load it.
    # `_load_image_text_model` routes through `AutoModelForImageTextToText`
    # which is registered for VL configs; text-only generate still works.
    print(f"[load] concerns {args.concerns_model_id} -> {args.concerns_device}  (cache_dir={args.cache_dir})  [VL loader]")
    concerns_model, concerns_tokenizer = _load_image_text_model(
        model_id=args.concerns_model_id,
        device=args.concerns_device,
        cache_dir=args.cache_dir,
    )
    # Sampled decoding so Ditto's first-person stakeholder voice has
    # some diversity (greedy collapses to flatter, template-style
    # output that doesn't reveal much).
    concerns_call_llm = make_call_llm(
        concerns_model, concerns_tokenizer, do_sample=True, temperature=0.7,
    )

    _patch_pipeline(primary_call_llm, args.use_weak)

    # Map the 4-letter config to which backend each of
    # (concerns, veto) uses. Identify+filter always run on the primary
    # backend in this script — both "q" letters in --config are
    # informational. Each entry below selects the call_llm/model_id
    # passed for that step; identify/filter ignore these.
    concerns_letter = args.config[2]
    veto_letter = args.config[3]
    if concerns_letter == "d":
        sim_concerns_call_llm = concerns_call_llm
        sim_concerns_model_id = args.concerns_model_id
    else:
        sim_concerns_call_llm = primary_call_llm
        sim_concerns_model_id = args.primary_model_id
    if veto_letter == "d":
        sim_veto_call_llm = concerns_call_llm
        sim_veto_model_id = args.concerns_model_id
    else:
        sim_veto_call_llm = primary_call_llm
        sim_veto_model_id = args.primary_model_id

    print(
        f"[pipeline] {'weak' if args.use_weak else 'v2'}  config={args.config}  "
        f"primary={args.primary_model_id}  concerns={sim_concerns_model_id}  "
        f"veto={sim_veto_model_id}"
    )

    sim = SimulatedStakeholders(
        call_llm=primary_call_llm,
        model_id=args.primary_model_id,
        client=None,
        stakeholder_max_tokens=args.max_tokens,
        veto_max_tokens=args.veto_max_tokens,
        use_weak_model=args.use_weak,
        concerns_call_llm=sim_concerns_call_llm,
        concerns_model_id=sim_concerns_model_id,
        concerns_client=None,
        veto_call_llm=sim_veto_call_llm,
        veto_model_id=sim_veto_model_id,
        veto_client=None,
    )

    print(f"[data] loading OEQ from {oeq_csv}")
    data = load_oeq(oeq_csv)
    sample = sample_uniform(data, args.n, args.seed)
    print(f"[data] sampled {len(sample)} prompts (seed={args.seed})")
    print()

    # Three sampling buckets: 2 responses each from the base prompt,
    # the "Be critical." suffix, and the "Don't be sycophantic." suffix.
    # Veto adjudication uses the original post_text regardless of which
    # suffix produced the response — we're judging the response, not
    # the prompt that produced it.
    sampling_buckets = [
        ("base", ""),
        ("be_critical", "\n\nBe critical."),
        ("no_sycophancy", "\n\nDon't be sycophantic."),
    ]
    samples_per_bucket = 2

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        post_id = row["id"]
        post_text = row["post_text"]
        print(f"############### prompt {i}/{len(sample)}  id={post_id} ###############")
        print(post_text)
        print()

        print("=== running stakeholder pipeline (identify+filter: primary, concerns: concerns) ===")
        candidates, entries = sim._stakeholder_data_for(post_text)
        kept_names = [e["stakeholder"] for e in entries]
        kept_set = set(kept_names)
        dropped = [c for c in candidates if c not in kept_set]
        print(f"--- identify ({len(candidates)} candidates) ---")
        for c in candidates:
            print(f"  - {c}")
        print(f"--- filter: kept ({len(kept_names)}) ---")
        for s in kept_names:
            print(f"  + {s}")
        print(f"--- filter: dropped ({len(dropped)}) ---")
        for s in dropped:
            print(f"  - {s}")
        print()
        for j, e in enumerate(entries, 1):
            print(f"--- stakeholder {j}/{len(entries)}: {e['stakeholder']} ---")
            print(f"concerns ({len(e['concerns'])}):")
            for c in e["concerns"]:
                print(f"  - {c}")
        print()

        # Sample from three prompt variants. Seed per (prompt, bucket)
        # so the response set is reproducible across reruns / configs.
        print(f"=== sampling {samples_per_bucket} candidates per bucket × {len(sampling_buckets)} buckets, plus 1 per kept stakeholder ===")
        responses = []
        response_buckets = []
        for b_idx, (b_name, b_suffix) in enumerate(sampling_buckets):
            torch.manual_seed(args.seed + i * 100 + b_idx)
            new_responses = _sample_responses(
                primary_model, primary_tokenizer, post_text + b_suffix,
                n=samples_per_bucket,
                max_new_tokens=args.sample_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            responses.extend(new_responses)
            response_buckets.extend([b_name] * samples_per_bucket)

        # One additional response per kept stakeholder, generated by
        # asking the model to roleplay that stakeholder. These responses
        # are then judged against the original (unsuffixed) post by the
        # veto step — same contract as the other buckets.
        for s_idx, entry in enumerate(entries):
            sh = entry["stakeholder"]
            rp_prompt = (
                f"Respond the to the following, roleplaying the stakeholder: {sh}. "
                f"RESPONSE EXACTLY AS THE STAKEHOLDER WOULD.\n\n{post_text}"
            )
            torch.manual_seed(args.seed + i * 100 + 100 + s_idx)
            new_responses = _sample_responses(
                primary_model, primary_tokenizer, rp_prompt,
                n=1,
                max_new_tokens=args.sample_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            responses.extend(new_responses)
            response_buckets.append(f"stakeholder:{sh}")
        for k, (r, b) in enumerate(zip(responses, response_buckets), 1):
            print(f"--- response {k}/{len(responses)} [bucket={b}] ---")
            print(r)
        print()

        print("=== per-(stakeholder, response) VETO/ACCEPT (veto backend) ===")
        for k, (r, b) in enumerate(zip(responses, response_buckets), 1):
            print(f"~~~ response {k}/{len(responses)} [bucket={b}] ~~~")
            for e in entries:
                decision, raw = sim._stakeholder_vetos(post_text, r, e)
                tag = "VETO  " if decision else "ACCEPT"
                print(f"  [{tag}] {e['stakeholder']}")
                # raw_text contains the verdict token + the one-sentence
                # justification; print verbatim so the reader can see the
                # judge's reasoning.
                for ln in (raw or "").splitlines():
                    print(f"    | {ln}")
            print()

    print("=== done ===")


if __name__ == "__main__":
    main()
