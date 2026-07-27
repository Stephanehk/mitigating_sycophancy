"""In-the-wild qualitative probe: run "Our Model (3+1)" (OEQ arm) on the first
user utterances from the Study-3 human-chat transcripts.

WHY: every number we have for this arm comes from the frozen OEQ/PAS held-out
splits, i.e. Reddit-derived posts. Study 3 collected live chats between real
participants and a GPT assistant, so its FIRST user turn is an unprompted,
in-the-wild version of the same ask ("here is a thing I did — was I wrong?").
This script generates one response per utterance so we can read what the trained
policy actually says to those users. It is a QUALITATIVE dump: no judge, no
metric, no comparison arm.

Pipeline (single GPU, ~50 prompts):
  1. Parse `LUCIDUserFacingHistory` and take the first user utterance: the text
     after the FIRST `[user]:` tag and before the SECOND `[assistant]:` tag.
     (Every transcript opens with an `[assistant]:` greeting, so the assistant
     tag that closes the first user turn IS the second one overall; the script
     asserts this rather than assuming it.)
  2. Score every utterance with a deterministic, offline quality heuristic
     (no API calls, see `quality_score`) and keep the top --n.
  3. Sample ONE response per kept utterance from Qwen3-8B + the OEQ voice2 LoRA
     ("Our Model (3+1)"), through the same chat-template path
     (`build_chat_input`, system=None) that training and every held-out eval use,
     so the model sees the utterance exactly as it saw a bare post.
  4. Write a CSV of (rank, row index, utterance, response, quality features).

Modes:
  --extract-only  stop after step 2 and write just the selected prompts. Runs on
                  a login node (no torch/GPU needed) — use it to eyeball the
                  selection before spending a GPU allocation.
  (default)       full run; needs a GPU (see slurm_scripts/marlowe_in_the_wild.sbatch).

Assumptions (asserted in code):
  - The adapter dir holds a PEFT adapter (adapter_config.json at its root).
  - Every transcript has at least one `[user]:` turn.

NOT comparable to the held-out endorsement numbers: different prompt
distribution (live chat openers, not Reddit posts), different generation cap
(--max-new-tokens defaults to 1024 here, vs the frozen 512 the final OEQ/PAS
metrics are computed on — see the gen-cap decision), and nothing is judged.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # training_mitigations/
_REPO_ROOT = os.path.dirname(_HERE)

DEFAULT_CSV = os.path.join(_REPO_ROOT, "data", "study3_with_history.csv")
DEFAULT_ADAPTER = "/projects/m000254/runs/dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5"
HISTORY_COL = "LUCIDUserFacingHistory"

USER_TAG = "[user]:"
ASSISTANT_TAG = "[assistant]:"

# The transcripts were name-scrubbed with a regex that also ate sentence-initial
# capitalised words ("How do I..." -> "[REDACTED_NAME] do I..."), so a redacted
# utterance is not just anonymised, it is textually damaged. 52% of rows carry
# one; we have far more clean rows than we need, so they are excluded outright.
# Two variants appear in the corpus: [REDACTED_NAME] (20394) and [REDACTED] (20).
REDACTION = re.compile(r"\[REDACTED[A-Z_]*\]")


# --- extraction ------------------------------------------------------------
def first_user_utterance(history):
    """Text after the first `[user]:` tag and before the next `[assistant]:`.

    Returns (utterance, n_assistant_tags_before_close). The caller asserts the
    closing assistant tag is the SECOND one overall, which is the documented
    shape (assistant greeting -> user opener -> assistant reply). Returns
    (None, 0) when the row has no user turn at all.
    """
    if not isinstance(history, str):
        return None, 0
    i = history.find(USER_TAG)
    if i < 0:
        return None, 0
    start = i + len(USER_TAG)
    j = history.find(ASSISTANT_TAG, start)
    end = j if j >= 0 else len(history)
    # How many assistant tags precede the closing one (1 == the opening greeting,
    # so the closing tag is the second assistant tag overall).
    n_before = history.count(ASSISTANT_TAG, 0, start)
    return history[start:end].strip(), n_before


# --- quality heuristic (deterministic, offline) ----------------------------
# What "high quality" means here: an utterance that is a usable OEQ-style prompt
# — a self-contained, first-person account of something the user did that another
# person reacted to, written in clean enough English that the model's answer is
# about the situation rather than about decoding the text.

RELATIONS = (
    "friend", "wife", "husband", "girlfriend", "boyfriend", "partner", "sister",
    "brother", "mom", "mother", "dad", "father", "parent", "son", "daughter",
    "kid", "child", "cousin", "aunt", "uncle", "roommate", "coworker",
    "colleague", "boss", "neighbor", "neighbour", "family", "sibling", "in-law",
    "grandma", "grandmother", "grandpa", "grandfather", "fiance", "ex",
    "classmate", "teammate", "spouse",
)
CONFLICT = (
    "mad", "upset", "angry", "hurt", "annoyed", "offended", "wrong", "guilty",
    "argument", "argue", "fight", "fought", "apolog", "blame", "rude", "sorry",
    "uncomfortable", "excluded", "left out", "betray", "lied", "lie", "regret",
    "awkward", "tension", "conflict", "disagree", "resent", "jealous",
)
# Openers with no scenario in them ("I have a question about whether I was right
# or wrong") — the participant is addressing the chat UI, not telling a story.
META_OPENER = re.compile(
    r"^(hi|hello|hey|ok|okay|so)?\b[^.?!]{0,80}?"
    r"(i have a question|i need (some )?(help|advice)|can (you|i) help|"
    r"i want (to ask|advice)|question about)",
    re.IGNORECASE,
)

MIN_CHARS, MAX_CHARS = 120, 1200
MIN_WORDS = 25


def _features(text):
    words = text.split()
    n_words = len(words)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    low = text.lower()
    alpha = sum(c.isalpha() or c.isspace() or c in ".,'!?-" for c in text)
    caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    return {
        "n_chars": len(text),
        "n_words": n_words,
        "n_sentences": len(sentences),
        "n_relations": sum(1 for r in RELATIONS if r in low),
        "n_conflict": sum(1 for c in CONFLICT if c in low),
        "first_person": bool(re.search(r"\b(i|my|me|we|our)\b", low)),
        "ends_clean": text.rstrip().endswith((".", "!", "?")),
        "clean_ratio": alpha / max(len(text), 1),
        "caps_words": caps_words,
        "meta_opener": bool(META_OPENER.match(text.strip())),
        "redacted": bool(REDACTION.search(text)),
    }


def passes_filters(f):
    """Hard gate. An utterance must be a clean, self-contained first-person
    scenario; anything else is dropped before ranking (we have ~4x more clean
    rows than we need, so the gate can afford to be strict)."""
    return (
        not f["redacted"]
        and MIN_CHARS <= f["n_chars"] <= MAX_CHARS
        and f["n_words"] >= MIN_WORDS
        and f["first_person"]
        and f["n_relations"] >= 1        # someone else is involved
        and not f["meta_opener"]         # not "I have a question about..."
        and f["clean_ratio"] >= 0.97     # no markup / pasted junk
        and f["caps_words"] <= 2         # not shouted
    )


def quality_score(f):
    """Rank surviving utterances. Higher = better probe prompt.

    Weights are a judgement call, not a fitted model; they encode: prefer a
    fully-told story (length in a sweet spot, several sentences), with a named
    other party and an explicit interpersonal rupture, punctuated like prose.
    """
    n_words = f["n_words"]
    # Length sweet spot 45-140 words: long enough to carry a real situation,
    # short enough that the whole thing is the user's own account.
    if n_words < 45:
        length = (n_words - MIN_WORDS) / 20.0          # 0 -> 1 over [25, 45]
    elif n_words <= 140:
        length = 1.0
    else:
        length = max(0.0, 1.0 - (n_words - 140) / 120.0)
    return (
        3.0 * length
        + 1.0 * min(f["n_sentences"], 5) / 5.0
        + 1.5 * min(f["n_relations"], 3) / 3.0
        + 1.5 * min(f["n_conflict"], 4) / 4.0
        + 1.0 * float(f["ends_clean"])
    )


def _dedupe_key(text):
    """Collapse to a bag of the first 12 alphanumeric tokens — near-identical
    retellings (participants who restarted the chat) share a prefix."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(toks[:12])


def select_prompts(df, n):
    """Extract, filter, rank, dedupe. Returns a DataFrame of the top `n`."""
    assert HISTORY_COL in df.columns, f"missing column {HISTORY_COL!r}"
    rows = []
    n_no_user = 0
    for idx, hist in zip(df.index, df[HISTORY_COL]):
        utt, n_before = first_user_utterance(hist)
        if utt is None:
            n_no_user += 1
            continue
        assert n_before == 1, (
            f"row {idx}: first [user] turn is closed by assistant tag #{n_before + 1}, "
            f"not #2 — transcript shape differs from the documented "
            f"assistant-greeting/user/assistant layout"
        )
        f = _features(utt)
        rows.append({"row_index": idx, "utterance": utt, **f})
    assert rows, "no user utterances extracted"
    print(f"[extract] {len(rows)} utterances from {len(df)} rows "
          f"({n_no_user} rows had no [user] turn)")

    ex = pd.DataFrame(rows)
    kept = ex[ex.apply(lambda r: passes_filters(r), axis=1)].copy()
    print(f"[filter] {len(kept)}/{len(ex)} pass the hard gate "
          f"(dropped: {int(ex['redacted'].sum())} redacted, "
          f"{int((~ex['first_person']).sum())} no first person, "
          f"{int(ex['meta_opener'].sum())} meta openers)")
    assert len(kept) >= n, f"only {len(kept)} utterances pass the gate, need {n}"

    kept["quality_score"] = kept.apply(quality_score, axis=1)
    # Stable order: score desc, then row_index asc, so a rerun picks the same 50.
    kept = kept.sort_values(["quality_score", "row_index"],
                            ascending=[False, True], kind="mergesort")
    seen, picked = set(), []
    for _, r in kept.iterrows():
        k = _dedupe_key(r["utterance"])
        if k in seen:
            continue
        seen.add(k)
        picked.append(r)
        if len(picked) == n:
            break
    assert len(picked) == n, f"only {len(picked)} unique utterances after dedupe"
    out = pd.DataFrame(picked).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


# --- generation ------------------------------------------------------------
def _add_repo_paths():
    """Put training_mitigations/, core/ and eval/ on sys.path so the harness
    modules (llama_local, trainer, eval_endorsement_heldout) import. Called from
    every GPU/steer path; --extract-only never needs it, which is what keeps
    that mode runnable on a login node with no torch."""
    for p in (_HERE, os.path.join(_HERE, "core"), os.path.join(_HERE, "eval")):
        if p not in sys.path:
            sys.path.insert(0, p)


def generate(prompts, adapter_dir, model_id, device, cache_dir, max_new_tokens,
             temperature, top_p, seed, greedy, stream_path=None):
    """Sample one response per prompt from base+adapter. Imports torch and the
    eval harness lazily so --extract-only runs on a login node.

    Each response is appended to `stream_path` (jsonl, flushed per line) as soon
    as it is produced, mirroring eval_endorsement_heldout.py. The final CSV is
    written only after the whole loop, so without this a preemption or OOM at
    prompt 49/50 would lose the entire GPU run.
    """
    import torch
    _add_repo_paths()
    from llama_local import load_llama                       # noqa: E402
    from eval_endorsement_heldout import attach_adapter, generate_one  # noqa: E402

    print(f"[load] base {model_id} -> {device}")
    model, tokenizer = load_llama(model_id=model_id, device=device,
                                  cache_dir=cache_dir)
    if adapter_dir:
        print(f"[load] attaching adapter {adapter_dir}")
        model = attach_adapter(model, adapter_dir)
    else:
        print("[load] no adapter (base model)")

    torch.manual_seed(seed)
    responses = []
    sf = open(stream_path, "w") if stream_path else None
    try:
        for k, p in enumerate(prompts):
            resp = generate_one(model, tokenizer, p, max_new_tokens,
                                temperature, top_p, greedy)
            responses.append(resp)
            if sf:
                sf.write(json.dumps({"i": k, "prompt": p, "response": resp}) + "\n")
                sf.flush()
            print(f"[gen] {k + 1}/{len(prompts)}  ({len(resp)} chars)", flush=True)
    finally:
        if sf:
            sf.close()
    return responses


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help=f"Study-3 transcript CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--n", type=int, default=50,
                    help="how many utterances to keep (default: 50)")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER,
                    help="LoRA adapter dir — Our Model (3+1), OEQ arm "
                         f"(default: {DEFAULT_ADAPTER})")
    ap.add_argument("--base", action="store_true",
                    help="generate from the untrained base model instead "
                         "(comparison dump; ignores --adapter)")
    ap.add_argument("--be-critical", action="store_true",
                    help="prompt-only steer: append the trainer's "
                         "CRITICAL_SUFFIX ('\\n\\nBe critical.') to each "
                         "utterance at generation time, same mechanics as the "
                         "held-out be-critical baseline")
    ap.add_argument("--tag", default=None,
                    help="condition tag for the output filename (default: "
                         "derived from --base/--adapter/--be-critical)")
    ap.add_argument("--out-dir", default=None,
                    help="directory for the output CSV (default: the adapter's "
                         "eval/ dir, or the cwd under --base)")
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B",
                    help="base HF model id (default: Qwen/Qwen3-8B)")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="generation cap (default: 1024 — higher than the "
                         "frozen 512 used for the final endorsement metric, "
                         "because this dump is meant to be READ end-to-end)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <adapter>/eval/in_the_wild_"
                         "study3_<tag>.csv, or ./in_the_wild_study3_base.csv "
                         "under --base)")
    ap.add_argument("--extract-only", action="store_true",
                    help="write the selected prompts and stop (no GPU)")
    args = ap.parse_args()

    assert os.path.exists(args.csv), f"missing {args.csv}"
    df = pd.read_csv(args.csv)
    sel = select_prompts(df, args.n)
    print(f"[select] kept {len(sel)} utterances; "
          f"score {sel['quality_score'].min():.2f}-{sel['quality_score'].max():.2f}, "
          f"median {int(sel['n_words'].median())} words")

    if args.tag:
        tag = args.tag
    elif args.base:
        tag = "base-becritical" if args.be_critical else "base"
    else:
        tag = "ourmodel3plus1_oeq"
        if args.be_critical:
            tag += "-becritical"
    decode = "greedy" if args.greedy else f"temp{args.temperature}"
    fname = f"in_the_wild_study3_{tag}_{decode}_seed{args.seed}.csv"
    if args.out:
        out_path = args.out
    elif args.out_dir:
        out_path = os.path.join(args.out_dir, fname)
    elif args.base:
        out_path = os.path.join(os.getcwd(), fname)
    else:
        out_path = os.path.join(args.adapter, "eval", fname)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if args.extract_only:
        sel.to_csv(out_path, index=False)
        print(f"[out] {len(sel)} prompts -> {out_path}")
        return

    adapter_dir = None if args.base else args.adapter
    if adapter_dir:
        assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json")), (
            f"no adapter_config.json in {adapter_dir} — not a PEFT adapter dir")
    # `utterance` stays the bare participant text (what the pages/tables quote);
    # `gen_prompt` is what the model was actually shown, so a prompt-only steer
    # is visible in the output rather than silently folded into the prompt.
    gen_prompts = sel["utterance"].tolist()
    if args.be_critical:
        # Reuse the trainer's constant so the steer is byte-identical to the
        # held-out be-critical baseline. Imported lazily: core/trainer.py pulls
        # in torch/trl, which --extract-only must not require.
        _add_repo_paths()
        from trainer import CRITICAL_SUFFIX
        gen_prompts = [p + CRITICAL_SUFFIX for p in gen_prompts]
        print(f"[steer] be-critical: appended {CRITICAL_SUFFIX!r}")
    sel["gen_prompt"] = gen_prompts

    sel["response"] = generate(
        gen_prompts, adapter_dir, args.model_id, args.device,
        args.cache_dir, args.max_new_tokens, args.temperature, args.top_p,
        args.seed, args.greedy, stream_path=out_path.replace(".csv", ".jsonl"))
    # Human-readable condition name for tables/pages; falls back to the tag for
    # conditions this map doesn't know about.
    LABELS = {
        "ourmodel3plus1_oeq": "Our Model (3+1) / OEQ",
        "base": "Qwen3-8B (untrained)",
        "base-becritical": "Qwen3-8B + “Be critical.”",
        "neutralmixed": "Neutral-mixed DPO / OEQ",
        "ourmodel3plus1_oeq_32b": "Our Model (3+1) / OEQ — 32B",
        "base_32b": "Qwen3-32B (untrained)",
    }
    sel["model"] = LABELS.get(tag, tag)
    sel["condition"] = tag
    sel["adapter"] = adapter_dir or ""
    sel["be_critical"] = bool(args.be_critical)
    sel["max_new_tokens"] = args.max_new_tokens
    # Truncation flag: a response that does not end on terminal punctuation hit
    # the token cap mid-sentence (same signal used in the final-gen audit).
    sel["likely_truncated"] = ~sel["response"].str.rstrip().str.endswith(
        (".", "!", "?", '"', ")"))

    cols = ["rank", "row_index", "utterance", "gen_prompt", "response",
            "likely_truncated", "quality_score", "n_words", "n_sentences",
            "n_relations", "n_conflict", "model", "condition", "adapter",
            "be_critical", "max_new_tokens"]
    sel[cols].to_csv(out_path, index=False)
    print(f"[out] {len(sel)} responses -> {out_path}")
    print(f"[out] likely truncated: {int(sel['likely_truncated'].sum())}/{len(sel)}")
    with open(out_path.replace(".csv", ".json"), "w") as f:
        json.dump({"n": len(sel), "condition": tag, "label": LABELS.get(tag, tag),
                   "adapter": adapter_dir or "", "base": args.base,
                   "be_critical": bool(args.be_critical),
                   "model_id": args.model_id, "max_new_tokens": args.max_new_tokens,
                   "temperature": args.temperature, "top_p": args.top_p,
                   "greedy": args.greedy, "seed": args.seed,
                   "source_csv": args.csv}, f, indent=2)


if __name__ == "__main__":
    main()
