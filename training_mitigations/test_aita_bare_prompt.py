"""Guards for the AITA / AITA-NTA-FLIP no-label-suffix surface + the classifier
verdict path that replaced the model's self-emitted YTA/NTA label.

Guarantees under test:
  1. `format_prompt` / `build_train_prompts` return the BARE post for every env
     (OEQ / PAS / AITA / AITA-NTA-FLIP) — no output-format nudge is ever appended.
  2. In the trainer, EVERY candidate-generation prompt and the DPO pair `prompt`
     column are bare (no leaked label instruction), for AITA and OEQ alike.
  3. The classifier `parse_label` recovers YTA / NTA / UNCLEAR (and None for an
     unparsable judge output).
  4. `compute_fnr_fpr` / `compute_aita_nta_flip_metrics` drop UNCLEAR from the
     rate denominators and report it as `n_unclear`.

Runs on CPU with no model (generation entry points are monkeypatched). Run with:
    /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/test_aita_bare_prompt.py
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, os.path.join(_HERE, "eval"))

from trainer import Trainer, STAKEHOLDER_RESPONSE_PROMPT_VOICE  # noqa: E402
from train_data import format_prompt, build_train_prompts  # noqa: E402
from classify_aita_responses import parse_label  # noqa: E402
from core.metrics import compute_fnr_fpr, compute_aita_nta_flip_metrics  # noqa: E402

# The old forced-verdict nudge text — must never appear in any prompt now.
_NUDGE_MARKERS = ("output a label", "NTA or YTA", "YTA or NTA")

BARE = "My roommate keeps eating my food. I changed the locks on the fridge. AITA?"
ENTRIES = [
    {"stakeholder": "the roommate", "concerns": ["needs to eat", "short notice"]},
    {"stakeholder": "the landlord", "concerns": ["fridge is shared property"]},
    {"stakeholder": "mutual friends", "concerns": ["being asked to take sides"]},
]
ENVS = ("OEQ", "PAS", "AITA", "AITA-NTA-FLIP")


def _has_nudge(text):
    low = str(text).lower()
    return any(m.lower() in low for m in _NUDGE_MARKERS)


def test_format_prompt_bare():
    """(1) format_prompt + build_train_prompts return the bare post for all envs."""
    for env in ENVS:
        assert format_prompt(env, BARE) == BARE, env
    # AITA: one bare prompt per row.
    df_aita = pd.DataFrame({"post_text": [BARE, "second post"]})
    assert build_train_prompts("AITA", df_aita) == [BARE, "second post"]
    # FLIP: two bare prompts per pair, original then flipped, in order.
    df_flip = pd.DataFrame({"post_text": ["orig"], "flipped_post_text": ["flip"]})
    assert build_train_prompts("AITA-NTA-FLIP", df_flip) == ["orig", "flip"]
    print("[ok] format_prompt/build_train_prompts bare for all envs")


def _make_trainer():
    t = Trainer(
        sim_stake=None, model=None, tokenizer=None,
        n_samples=10, mixed_sampling=True,
        n_source_model=2, n_source_critical=2,
        perspective_shift_steer=True, perspshift_assess=True,
        stakeholder_prompt=STAKEHOLDER_RESPONSE_PROMPT_VOICE,
        frozen_stakeholder=True,
    )
    gen_calls, rewrite_calls = [], []

    def fake_generate_n(gen_prompt, n, use_base=False):
        gen_calls.append((gen_prompt, use_base))
        return ["resp %d" % i for i in range(int(n))]

    def fake_greedy(user_prompt, max_new_tokens):
        rewrite_calls.append(user_prompt)
        return "Someone changed the locks on the shared fridge."

    t._generate_n = fake_generate_n
    t._policy_generate_greedy = fake_greedy
    return t, gen_calls, rewrite_calls


def test_trainer_prompts_bare():
    """(2) No candidate-gen prompt and no DPO pair prompt carries a label nudge."""
    t, gen_calls, rewrite_calls = _make_trainer()
    responses, sources = t._sample_responses_mixed(BARE, ENTRIES)
    assert len(responses) == 10, len(responses)
    assert sources.count("model") == 2 and sources.count("perspshift") == 2, sources
    assert sum(s.startswith("stakeholder-base:") for s in sources) == 6, sources

    for gen_prompt, _use_base in gen_calls:
        assert not _has_nudge(gen_prompt), gen_prompt[-120:]
    assert len(rewrite_calls) == 1 and BARE in rewrite_calls[0]
    assert not _has_nudge(rewrite_calls[0])

    record = {"prompt": BARE,
              "responses": ["It's fine.", "That's harsh, reconsider."],
              "vetoed": ["That's harsh, reconsider."]}
    pairs, _vetoed, _non_vetoed = t._record_contribution(record)
    assert len(pairs) == 1 and pairs[0]["prompt"] == BARE, pairs
    assert record["prompt"] == BARE
    print("[ok] trainer candidate-gen + DPO pair prompts are bare (no nudge)")


def test_parse_label():
    """(3) classifier parse recovers YTA/NTA/UNCLEAR and None for junk."""
    assert parse_label("YTA") == "YTA"
    assert parse_label("the verdict is nta") == "NTA"
    assert parse_label("UNCLEAR") == "UNCLEAR"
    assert parse_label("NTA then UNCLEAR") == "UNCLEAR"   # last token wins
    assert parse_label("no verdict here") is None
    print("[ok] parse_label YTA/NTA/UNCLEAR/None")


def test_metrics_drop_unclear():
    """(4) UNCLEAR is dropped from the FNR/FPR + FLIP denominators, reported as
    n_unclear."""
    a = pd.DataFrame({
        "pred":       ["NTA", "NTA", "YTA", "UNCLEAR", "YTA", "NTA", "NTA", "OTHER"],
        "is_asshole": [1,     1,     1,     1,         0,     0,     0,     0],
    })
    m = compute_fnr_fpr(a)
    assert m["n_unclear"] == 1 and m["n_dropped_other"] == 2, m
    assert m["n_yta_truth"] == 3 and m["fn"] == 2, m       # UNCLEAR YTA-truth dropped
    f = pd.DataFrame({
        "pred_original": ["NTA", "YTA", "NTA", "UNCLEAR"],
        "pred_flipped":  ["NTA", "YTA", "YTA", "NTA"],
        "sycophancy_score": [1, 0, 0, 0],
        "critical_score":   [0, 1, 0, 0],
    })
    m2 = compute_aita_nta_flip_metrics(f)
    assert m2["n_unclear"] == 1 and m2["n_dropped_other"] == 1, m2
    assert m2["n_sycophancy"] == 1 and m2["n_critical"] == 1, m2
    print("[ok] metrics drop UNCLEAR from denominators, report n_unclear")


def main():
    test_format_prompt_bare()
    test_trainer_prompts_bare()
    test_parse_label()
    test_metrics_drop_unclear()
    print("ALL PASS")


if __name__ == "__main__":
    main()
