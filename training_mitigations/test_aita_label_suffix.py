"""Smoke test for the AITA/FLIP YTA/NTA label-nudge plumbing in core/trainer.py.

Guarantees under test (the design the fix implements):
  1. For AITA / AITA-NTA-FLIP, EVERY candidate-generation prompt — all N=10
     samples across all sources (model / perspshift / stakeholder incl. the
     frozen-base ones) — ends with `AITA_LABEL_SUFFIX` (terminal instruction).
  2. The perspective-shift REWRITE (an auxiliary stage) runs on the BARE post:
     the nudge never leaks into the rewriter.
  3. The DPO pair `prompt` column carries the nudge (matching the eval-time
     input surface), while the persisted responses record stays BARE.
  4. For OEQ the nudge is absent everywhere — the regression guard for the
     "" no-op invariant that keeps existing OEQ/PAS runs byte-identical.

Runs on CPU with no model: `_generate_n` and `_policy_generate_greedy` are
monkeypatched to record the prompts they receive. Run with the project python:
    /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/test_aita_label_suffix.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

from trainer import Trainer, STAKEHOLDER_RESPONSE_PROMPT_VOICE  # noqa: E402
from train_data import AITA_LABEL_SUFFIX, label_suffix_for  # noqa: E402

BARE = "My roommate keeps eating my food. I changed the locks on the fridge. AITA?"
ENTRIES = [
    {"stakeholder": "the roommate", "concerns": ["needs to eat", "short notice"]},
    {"stakeholder": "the landlord", "concerns": ["fridge is shared property"]},
    {"stakeholder": "mutual friends", "concerns": ["being asked to take sides"]},
]


def _make_trainer(env):
    """A Trainer wired for the perspshift+stakeholder mixed recipe, with the
    two generation entry points monkeypatched to record their prompts."""
    t = Trainer(
        sim_stake=None, model=None, tokenizer=None,
        n_samples=10, mixed_sampling=True,
        n_source_model=2, n_source_critical=2,
        perspective_shift_steer=True, perspshift_assess=True,
        stakeholder_prompt=STAKEHOLDER_RESPONSE_PROMPT_VOICE,
        frozen_stakeholder=True,
        label_suffix=label_suffix_for(env),
    )
    gen_calls = []       # (final_gen_prompt, use_base) for every candidate draw
    rewrite_calls = []   # user prompt sent to the perspective-shift rewriter

    def fake_generate_n(gen_prompt, n, use_base=False):
        gen_calls.append((gen_prompt, use_base))
        return ["resp %d" % i for i in range(int(n))]

    def fake_greedy(user_prompt, max_new_tokens):
        rewrite_calls.append(user_prompt)
        return "Someone changed the locks on the shared fridge."

    t._generate_n = fake_generate_n
    t._policy_generate_greedy = fake_greedy
    return t, gen_calls, rewrite_calls


def _check_env(env):
    suffix = label_suffix_for(env)
    is_aita = env in ("AITA", "AITA-NTA-FLIP")
    assert (suffix == AITA_LABEL_SUFFIX) == is_aita, env

    t, gen_calls, rewrite_calls = _make_trainer(env)
    responses, sources = t._sample_responses_mixed(BARE, ENTRIES)

    # All N=10 candidates produced, split 2 model / 2 perspshift / 6 stakeholder.
    assert len(responses) == 10, (env, len(responses))
    assert sources.count("model") == 2, sources
    assert sources.count("perspshift") == 2, sources
    assert sum(s.startswith("stakeholder-base:") for s in sources) == 6, sources
    total_drawn = sum(1 for _ in responses)
    assert total_drawn == 10

    # (1)/(4): every candidate-generation prompt ends with the nudge for AITA;
    # for OEQ no candidate prompt contains any AITA label text.
    for gen_prompt, use_base in gen_calls:
        if is_aita:
            assert gen_prompt.endswith(AITA_LABEL_SUFFIX), (env, gen_prompt[-80:])
        else:
            assert AITA_LABEL_SUFFIX not in gen_prompt, (env, gen_prompt[-80:])
    # frozen-base sampling is used only for the stakeholder draws: 3 gen calls
    # (one per entry, n=2 each) carry use_base=True; the model + perspshift
    # calls carry use_base=False.
    assert sum(1 for _p, ub in gen_calls if ub) == 3, gen_calls
    assert sum(1 for _p, ub in gen_calls if not ub) == 2, gen_calls

    # (2): the rewrite ran exactly once, on the BARE post, with NO nudge.
    assert len(rewrite_calls) == 1, rewrite_calls
    assert BARE in rewrite_calls[0], env
    assert AITA_LABEL_SUFFIX not in rewrite_calls[0], (env, rewrite_calls[0][-80:])

    # (3): DPO pair prompt carries the nudge; the persisted record stays bare.
    record = {"prompt": BARE,
              "responses": ["It's fine — NTA.", "That's harsh — YTA."],
              "vetoed": ["That's harsh — YTA."]}
    pairs, vetoed, non_vetoed = t._record_contribution(record)
    assert len(pairs) == 1, pairs
    if is_aita:
        assert pairs[0]["prompt"] == BARE + AITA_LABEL_SUFFIX, env
    else:
        assert pairs[0]["prompt"] == BARE, env
    assert record["prompt"] == BARE, "the persisted record must stay bare"

    print("[ok] %-14s candidates=%d suffix=%r rewrite_bare=%s pair_prompt_ok"
          % (env, len(responses), suffix[:20], AITA_LABEL_SUFFIX not in rewrite_calls[0]))


def main():
    for env in ("OEQ", "AITA", "AITA-NTA-FLIP"):
        _check_env(env)
    print("ALL PASS")


if __name__ == "__main__":
    main()
