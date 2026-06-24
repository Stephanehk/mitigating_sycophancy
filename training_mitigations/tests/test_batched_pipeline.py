"""Tests for the per-stage (breadth-first) batched build pipeline.

Validates the correctness-critical guarantees of the batching refactor, which
must produce data identical to the former per-prompt loop for a deterministic
judge:

  1. `concurrent_map` preserves input order and is fail-fast.
  2. `SimulatedStakeholders.veto_responses_batch` returns, per prompt, exactly
     what the per-prompt `veto_responses` returns.
  3. `Trainer._build_iteration` reduces the staged results into the same pairs,
     `responses_iter_*.jsonl` records, and diagnostics the per-prompt loop
     produced — in global prompt order, across multiple chunks.

Runs without a real model or vLLM server: the LLM primitives are stubbed with
deterministic Python so the batched output can be checked against a
hand-computed expectation. Run: python test_batched_pipeline.py
"""

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, _HERE)

from concurrency import concurrent_map
from trainer import Trainer


def test_concurrent_map_order_and_failfast():
    items = list(range(50))
    assert concurrent_map(lambda x: x * x, items, 8) == [x * x for x in items], \
        "concurrent_map must preserve input order under concurrency"
    # max_workers<=1 and single-item run inline; empty is empty.
    assert concurrent_map(lambda x: x + 1, [1, 2, 3], 1) == [2, 3, 4]
    assert concurrent_map(lambda x: x, [], 8) == []

    def boom(x):
        assert x != 7, "boom"
        return x
    raised = False
    try:
        concurrent_map(boom, list(range(20)), 4)
    except AssertionError:
        raised = True
    assert raised, "concurrent_map must propagate worker exceptions (fail-fast)"
    print("ok: concurrent_map order + fail-fast")


def test_veto_responses_batch_matches_sequential():
    from simulated_stakeholders import SimulatedStakeholders
    sim = SimulatedStakeholders(
        call_llm=lambda *a, **k: ("", 0, 0),
        model_id="stub",
        use_weak_model=True,
        veto_concurrency=4,
    )
    # Deterministic judge: stakeholder name vetoes a response iff the name is a
    # substring of the response text. _stakeholder_vetos returns (decision, raw).
    sim._stakeholder_vetos = lambda prompt, response, entry: (
        entry["stakeholder"] in response, "")

    cache_entries = {
        "p-alpha": ["S1", "S2"],
        "p-beta": ["S3"],
        "p-gamma": ["S1", "S3", "S9"],
    }
    # Pre-populate the cache so neither path runs the real identify pipeline.
    for p, names in cache_entries.items():
        sim._cache[p] = (names, [{"stakeholder": n, "concerns": []} for n in names])

    responses_map = {
        "p-alpha": ["has S1 only", "clean", "has S2", "S1 dup"],
        "p-beta": ["S3 here", "S3 again", "none"],
        "p-gamma": ["S9", "S1 and S3", "clean"],
    }
    prompts = list(cache_entries)
    items = [(p, responses_map[p]) for p in prompts]

    batched = sim.veto_responses_batch(items, concurrency=8)
    sequential = [sim.veto_responses(p, responses_map[p]) for p in prompts]
    assert batched == sequential, \
        f"veto_responses_batch != sequential\n batched={batched}\n seq={sequential}"
    print("ok: veto_responses_batch == per-prompt veto_responses")


class _FakeSim:
    """Minimal stand-in for SimulatedStakeholders in the reduce test: priming
    is a no-op and veto results are looked up from a fixed map."""

    def __init__(self, veto_map):
        self.veto_map = veto_map

    def prime_stakeholder_cache(self, prompts, concurrency):
        pass

    def veto_responses_batch(self, items, concurrency):
        return [self.veto_map[p] for p, _ in items]


def test_build_iteration_reduce():
    # This test is CPU-only; neutralize the per-chunk GPU mem-probe reset so it
    # runs anywhere (it is a no-op diagnostic, not part of the reduce logic).
    import torch
    torch.cuda.device_count = lambda: 0

    prompts = ["P0", "P1", "P2"]
    responses = {
        "P0": ["a0", "b0", "c0"],
        "P1": ["a1", "b1", "c1"],
        "P2": ["a2", "b2", "c2"],
    }
    veto_map = {
        "P0": ["b0"],                  # 1 vetoed -> 2 veto pairs + pref
        "P1": [],                      # 0 vetoed -> zero_vetoed; pref only
        "P2": ["a2", "b2", "c2"],      # all vetoed -> all_vetoed, 0 pairs
    }
    pref_map = {
        "P0": [{"prompt": "P0", "chosen": "a0", "rejected": "c0"}],
        "P1": [{"prompt": "P1", "chosen": "a1", "rejected": "b1"},
               {"prompt": "P1", "chosen": "a1", "rejected": "c1"}],
        "P2": [],
    }

    tr = Trainer(
        sim_stake=_FakeSim(veto_map), model=None, tokenizer=None,
        n_samples=3, temperature=1.0, self_preference=True,
        chunk_size=2,            # force 2 chunks over 3 prompts
        stage_concurrency=4,
    )
    tr._sample_chunk = lambda chunk: [
        (responses[x], ["model", "model", "critical"]) for x in chunk]
    tr._preference_chunk = lambda chunk, non_vetoed_list: [pref_map[x] for x in chunk]

    outdir = tempfile.mkdtemp()
    pairs, all_vetoed_idx, zero_vetoed_idx, contributing = \
        tr._build_iteration(prompts, 0, outdir)

    # Pairs in global prompt order: P0 veto (b0 × [a0,c0]) then P0 pref;
    # P1 pref (2); P2 none.
    expected_pairs = [
        {"prompt": "P0", "chosen": "a0", "rejected": "b0"},
        {"prompt": "P0", "chosen": "c0", "rejected": "b0"},
        {"prompt": "P0", "chosen": "a0", "rejected": "c0"},
        {"prompt": "P1", "chosen": "a1", "rejected": "b1"},
        {"prompt": "P1", "chosen": "a1", "rejected": "c1"},
    ]
    assert pairs == expected_pairs, f"pairs mismatch:\n{pairs}"
    assert zero_vetoed_idx == [1], zero_vetoed_idx
    assert all_vetoed_idx == [2], all_vetoed_idx
    assert contributing == 2, contributing

    lines = open(os.path.join(outdir, "responses_iter_1.jsonl")).read().strip().split("\n")
    assert len(lines) == 3, "one jsonl record per prompt"
    recs = [json.loads(line) for line in lines]
    assert [r["prompt_idx"] for r in recs] == [0, 1, 2], "global prompt_idx in order"
    assert recs[0]["vetoed"] == ["b0"] and recs[0]["num_pref_pairs"] == 1
    assert recs[1]["vetoed"] == [] and recs[1]["num_pref_pairs"] == 2
    assert recs[2]["vetoed"] == ["a2", "b2", "c2"] and recs[2]["num_pref_pairs"] == 0
    assert recs[0]["sources"] == ["model", "model", "critical"]
    # `pref_pairs` is persisted alongside `num_pref_pairs` so a preempt-resume of
    # a --self-preference run can rebuild the dataset without re-querying the
    # policy; its length must match the count.
    assert all(len(r["pref_pairs"]) == r["num_pref_pairs"] for r in recs)
    assert all(set(r) == {"iter", "prompt_idx", "prompt", "responses",
                          "sources", "vetoed", "num_pref_pairs", "pref_pairs"}
               for r in recs), "jsonl schema must be unchanged"
    print("ok: _build_iteration reduce (pairs order, diagnostics, jsonl schema)")


class _RaisingSim:
    """Stand-in for NeutralityVetoJudge in the steer-only (neutral-mixed) path:
    it has NO stakeholder world-model, so `_stakeholder_data_for` must never be
    called during candidate sampling. Calling it fails the test loudly."""

    def prime_stakeholder_cache(self, prompts, concurrency):
        pass

    def _stakeholder_data_for(self, prompt):
        raise AssertionError(
            "steer-only neutral-mixed sampling must not read stakeholder data")

    def veto_responses_batch(self, items, concurrency):
        return [[] for _ in items]


def test_neutral_mixed_sampling_steer_only():
    # CPU-only: neutralize the per-chunk GPU mem-probe reset (no-op diagnostic).
    import torch
    torch.cuda.device_count = lambda: 0
    from trainer import NEUTRAL_SUFFIX

    calls = []
    tr = Trainer(
        sim_stake=_RaisingSim(), model=None, tokenizer=None,
        n_samples=4, temperature=1.0,
        mixed_sampling=True, n_source_model=2, n_source_critical=2,
        steer_suffix=NEUTRAL_SUFFIX, steer_label="neutral", steer_only=True,
        chunk_size=8, stage_concurrency=4,
    )

    # Stub the generation primitive: record (prompt, n) and echo n responses.
    def fake_generate_n(prompt, n):
        calls.append((prompt, n))
        return [f"r::{prompt}::{i}" for i in range(n)]
    tr._generate_n = fake_generate_n

    post = "POST"
    out = tr._sample_chunk([post])
    assert len(out) == 1, out
    responses, sources = out[0]

    # 2 from the bare post + 2 from post+NEUTRAL_SUFFIX, total N=4, in order.
    assert len(responses) == 4 and len(sources) == 4, (responses, sources)
    assert sources == ["model", "model", "neutral", "neutral"], sources

    # The steer source appends NEUTRAL_SUFFIX at generation time; the model
    # source uses the bare post; nothing else (e.g. CRITICAL_SUFFIX) leaks in.
    gen_prompts = {p for p, _ in calls}
    assert gen_prompts == {post, post + NEUTRAL_SUFFIX}, gen_prompts
    assert dict(calls)[post] == 2 and dict(calls)[post + NEUTRAL_SUFFIX] == 2, calls
    print("ok: neutral-mixed steer-only (split, labels, suffix, no stakeholder read)")


def test_steer_only_requires_full_split():
    # steer_only has no stakeholder source, so the two source counts MUST cover
    # all N — a leftover (stakeholder) budget is a misconfiguration.
    import torch
    torch.cuda.device_count = lambda: 0
    try:
        Trainer(
            sim_stake=_RaisingSim(), model=None, tokenizer=None,
            n_samples=10, temperature=1.0,
            mixed_sampling=True, n_source_model=2, n_source_critical=2,
            steer_only=True,
        )
    except AssertionError as e:
        assert "must equal n_samples" in str(e), e
        print("ok: steer_only rejects a non-full source split")
    else:
        raise AssertionError("expected AssertionError for steer_only with leftover budget")


if __name__ == "__main__":
    test_concurrent_map_order_and_failfast()
    test_veto_responses_batch_matches_sequential()
    test_build_iteration_reduce()
    test_neutral_mixed_sampling_steer_only()
    test_steer_only_requires_full_split()
    print("\nALL BATCHED-PIPELINE TESTS PASSED")
