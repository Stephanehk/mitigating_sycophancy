"""Tests for the vLLM sim backend plumbing.

Covers the assumptions the speedup changes rely on, without needing a
GPU or a live server:
  1. `SimulatedStakeholders.veto_responses` returns the SAME veto set
     under the concurrent path (veto_concurrency > 1) as under the
     original sequential path, including per-occurrence semantics for
     duplicate response texts and OR-over-stakeholders.
  2. `vllm_sim._visible_device_for` maps a torch device index to the
     correct physical id both with and without an existing
     CUDA_VISIBLE_DEVICES mask (the SLURM case).

Run with the project interpreter:
  /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/test_sim_vllm.py
"""

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from simulated_stakeholders import SimulatedStakeholders
from vllm_sim import _visible_device_for


POST = "My roommate keeps borrowing my car without asking."

# Stakeholder entries as `_stakeholder_data_for` would cache them.
ENTRIES = [
    {"stakeholder": "the roommate", "concerns": ["I rely on the car for work"]},
    {"stakeholder": "the narrator's partner", "concerns": ["safety of the car"]},
]


def _make_stub_call_llm(call_log):
    """A deterministic veto judge: vetoes iff the candidate response
    embedded in the veto prompt contains the marker 'HARSH', and only
    when role-playing 'the roommate' (so OR-over-stakeholders is
    exercised — the partner never vetoes). Thread-safe; appends to
    `call_log` under a lock so the test can count calls.
    """
    lock = threading.Lock()

    def call_llm(client, model, prompt, max_tokens, system=None):
        with lock:
            call_log.append(prompt)
        # VETO_PROMPT opens with "...described below: {stakeholder}."
        if "described below: the roommate" in prompt and "HARSH" in prompt:
            return "VETO\nThis dismisses my need for the car.", 10, 10
        return "ACCEPT\nFine by me.", 10, 10

    return call_llm


def _make_sim(veto_concurrency, call_log):
    sim = SimulatedStakeholders(
        call_llm=_make_stub_call_llm(call_log),
        model_id="stub",
        client=None,
        veto_concurrency=veto_concurrency,
    )
    # Pre-seed the per-prompt cache so veto_responses never runs the
    # identify/filter/concerns pipeline (which would need a real model).
    sim._cache[POST] = (["the roommate", "the narrator's partner"], ENTRIES)
    return sim


def test_concurrent_matches_sequential():
    # Duplicate vetoed text on purpose: occurrence semantics must hold.
    responses = [
        "You should talk to them calmly.",
        "HARSH: just take their stuff in retaliation.",
        "Set a clear boundary together.",
        "HARSH: just take their stuff in retaliation.",
    ]
    seq_log, conc_log = [], []
    seq = _make_sim(1, seq_log).veto_responses(POST, responses)
    conc = _make_sim(8, conc_log).veto_responses(POST, responses)

    assert seq == [responses[1], responses[3]], f"sequential veto set wrong: {seq}"
    assert conc == seq, f"concurrent != sequential: {conc} vs {seq}"
    # Sequential short-circuits (vetoed responses stop at the roommate),
    # concurrent always issues every (response, stakeholder) call.
    assert len(conc_log) == len(responses) * len(ENTRIES), (
        f"concurrent path should issue all calls, got {len(conc_log)}"
    )
    assert len(seq_log) < len(conc_log), "sequential path should short-circuit"


def test_no_vetoes_and_all_vetoes():
    clean = ["Talk it out.", "Set a boundary."]
    harsh = ["HARSH a", "HARSH b"]
    assert _make_sim(8, []).veto_responses(POST, clean) == []
    assert _make_sim(8, []).veto_responses(POST, harsh) == harsh


def test_visible_device_for():
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    # No mask: torch index == physical id.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    assert _visible_device_for("cuda:1") == "1"
    assert _visible_device_for("cuda") == "0"
    # SLURM-style mask: cuda:1 is the second visible entry.
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    assert _visible_device_for("cuda:1") == "3"
    assert _visible_device_for("cuda:0") == "2"
    if saved is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved


if __name__ == "__main__":
    test_concurrent_matches_sequential()
    test_no_vetoes_and_all_vetoes()
    test_visible_device_for()
    print("all tests passed")
