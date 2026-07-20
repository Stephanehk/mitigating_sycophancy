"""End-to-end wiring test for --sim-backend gemini in train_dpo_lora.main().

Companion to test_openrouter_wiring.py. It answers, executably, "which calls go
to the Gemini API, which stay on the local Qwen3-8B, and which PROMPTS does the
sim use?" -- the two questions this backend swap exists to control.

Everything expensive is stubbed (no GPU, no vLLM server, no network, no model
download): `load_llama`, the vLLM helpers, the Gemini client, and `Trainer` are
replaced, and `Trainer.train` raises a sentinel so main() stops right after
wiring. AITA is used because data/aita_clean.csv is checked in.

Asserted:
  1. --sim-backend gemini loads NO local sim model (only the policy), and the
     sim's call_llm is the Gemini closure.
  2. SimulatedStakeholders gets the SIM's model id (--sim-model), not the
     policy's --model-id, and max_concurrency=--sim-concurrency.
  3. --prompt-variant gemini reaches SimulatedStakeholders, so identify/filter/
     concerns resolve to gemini_pipeline; without it they stay on weak_pipeline.
  4. gemini_pipeline gets BOTH monkey-patches main() applies to the other
     pipeline modules: call_llm AND the tolerant parse_stakeholders. (Missing
     the latter would abort a run on any >=10-item emission.)
  5. The guard asserts fire: --prompt-variant gemini without --sim-backend
     gemini, and an OpenRouter-style "google/..." id passed to --sim-backend
     gemini.
  6. The retry/backoff and blocked-action knobs reach the closure factory.

Run with the project interpreter:
  /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/tests/test_gemini_wiring.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "core"))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

import train_dpo_lora as T
import gemini_pipeline
import weak_pipeline
import our_mitigation

_SCRATCH = os.environ.get("TMPDIR", "/tmp") + "/gemini_wiring_test"


class _Sentinel(Exception):
    """Raised by the stub Trainer.train to stop main() right after wiring."""


def _run(argv, captured):
    """Run main() with everything expensive stubbed; capture the wiring."""
    made = {}

    def fake_make_gemini(**kw):
        made.update(kw)

        def call_llm(client, model, prompt, max_tokens, system=None):
            return "STUB", 1, 1
        call_llm._is_gemini_stub = True
        captured["factory_kwargs"] = kw
        return call_llm

    def fake_load_llama(*a, **kw):
        captured.setdefault("load_llama_calls", []).append(kw.get("model_id", a[0] if a else None))
        class _Tok:
            pad_token_id = 0
            eos_token_id = 1
        return object(), _Tok()

    class FakeTrainer:
        def __init__(self, **kw):
            captured["trainer_kwargs"] = kw
        def train(self, *a, **kw):
            raise _Sentinel()

    class FakeSim:
        def __init__(self, **kw):
            captured["sim_kwargs"] = kw
            self.call_llm = kw["call_llm"]
            self.prompt_variant = kw.get("prompt_variant")
            self.use_weak_model = kw.get("use_weak_model", True)
            self._cache = {}
        def _pipeline_fns(self):
            from simulated_stakeholders import SimulatedStakeholders as Real
            return Real._pipeline_fns(self)

    T.make_gemini_call_llm = fake_make_gemini
    T.load_llama = fake_load_llama
    T.Trainer = FakeTrainer
    T.SimulatedStakeholders = FakeSim
    T.start_mem_sampler = lambda *a, **kw: None

    sys.argv = ["train_dpo_lora.py"] + argv
    try:
        T.main()
    except _Sentinel:
        return "ok"
    return "no-sentinel"


BASE = ["--env", "AITA", "--output-dir", _SCRATCH, "--train-n", "4", "--test-n", "2",
        "--num-iterations", "1", "--n-samples", "2", "--sim-backend", "gemini",
        "--sim-model", "gemini-3.1-flash-lite", "--sim-concurrency", "32",
        "--veto-prompt", "P6", "--device", "cpu"]


def test_gemini_wiring():
    cap = {}
    r = _run(BASE + ["--prompt-variant", "gemini", "--sim-max-retries", "5",
                     "--sim-backoff-max", "300"], cap)
    assert r == "ok", f"main() did not reach Trainer.train ({r})"

    # 1. No local SIM model: the only load_llama call is the policy.
    loads = cap.get("load_llama_calls", [])
    assert len(loads) == 1, f"expected ONLY the policy load, got {loads}"

    # 1b. The sim's call_llm is the Gemini closure.
    sim_kw = cap["sim_kwargs"]
    assert getattr(sim_kw["call_llm"], "_is_gemini_stub", False), \
        "sim call_llm is not the Gemini closure"

    # 2. Sim model id + concurrency.
    assert sim_kw["model_id"] == "gemini-3.1-flash-lite", sim_kw["model_id"]
    assert sim_kw["max_concurrency"] == 32, sim_kw["max_concurrency"]

    # 3. prompt_variant reaches the sim and resolves to gemini_pipeline.
    assert sim_kw["prompt_variant"] == "gemini", sim_kw
    sim = T.SimulatedStakeholders(**sim_kw)
    _i, _f, _c, mod = sim._pipeline_fns()
    assert mod is gemini_pipeline, f"pipeline module is {mod.__name__}"

    # 6. retry/backoff/blocked knobs reach the factory.
    fk = cap["factory_kwargs"]
    assert fk["max_retries"] == 5, fk
    assert fk["backoff_max"] == 300, fk
    # Refusals are ALWAYS raised as SimRefusedError and skipped per-stage by
    # simulated_stakeholders; there is no "return empty" mode to misconfigure.
    assert "blocked_action" not in fk, f"blocked_action should be gone: {fk}"
    assert fk["on_blocked"] is not None, "refusals must be counted"
    assert fk["thinking_budget"] == 0, fk
    assert fk["cost_tracker"] is not None, "sim spend is not metered"
    print("test_gemini_wiring: OK")


def test_default_variant_stays_weak():
    cap = {}
    assert _run(BASE, cap) == "ok"
    sim_kw = cap["sim_kwargs"]
    assert sim_kw["prompt_variant"] is None, sim_kw
    sim = T.SimulatedStakeholders(**sim_kw)
    _i, _f, _c, mod = sim._pipeline_fns()
    assert mod is weak_pipeline, f"default must stay weak_pipeline, got {mod.__name__}"
    print("test_default_variant_stays_weak: OK")


def test_tolerant_parser_patched_onto_gemini_pipeline():
    """gemini_pipeline must get the SAME tolerant parser as its siblings, or a
    >=10-item emission aborts the run (weak_pipeline's exact failure mode)."""
    cap = {}
    assert _run(BASE + ["--prompt-variant", "gemini"], cap) == "ok"
    assert gemini_pipeline.parse_stakeholders is T._tolerant_parse_stakeholders, \
        "gemini_pipeline.parse_stakeholders was NOT patched to the tolerant parser"
    assert weak_pipeline.parse_stakeholders is T._tolerant_parse_stakeholders
    # and the sim's call_llm binding was rewritten on gemini_pipeline too
    assert getattr(gemini_pipeline.call_llm, "_is_gemini_stub", False), \
        "gemini_pipeline.call_llm was NOT rebound to the sim backend"
    print("test_tolerant_parser_patched_onto_gemini_pipeline: OK")


def test_guards():
    common = ["--env", "AITA", "--output-dir", _SCRATCH, "--train-n", "4",
              "--test-n", "2", "--device", "cpu"]
    for argv, needle in (
        (common + ["--sim-backend", "vllm", "--prompt-variant", "gemini"],
         "tuned for the Gemini sim"),
        (common + ["--sim-backend", "gemini", "--prompt-variant", "gemini",
                   "--sim-model", "google/gemini-3.1-flash-lite"],
         "OpenRouter id"),
    ):
        try:
            _run(argv, {})
        except AssertionError as e:
            assert needle in str(e), f"wrong assert: {e}"
            print(f"test_guards: OK ({needle})")
            continue
        raise SystemExit(f"guard did NOT fire for {argv}")


if __name__ == "__main__":
    test_gemini_wiring()
    test_default_variant_stays_weak()
    test_tolerant_parser_patched_onto_gemini_pipeline()
    test_guards()
    print("\nall gemini wiring tests passed")
