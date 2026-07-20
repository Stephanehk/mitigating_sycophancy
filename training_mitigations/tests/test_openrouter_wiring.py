"""End-to-end wiring test for --sim-backend openrouter in train_dpo_lora.main().

The unit tests in test_openrouter_sim.py cover the pieces (the closure, the
concurrency clamp, the rewrite routing). This test covers the thing those cannot:
that `main()` hands those pieces to the right objects. It answers, executably,
"which calls go to the API model and which stay on the local Qwen3-8B?" — the
question the backend swap exists to control.

Everything expensive is stubbed (no GPU, no vLLM server, no network, no model
download): `load_llama`, the vLLM helpers, and `Trainer` are replaced, and
`Trainer.train` raises a sentinel so main() stops right after wiring. AITA is
used because `data/aita_clean.csv` is checked in.

Asserted:
  1. --sim-backend openrouter loads NO local sim model (only the policy), and
     the sim's call_llm is the OpenRouter closure, not a vLLM/HF one.
  2. SimulatedStakeholders gets the SIM's model id (--sim-model), not the
     policy's --model-id, and gets max_concurrency=--sim-concurrency.
  3. --sim-backend openrouter COMPOSES with --policy-on-vllm: the engine is
     still launched (for the policy), and the policy sampler is still built.
  4. --rewrite-on-sim routes the perspective-shift rewrite to the sim; without
     it, rewrite_call_llm stays None (every existing arm unchanged).
  5. The local backends are untouched: --sim-backend vllm still builds a vLLM
     sim call_llm, passes the policy's model id, and sets no concurrency cap.

Run with the project interpreter:
  /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/tests/test_openrouter_wiring.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "core"))

import train_dpo_lora as T

_SCRATCH = os.environ.get("TMPDIR", "/tmp") + "/or_wiring_test"


class _Stop(Exception):
    """Raised by the stub Trainer.train to halt main() once wiring is done."""


class _FakeTrainer:
    last = None

    def __init__(self, **kwargs):
        _FakeTrainer.last = kwargs

    def train(self, *a, **k):
        raise _Stop()


def _run(extra_argv):
    """Drive main() with the heavy pieces stubbed; return (captured, loaded).

    `captured` is the Trainer's kwargs (incl. the wired sim object);
    `loaded` records every load_llama / launch_vllm_server call so the test can
    assert what was and was NOT stood up locally.
    """
    loaded = {"llama": [], "server": [], "policy_sampler": []}

    def fake_load_llama(model_id=None, device=None, cache_dir=None, **k):
        loaded["llama"].append((model_id, device))
        return object(), object()

    def fake_launch(model_id, device, **k):
        loaded["server"].append((model_id, device))
        return object(), "http://stub:8000"

    def fake_policy_sampler(base_url, model_id, **k):
        loaded["policy_sampler"].append(model_id)
        return lambda *a, **kw: ["stub"]

    saved = {n: getattr(T, n) for n in
             ("load_llama", "launch_vllm_server", "wait_for_vllm_server",
              "make_vllm_policy_sampler", "Trainer", "start_mem_sampler",
              "make_vllm_call_llm")}
    T.load_llama = fake_load_llama
    T.launch_vllm_server = fake_launch
    T.wait_for_vllm_server = lambda *a, **k: None
    T.make_vllm_policy_sampler = fake_policy_sampler
    T.make_vllm_call_llm = lambda base_url, model_id, **k: _tagged("vllm", model_id)
    T.Trainer = _FakeTrainer
    T.start_mem_sampler = lambda *a, **k: None
    argv = sys.argv
    sys.argv = ["train_dpo_lora.py", "--env", "AITA",
                "--output-dir", _SCRATCH,
                "--train-n", "4", "--test-n", "2"] + extra_argv
    try:
        T.main()
        raise AssertionError("main() should have reached the stub Trainer.train")
    except _Stop:
        pass
    finally:
        sys.argv = argv
        for n, v in saved.items():
            setattr(T, n, v)
    return _FakeTrainer.last, loaded


def _tagged(kind, model_id):
    """A call_llm closure carrying its backend identity, so the test can tell
    which backend main() wired in without making a call."""
    def call_llm(client, model, prompt, max_tokens, system=None):
        return f"[{kind}]", 1, 1
    call_llm.kind = kind
    call_llm.model_id = model_id
    return call_llm


def test_openrouter_sim_loads_no_local_sim():
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    cap, loaded = _run([
        "--sim-backend", "openrouter", "--sim-model", "openai/gpt-5-mini",
        "--sim-concurrency", "8", "--sim-max-cost", "5",
        "--veto-prompt", "P6", "--user-stakeholder", "--mixed-sampling",
        "--n-source-model", "2", "--n-source-critical", "2",
    ])
    # Exactly ONE local model: the policy. No sim copy, no vLLM server.
    assert len(loaded["llama"]) == 1, (
        f"expected only the policy loaded locally, got {loaded['llama']}")
    assert loaded["server"] == [], (
        f"no vLLM server should launch without --policy-on-vllm: {loaded['server']}")

    sim = cap["sim_stake"]
    # The sim's calls go to the API model, under the API model's id.
    assert sim.model_id == "openai/gpt-5-mini", sim.model_id
    assert sim.veto_model_id == "openai/gpt-5-mini", sim.veto_model_id
    assert sim.concerns_model_id == "openai/gpt-5-mini", sim.concerns_model_id
    assert getattr(sim.call_llm, "kind", None) is None, (
        "sim.call_llm must be the OpenRouter closure, not a tagged vLLM stub")
    # Fan-out is bounded for the API, not left at --stage-concurrency (128).
    assert sim.max_concurrency == 8, sim.max_concurrency
    assert sim._bounded(128) == 8
    print("ok: openrouter sim => no local sim model, API model id, capped fan-out")


def test_openrouter_sim_composes_with_policy_on_vllm():
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    cap, loaded = _run([
        "--sim-backend", "openrouter", "--policy-on-vllm",
        "--device", "cuda:0", "--sim-device", "cuda:1",
        "--veto-prompt", "P6", "--user-stakeholder", "--mixed-sampling",
        "--n-source-model", "2", "--n-source-critical", "2",
    ])
    # The engine still launches — for the POLICY, not the sim.
    assert len(loaded["server"]) == 1, (
        f"--policy-on-vllm must still launch the engine: {loaded['server']}")
    assert loaded["policy_sampler"] == ["Qwen/Qwen3-8B"], loaded["policy_sampler"]
    assert cap["policy_sampler"] is not None
    # ...and the sim is still the API model.
    assert cap["sim_stake"].model_id == "openai/gpt-5-mini"
    print("ok: openrouter sim composes with --policy-on-vllm (engine=policy only)")


def test_rewrite_on_sim_routing():
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    base = ["--sim-backend", "openrouter", "--veto-prompt", "P6",
            "--user-stakeholder", "--mixed-sampling",
            "--n-source-model", "2", "--n-source-critical", "2",
            "--perspective-shift-sampling"]
    cap_off, _ = _run(base)
    assert cap_off["rewrite_call_llm"] is None, (
        "without --rewrite-on-sim the rewrite must stay on the policy so "
        "existing perspshift arms are unchanged")
    cap_on, _ = _run(base + ["--rewrite-on-sim"])
    assert cap_on["rewrite_call_llm"] is not None
    assert cap_on["rewrite_call_llm"] is cap_on["sim_stake"].call_llm, (
        "the rewrite must use the SAME sim backend closure")
    print("ok: --rewrite-on-sim routes the rewrite to the sim; default = policy")


def test_local_vllm_backend_unchanged():
    cap, loaded = _run([
        "--sim-backend", "vllm", "--device", "cuda:0", "--sim-device", "cuda:1",
        "--veto-prompt", "P6", "--user-stakeholder", "--mixed-sampling",
        "--n-source-model", "2", "--n-source-critical", "2",
    ])
    assert len(loaded["server"]) == 1, loaded["server"]
    sim = cap["sim_stake"]
    assert sim.call_llm.kind == "vllm", "vllm backend must still build a vLLM sim"
    # The sim IS the policy's checkpoint here, so it keeps --model-id...
    assert sim.model_id == "Qwen/Qwen3-8B", sim.model_id
    # ...and imposes no cap: --stage-concurrency is tuned for this backend.
    assert sim.max_concurrency is None, sim.max_concurrency
    assert sim._bounded(128) == 128
    print("ok: local vllm backend unchanged (policy model id, no cap)")


if __name__ == "__main__":
    test_openrouter_sim_loads_no_local_sim()
    test_openrouter_sim_composes_with_policy_on_vllm()
    test_rewrite_on_sim_routing()
    test_local_vllm_backend_unchanged()
    print("\nall openrouter wiring tests passed")
