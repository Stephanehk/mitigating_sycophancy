"""Unit tests for the voice classifier's backend selection and label parsing.

These are network-free: they exercise the pure helpers and assert that the
API backends are wired without importing vLLM/CUDA or contacting any provider.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "core"))

import classify_voice_only as C


def test_parse_voice_reads_first_label():
    assert C.parse_voice("OUTSIDER") == "OUTSIDER"
    assert C.parse_voice("  stakeholder  ") == "STAKEHOLDER"
    # UNCLEAR is a real label in V2.LABELS, so it parses to itself.
    assert C.parse_voice("unclear") == "UNCLEAR"


def test_parse_voice_unparsable_never_guesses():
    assert C.parse_voice("") == "UNPARSED"
    assert C.parse_voice(None) == "UNPARSED"
    assert C.parse_voice("the response is neutral in tone") == "UNPARSED"


def test_voice_to_label_mapping():
    assert C.voice_to_label("OUTSIDER") == 0        # chosen-eligible
    assert C.voice_to_label("STAKEHOLDER") == 2     # rejected-eligible
    assert C.voice_to_label("UNCLEAR") == -1        # dropped
    assert C.voice_to_label("UNPARSED") == -1       # dropped, never guessed


def test_all_three_backends_registered():
    assert set(C.BACKENDS) == {"local", "gemini", "openrouter"}


def test_gemini_backend_rejects_openrouter_style_id():
    class A:
        backend = "gemini"
        model_id = "google/gemini-3.6-flash"  # slash id belongs to openrouter
    try:
        C._gemini_classifier(A())
        assert False, "expected an assertion on a slash id"
    except AssertionError as e:
        assert "plain Gemini id" in str(e)


def test_openrouter_backend_requires_key(monkeypatch=None):
    class A:
        backend = "openrouter"
        model_id = "openai/gpt-5-mini"
    saved = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        C._openrouter_classifier(A())
        assert False, "expected an assertion when OPENROUTER_API_KEY is unset"
    except AssertionError as e:
        assert "OPENROUTER_API_KEY" in str(e)
    finally:
        if saved is not None:
            os.environ["OPENROUTER_API_KEY"] = saved


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all voice-backend tests passed")
