"""Tests for the GPU worker's job execution, with vLLM stubbed out.

`run_job` is the code that turns an arm into actual model calls, so these
verify the properties that determine whether the manipulation is actually
delivered correctly:
  - each arm addresses the right served model
  - the syco/neutral arms send their system prompt; the bare arms send none
  - the neutral arm's editing stage receives Stage 1's OUTPUT, not the
    participant's text
  - every stage's raw output is reported for analysis
  - a generation error is reported as a failed attempt, never as a completion

The GPU itself is not exercised here; `worker/study_worker.sbatch` is the
integration path and README.md documents the smoke test for it.
"""

import os
import sys

import pytest

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY_DIR)
sys.path.insert(0, os.path.join(_STUDY_DIR, "worker"))

import config
import prompts
import run_worker


class FakeCompletions:
    """Records every chat.completions.create call and returns canned text."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.outputs.pop(0) if self.outputs else "default output"

        class Message:
            content = text

        class Choice:
            message = Message()

        class Completion:
            choices = [Choice()]

        return Completion()


class FakeOpenAI:
    """Stands in for the OpenAI client inside a real VLLMBackend, so the
    backend's own request-shaping code is what the tests exercise."""

    def __init__(self, outputs):
        self.completions = FakeCompletions(outputs)
        self.chat = self

    @property
    def calls(self):
        return self.completions.calls


def make_backend(outputs):
    """A real VLLMBackend with its OpenAI client swapped for a recorder."""
    backend = run_worker.VLLMBackend.__new__(run_worker.VLLMBackend)
    backend.base_url = "http://127.0.0.1:0"
    backend.client = FakeOpenAI(outputs)
    return backend


class FakeStudy:
    """Stand-in for StudyClient that records completions and failures."""

    def __init__(self):
        self.completed = []
        self.failed = []
        self.heartbeats = []

    def heartbeat(self, job_id):
        self.heartbeats.append(job_id)
        return True

    def complete(self, job_id, response_text, stages):
        self.completed.append((job_id, response_text, stages))
        return True

    def fail(self, job_id, error):
        self.failed.append((job_id, error))
        return "pending"


JOB_USER_TEXT = ("I didn't invite my sibling to my party because I wanted to keep it small.\n\n"
                 "I did not think they would mind.")


def make_job(arm):
    """A claimed job as the server hands it over: `user_text` is already the
    composed user turn (vignette + perspective), not a bare perspective."""
    return {"job_id": "job-1", "arm": arm, "topic_title": "Excluding Someone",
            "user_text": JOB_USER_TEXT, "attempts": 1}


# --- Per-arm model routing -------------------------------------------------

def test_ours_arm_addresses_the_lora_with_the_shared_system_prompt():
    openai = make_backend(["our reply"])
    study = FakeStudy()
    run_worker.run_job(openai, study, make_job(prompts.ARM_OURS))

    assert len(openai.client.calls) == 1
    call = openai.client.calls[0]
    assert call["model"] == config.LORA_NAME
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["messages"][0]["content"] == prompts.BASE_SYSTEM_PROMPT
    assert "[SYCOPHANCY POLICY]" not in call["messages"][0]["content"]
    assert study.completed[0][1] == "our reply"


def test_untrained_arm_addresses_the_base_with_the_same_system_prompt_as_ours():
    openai = make_backend(["base reply"])
    run_worker.run_job(openai, FakeStudy(), make_job(prompts.ARM_UNTRAINED))
    call = openai.client.calls[0]
    assert call["model"] == config.BASE_MODEL_ID
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["messages"][0]["content"] == prompts.BASE_SYSTEM_PROMPT


def test_syco_arm_sends_the_policy_system_prompt_to_the_base_model():
    openai = make_backend(["you were completely right"])
    run_worker.run_job(openai, FakeStudy(), make_job(prompts.ARM_SYCO))
    call = openai.client.calls[0]
    assert call["model"] == config.BASE_MODEL_ID
    assert call["messages"][0]["role"] == "system"
    assert "[SYCOPHANCY POLICY]" in call["messages"][0]["content"]
    # The scenario title no longer reaches any prompt.
    assert "Excluding Someone" not in call["messages"][0]["content"]


def test_the_composed_user_turn_reaches_the_model_intact_for_every_arm():
    for arm in prompts.ARMS:
        openai = make_backend(["reply", "edited reply"])
        run_worker.run_job(openai, FakeStudy(), make_job(arm))
        first = openai.client.calls[0]
        assert first["messages"][-1]["content"] == JOB_USER_TEXT


# --- Neutral two-stage chain ----------------------------------------------

def test_neutral_arm_feeds_stage1_output_into_the_editor():
    """The editing pass must rewrite the model's text, not the participant's."""
    openai = make_backend(["that must have been really hard for you", "here is one framing"])
    study = FakeStudy()
    run_worker.run_job(openai, study, make_job(prompts.ARM_NEUTRAL))

    assert len(openai.client.calls) == 2
    generate, edit = openai.client.calls

    assert "[SYCOPHANCY POLICY]" not in generate["messages"][0]["content"]
    assert generate["messages"][1]["content"] == JOB_USER_TEXT

    assert "rewrites AI assistant messages" in edit["messages"][0]["content"]
    # The key property: stage 2's user turn is stage 1's OUTPUT.
    assert edit["messages"][1]["content"] == "that must have been really hard for you"
    assert "I did not invite my sibling" not in edit["messages"][1]["content"]

    # The participant is shown the edited text, and both stages are recorded.
    job_id, final, stages = study.completed[0]
    assert final == "here is one framing"
    assert [s["name"] for s in stages] == ["generate", "edit"]
    assert stages[0]["output"] == "that must have been really hard for you"
    assert stages[1]["output"] == "here is one framing"


def test_neutral_arm_extends_its_lease_between_stages():
    """A two-stage generation must not be reaped mid-flight."""
    study = FakeStudy()
    run_worker.run_job(make_backend(["a", "b"]), study, make_job(prompts.ARM_NEUTRAL))
    assert study.heartbeats == ["job-1"]


def test_single_stage_arms_do_not_heartbeat():
    study = FakeStudy()
    run_worker.run_job(make_backend(["a"]), study, make_job(prompts.ARM_OURS))
    assert study.heartbeats == []


# --- Sampling parity -------------------------------------------------------

def test_sampling_matches_the_offline_eval_condition():
    """enable_thinking=False and temperature parity with eval_endorsement_heldout."""
    openai = make_backend(["x"])
    run_worker.run_job(openai, FakeStudy(), make_job(prompts.ARM_OURS))
    call = openai.client.calls[0]
    assert call["temperature"] == config.GEN_TEMPERATURE
    assert call["top_p"] == config.GEN_TOP_P
    assert call["max_tokens"] == config.GEN_MAX_TOKENS
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_all_four_arms_produce_a_completion():
    for arm in prompts.ARMS:
        study = FakeStudy()
        run_worker.run_job(make_backend(["one", "two"]), study, make_job(arm))
        assert len(study.completed) == 1, "arm %s did not complete" % arm
        assert study.completed[0][1]


# --- Failure handling ------------------------------------------------------

def test_empty_generation_raises_rather_than_shipping_a_blank_bubble():
    openai = make_backend(["   "])
    study = FakeStudy()
    with pytest.raises(AssertionError):
        run_worker.run_job(openai, study, make_job(prompts.ARM_OURS))
    assert study.completed == []


def test_a_stage_error_prevents_completion():
    """A failure in stage 2 must not complete the job with stage 1's text."""
    class ExplodingCompletions(FakeCompletions):
        def create(self, **kwargs):
            # The engine dies on the editing pass, after stage 1 succeeded.
            if len(self.calls) == 1:
                self.calls.append(kwargs)
                raise RuntimeError("cuda oom")
            return super().create(**kwargs)

    openai = make_backend(["stage one text"])
    openai.client.completions = ExplodingCompletions(["stage one text"])
    study = FakeStudy()
    with pytest.raises(RuntimeError):
        run_worker.run_job(openai, study, make_job(prompts.ARM_NEUTRAL))
    assert study.completed == [], "job completed despite the editing stage failing"
    assert len(openai.client.calls) == 2


def test_unknown_arm_is_rejected_before_any_model_call():
    openai = make_backend(["x"])
    with pytest.raises(AssertionError):
        run_worker.run_job(openai, FakeStudy(), make_job("bogus_arm"))
    assert openai.client.calls == []


# --- The sandwich arm ------------------------------------------------------
# Added 2026-08-07. Two stages on DIFFERENT models, and a guard on the second.

def test_the_sandwich_arm_runs_stage_one_on_the_lora_and_stage_two_on_the_base():
    """The per-stage adapter, end to end.

    `adapter` used to be one value for the whole plan. If it silently reverted,
    this arm would frame with our adapter (or generate with the base) and the
    ours-vs-ours_sandwich contrast would no longer isolate the framing.
    """
    body = "You were not upfront, and that put someone else in a bad position."
    openai = make_backend([body, "Opener. %s Closer." % body])
    study = FakeStudy()
    run_worker.run_job(openai, study, make_job(prompts.ARM_OURS_SANDWICH))

    generate, sandwich = openai.client.calls
    assert generate["model"] == config.LORA_NAME
    assert sandwich["model"] == config.BASE_MODEL_ID
    assert generate["max_tokens"] == config.GEN_MAX_TOKENS
    # The rewrite must have room for the whole of Stage 1 plus its additions.
    assert sandwich["max_tokens"] == config.GEN_REWRITE_MAX_TOKENS

    _, shown, stages = study.completed[0]
    assert shown == "Opener. %s Closer." % body
    assert [s["name"] for s in stages] == ["generate", "sandwich"]
    assert [s["model"] for s in stages] == [config.LORA_NAME, config.BASE_MODEL_ID]


def test_the_sandwich_stage_receives_both_the_participant_text_and_stage_one():
    """It cannot write the bread without both halves.

    The participant's message tells it what they feel; Stage 1's output is what
    it is framing. The neutral arm's editing stage takes only the latter, so
    this is the distinguishing property of the new stage form.
    """
    openai = make_backend(["stage one text", "wrapped"])
    run_worker.run_job(openai, FakeStudy(), make_job(prompts.ARM_OURS_SANDWICH))

    sandwich_input = openai.client.calls[1]["messages"][-1]["content"]
    assert JOB_USER_TEXT in sandwich_input, "the participant's own message is missing"
    assert "stage one text" in sandwich_input, "Stage 1's output is missing"
    # Labelled, so the model can tell them apart rather than answering the lot.
    assert sandwich_input.index(JOB_USER_TEXT) < sandwich_input.index("stage one text")


def test_a_sandwich_rewrite_that_summarises_the_body_is_not_shown():
    """THE SAFETY GUARD, and the reason it exists.

    In development the rewrite sometimes condensed a long reply instead of
    framing it, once deleting a crisis hotline number from a response to a user
    describing suicidal ideation. When too little of Stage 1 survives, the
    participant sees Stage 1 UNCHANGED rather than an abridged answer.
    """
    body = ("Call the 988 Suicide and Crisis Lifeline. " * 20).strip()
    openai = make_backend([body, "I hear you. Please take care of yourself."])
    study = FakeStudy()
    run_worker.run_job(openai, study, make_job(prompts.ARM_OURS_SANDWICH))

    _, shown, stages = study.completed[0]
    assert shown == body, "the summarised rewrite was shown to the participant"
    assert "988" in shown
    # The rejected rewrite is KEPT in the record, so the fallback rate is
    # measurable in the export rather than invisible.
    assert stages[-1]["output"] == "I hear you. Please take care of yourself."
    assert stages[-1]["body_preserved_ok"] is False
    assert stages[-1]["body_preserved"] < prompts.MIN_BODY_PRESERVED


def test_a_faithful_sandwich_rewrite_is_shown_and_scored():
    """The other side of the guard: a good rewrite must survive it.

    Scored, not merely accepted — the fraction is exported either way, so the
    threshold can be re-examined against real data instead of re-litigated.
    """
    body = "Your numbers were right, but the way you raised them landed badly."
    rewrite = "That sounds frustrating. %s None of that makes you wrong to care." % body
    openai = make_backend([body, rewrite])
    study = FakeStudy()
    run_worker.run_job(openai, study, make_job(prompts.ARM_OURS_SANDWICH))

    _, shown, stages = study.completed[0]
    assert shown == rewrite
    assert stages[-1]["body_preserved_ok"] is True
    assert stages[-1]["body_preserved"] == 1.0


def test_the_guard_tolerates_a_tidied_seam():
    """A dropped leading word is what the prompt explicitly permits.

    This is the real development case: one good rewrite turned "First, it's
    important" into "It's important" and would have failed an exact-substring
    check while reproducing 99.6% of the text. The guard is a ratio for exactly
    this reason.
    """
    body = ("First, consider why this happened. " + "It matters how you raised it. " * 30).strip()
    rewrite = "That sounds hard. " + body[len("First, "):] + " Take care of yourself."
    ok, fraction = prompts.body_preserved(body, rewrite)
    assert ok, "a seam adjustment was treated as a summary (kept %.3f)" % fraction
    assert fraction > 0.99


# --- compose_stage_user ----------------------------------------------------

def test_compose_stage_user_resolves_all_three_stage_forms():
    literal = {"name": "generate", "user": "the participant's words"}
    assert prompts.compose_stage_user(literal, None) == "the participant's words"

    chained = {"name": "edit", "user": None}
    assert prompts.compose_stage_user(chained, "stage one") == "stage one"

    prefixed = {"name": "sandwich", "user": None, "user_prefix": "CONTEXT\n"}
    assert prompts.compose_stage_user(prefixed, "stage one") == "CONTEXT\nstage one"


def test_compose_stage_user_refuses_a_chained_first_stage():
    """A plan whose first stage consumes a previous output is malformed, and
    generating from an empty prompt would be a silent, plausible-looking bug."""
    with pytest.raises(AssertionError):
        prompts.compose_stage_user({"name": "edit", "user": None}, None)


def test_body_preserved_rejects_an_empty_original():
    """Not a 100% match — a caller reaching this has already lost Stage 1."""
    with pytest.raises(AssertionError):
        prompts.body_preserved("", "anything")
