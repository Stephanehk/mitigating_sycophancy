"""Tests for views.py: the one-shot content bundle and the response state.

The bundle is now the client's entire source of truth for what the study says,
so the thing worth testing is that it is COMPLETE — every scenario, every
questionnaire item, every terminal message — and that it does not carry the one
fact the client must never learn.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import content
import questionnaires
import scenarios
import store
import views
from prompts import ARMS


@pytest.fixture()
def db(monkeypatch):
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    os.unlink(path)
    monkeypatch.setattr(config, "DATABASE_URL", "sqlite:///" + path)
    store.init_db()
    yield path
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.unlink(path + suffix)


# --- Content bundle --------------------------------------------------------

def test_the_bundle_carries_every_scenario_with_all_three_rendered_fields():
    bundle = views.content_bundle()
    assert set(bundle["screening"]["scenarios"]) == set(scenarios.scenario_keys())
    for key, scenario in bundle["screening"]["scenarios"].items():
        source = scenarios.get_scenario(key)
        assert scenario["title"] == source["title"]
        assert scenario["vignette"] == source["vignette"]
        assert scenario["question"] == source["question"]


def test_the_bundle_carries_every_questionnaire_page_and_item():
    bundle = views.content_bundle()
    pages = bundle["questionnaire_pages"]
    assert [p["page"] for p in pages] == questionnaires.QUESTIONNAIRE_PAGE_IDS

    bundled_keys = []
    for page in pages:
        for question in page["questions"]:
            bundled_keys.extend(questionnaires.response_keys(question))
    assert bundled_keys == questionnaires.all_response_keys(), \
        "the client cannot render an item the bundle does not carry"


def test_the_bundle_never_carries_the_arm_or_any_prompt():
    """The client must not be able to read, infer or claim its condition."""
    import prompts
    blob = json.dumps(views.content_bundle())
    for arm in ARMS:
        assert '"%s"' % arm not in blob
    assert prompts.BASE_SYSTEM_PROMPT not in blob
    assert prompts.SYCO_SYSTEM_PROMPT not in blob


def test_the_bundle_carries_all_three_terminal_destinations():
    terminal = views.content_bundle()["terminal"]
    assert set(terminal) == {"complete", "screened_out", "declined"}
    assert terminal["complete"]["redirect_url"] == config.PROLIFIC_COMPLETION_URL
    assert terminal["screened_out"]["redirect_url"] == config.PROLIFIC_SCREENOUT_URL
    assert terminal["declined"]["redirect_url"] == config.PROLIFIC_SCREENOUT_URL


def test_the_bundle_carries_the_validation_thresholds_the_client_enforces():
    bundle = views.content_bundle()
    assert bundle["screening"]["min_describe_chars"] == config.MIN_DESCRIBE_CHARS
    assert bundle["perspective"]["min_chars"] == config.MIN_PERSPECTIVE_CHARS
    assert bundle["max_text_chars"] == config.MAX_TEXT_CHARS


def test_the_bundle_agrees_with_scenarios_about_which_answer_selects():
    bundle = views.content_bundle()
    expected = [c["value"] for c in scenarios.SIMILARITY_CHOICES if c["selects"]]
    assert bundle["screening"]["selects_scenario"] == expected
    assert bundle["screening"]["describe_required_for"] == list(scenarios.DESCRIBE_REQUIRED_FOR)


def test_the_bundle_is_json_serializable():
    """It is sent verbatim over the wire, so a stray tuple would 500 at runtime."""
    json.dumps(views.content_bundle())


# --- Response state --------------------------------------------------------

def test_response_state_reports_pending_then_done(db):
    session_id, _ = store.create_session(["a"], "pytest")
    job_id = store.create_job(session_id, "ours", "Topic", "story")

    state = views.response_state(session_id)
    assert state["status"] == "pending"
    assert "text" not in state

    store.claim_job("w")
    store.complete_job(job_id, "w", "the reply", [])
    state = views.response_state(session_id)
    assert state["status"] == "done" and state["text"] == "the reply"


def test_response_state_reports_retryability_until_exhausted(db):
    session_id, _ = store.create_session(["a"], "pytest")
    job_id = store.create_job(session_id, "ours", "Topic", "story")
    with store.transaction() as conn:
        conn.execute("UPDATE jobs SET status = 'failed' WHERE job_id = ?", (job_id,))

    assert views.response_state(session_id)["can_retry"] is True
    store.requeue_job(job_id)
    with store.transaction() as conn:
        conn.execute("UPDATE jobs SET status = 'failed' WHERE job_id = ?", (job_id,))
    assert views.response_state(session_id)["can_retry"] is False


def test_response_state_without_a_job_is_a_programming_error(db):
    session_id, _ = store.create_session(["a"], "pytest")
    with pytest.raises(AssertionError):
        views.response_state(session_id)
