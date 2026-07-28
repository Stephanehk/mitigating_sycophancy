"""End-to-end tests for the participant API.

The API is now three endpoints wide, and the tests are organised around the two
questions that matter:

  1. Does the whole study arrive intact, exactly once, at Supabase?
  2. Are the guarantees the server refused to delegate actually enforced —
     one arm per session, one generation per session, and no way for a client
     to claim either?
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import questionnaires
import store
from flow_helpers import (DESCRIPTION_TEXT, PERSPECTIVE_TEXT, all_answers,
                          complete_study, drain_queue, eligible_payload, headers,
                          reach_response, request_response, screened_out_payload,
                          start_session)
from prompts import ARMS
import scenarios


# --- Session ---------------------------------------------------------------

def test_session_returns_a_token_the_order_and_the_whole_content_bundle(client):
    token, body = start_session(client)
    assert token
    assert sorted(body["screening_order"]) == sorted(scenarios.scenario_keys())
    assert body["has_job"] is False and body["submitted"] is False

    content = body["content"]
    # Everything the client needs to render the study without asking again.
    for key in ("consent", "prolific", "prelude", "screening", "perspective",
                "response_intro", "response", "questionnaire_pages", "debrief",
                "additional_thoughts", "terminal"):
        assert key in content, "content bundle is missing %r" % (key,)
    assert [p["page"] for p in content["questionnaire_pages"]] == \
        questionnaires.QUESTIONNAIRE_PAGE_IDS


def test_the_content_bundle_never_reveals_the_arm(client):
    """The arm is the randomized factor; a client that could read it could game it."""
    _, body = start_session(client)
    blob = json.dumps(body)
    for arm in ARMS:
        assert '"%s"' % arm not in blob, "arm %r leaked into the session payload" % (arm,)


def test_screening_order_varies_across_sessions(client):
    orders = {tuple(start_session(client)[1]["screening_order"]) for _ in range(25)}
    assert len(orders) > 1


def test_resuming_with_a_known_token_returns_the_same_session_and_order(client):
    token, body = start_session(client)
    again = client.post("/api/session", json={"session_token": token}).json()
    assert again["session_token"] == token
    assert again["screening_order"] == body["screening_order"]


def test_an_unknown_token_starts_a_fresh_session_rather_than_failing(client):
    body = client.post("/api/session", json={"session_token": "nope"}).json()
    assert body["session_token"] != "nope"


def test_endpoints_reject_a_missing_or_unknown_token(client):
    assert client.get("/api/response/status").status_code == 401
    assert client.get("/api/response/status", headers=headers("bogus")).status_code == 401
    assert client.post("/api/submit", json={"payload": {}},
                       headers=headers("bogus")).status_code == 401


# --- Generation ------------------------------------------------------------

def test_the_job_input_is_the_participants_own_words_only(client):
    """No vignette text may reach the model — it is a canned example, not their story."""
    token, body = start_session(client)
    payload = eligible_payload(body["screening_order"])
    request_response(client, token, payload)

    session = store.get_session_by_token(token)
    job = store.latest_job_for_session(session["session_id"])
    assert job["user_text"] == DESCRIPTION_TEXT + "\n\n" + PERSPECTIVE_TEXT.strip()
    for scenario in scenarios.SCENARIOS:
        assert scenario["vignette"] not in job["user_text"]
        assert scenario["question"] not in job["user_text"]


def test_starting_the_response_twice_queues_only_one_job(client):
    token, body = start_session(client)
    payload = eligible_payload(body["screening_order"])
    request_response(client, token, payload)
    request_response(client, token, payload)

    session = store.get_session_by_token(token)
    assert store.queue_depth() == {"pending": 1}
    assert store.latest_job_for_session(session["session_id"]) is not None


def test_a_second_start_cannot_change_the_model_input(client):
    """The one-way door: the stimulus cannot be edited after it exists."""
    token, body = start_session(client)
    payload = eligible_payload(body["screening_order"])
    request_response(client, token, payload)
    session = store.get_session_by_token(token)
    original = store.latest_job_for_session(session["session_id"])["user_text"]

    client.post("/api/response/start", json={
        "scenario_key": payload["selected_scenario"],
        "description": "a completely different story about something else entirely",
        "perspective": "and a completely different perspective on it, at length, here",
    }, headers=headers(token))

    assert store.latest_job_for_session(session["session_id"])["user_text"] == original


def test_the_arm_is_assigned_once_and_never_redrawn(client):
    token, body = start_session(client)
    payload = eligible_payload(body["screening_order"])
    request_response(client, token, payload)
    session = store.get_session_by_token(token)
    arm = session["arm"]
    assert arm in ARMS

    for _ in range(5):
        request_response(client, token, payload)
    assert store.get_session_by_token(token)["arm"] == arm


def test_a_session_that_never_requests_a_response_has_no_arm(client):
    """Screened-out participants must not consume an arm slot."""
    token, _ = start_session(client)
    assert store.get_session_by_token(token)["arm"] is None


def test_short_or_unknown_model_input_is_rejected_inline(client):
    token, body = start_session(client)
    order = body["screening_order"]
    short = client.post("/api/response/start", json={
        "scenario_key": order[0], "description": DESCRIPTION_TEXT, "perspective": "hi",
    }, headers=headers(token))
    assert short.status_code == 400 and "error" in short.json()

    unknown = client.post("/api/response/start", json={
        "scenario_key": "nope", "description": DESCRIPTION_TEXT,
        "perspective": PERSPECTIVE_TEXT,
    }, headers=headers(token))
    assert unknown.status_code == 400


def test_status_reports_done_and_the_text_after_the_worker_finishes(client):
    token, _, state = reach_response(client, text="A reply.")
    assert state["status"] == "done"
    assert state["text"] == "A reply."


def test_worker_failures_surface_a_retryable_state_then_exhaust(client, monkeypatch):
    monkeypatch.setattr(config, "JOB_MAX_ATTEMPTS", 1)
    token, body = start_session(client)
    request_response(client, token, eligible_payload(body["screening_order"]))

    worker = {"X-Worker-Token": config.WORKER_TOKEN}
    job = client.post("/api/worker/claim", json={"worker_id": "w"},
                      headers=worker).json()["job"]
    client.post("/api/worker/fail", json={"job_id": job["job_id"], "worker_id": "w",
                                          "error": "boom"}, headers=worker)

    state = client.get("/api/response/status", headers=headers(token)).json()["response"]
    assert state["status"] == "failed" and state["can_retry"] is True

    assert client.post("/api/response/retry", json={}, headers=headers(token)).status_code == 200
    job = client.post("/api/worker/claim", json={"worker_id": "w"},
                      headers=worker).json()["job"]
    client.post("/api/worker/fail", json={"job_id": job["job_id"], "worker_id": "w",
                                          "error": "boom"}, headers=worker)

    state = client.get("/api/response/status", headers=headers(token)).json()["response"]
    assert state["status"] == "failed" and state["can_retry"] is False
    assert client.post("/api/response/retry", json={}, headers=headers(token)).status_code == 409


# --- Submission ------------------------------------------------------------

def test_a_completed_study_reaches_supabase_once_and_intact(client, submitted):
    token, result = complete_study(client, text="The model said this.")
    assert result["ok"] is True and result["duplicate"] is False
    assert len(submitted) == 1

    row = submitted[0]
    assert row["consent"] == "agree"
    assert row["eligibility"] == "eligible"
    assert row["prolific_pid"] == "TESTPID"
    assert row["perspective_text"] == PERSPECTIVE_TEXT.strip()
    assert row["additional_thoughts"] == "Thanks!"
    assert row["answers"] == all_answers()
    assert row["missing_keys"] == []
    # Server-owned facts, merged in rather than trusted from the client.
    assert row["arm"] in ARMS
    assert row["shown_response"] == "The model said this."
    assert row["shown_response_chars"] == len("The model said this.")
    assert row["user_text"] == DESCRIPTION_TEXT + "\n\n" + PERSPECTIVE_TEXT.strip()
    assert row["stages"] == [{"name": "generate", "output": "The model said this."}]


def test_every_questionnaire_key_survives_the_round_trip(client, submitted):
    complete_study(client)
    row = submitted[0]
    assert set(row["answers"]) == set(questionnaires.all_response_keys()), \
        "a questionnaire item was lost between the browser and Supabase"


def test_a_client_cannot_forge_its_arm(client, submitted):
    """The arm is never sent to the browser, so a claimed one must be ignored."""
    token, payload, _ = reach_response(client)
    payload["arm"] = "syco"
    payload["shown_response"] = "something the participant made up"
    client.post("/api/submit", json={"payload": payload}, headers=headers(token))

    session = store.get_session_by_token(token)
    assert submitted[0]["arm"] == session["arm"]
    assert submitted[0]["shown_response"] == "Here is the model's reply."


def test_submitting_twice_writes_supabase_once(client, submitted):
    token, payload, _ = reach_response(client)
    first = client.post("/api/submit", json={"payload": payload}, headers=headers(token))
    second = client.post("/api/submit", json={"payload": payload}, headers=headers(token))
    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is True
    assert len(submitted) == 1


def test_a_screened_out_participant_still_submits(client, submitted):
    """They reached the end of their path; the screening funnel is a result."""
    token, body = start_session(client)
    payload = screened_out_payload(body["screening_order"])
    response = client.post("/api/submit", json={"payload": payload}, headers=headers(token))
    assert response.status_code == 200
    assert len(submitted) == 1
    assert submitted[0]["eligibility"] == "screened_out"
    assert submitted[0]["arm"] is None
    assert len(submitted[0]["screening"]) == 4


def test_a_decliner_submits_with_no_answers(client, submitted):
    token, body = start_session(client)
    payload = screened_out_payload(body["screening_order"],
                                   consent="decline", eligibility="declined",
                                   screening=[])
    assert client.post("/api/submit", json={"payload": payload},
                       headers=headers(token)).status_code == 200
    assert submitted[0]["consent"] == "decline"


def test_claiming_eligibility_without_a_generation_is_rejected(client, submitted):
    token, body = start_session(client)
    payload = eligible_payload(body["screening_order"])
    response = client.post("/api/submit", json={"payload": payload}, headers=headers(token))
    assert response.status_code == 400
    assert submitted == []


def test_a_malformed_payload_is_rejected_and_nothing_is_written(client, submitted):
    token, payload, _ = reach_response(client)
    payload["consent"] = "maybe"
    response = client.post("/api/submit", json={"payload": payload}, headers=headers(token))
    assert response.status_code == 400
    assert submitted == []
    assert store.get_session_by_token(token)["submitted_at"] is None


def test_a_supabase_outage_does_not_fail_the_participant(client, submitted, monkeypatch):
    """The row is already durable on disk; the participant has finished."""
    import supabase_client
    monkeypatch.setattr(supabase_client, "post_row", lambda row: "http 503: down")

    token, payload, _ = reach_response(client)
    response = client.post("/api/submit", json={"payload": payload}, headers=headers(token))
    assert response.status_code == 200
    assert response.json()["supabase_error"] == "http 503: down"

    with open(config.SUBMISSION_FALLBACK_PATH, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert len(lines) == 1
    assert lines[0]["session_id"] == store.get_session_by_token(token)["session_id"]


def test_the_fallback_file_is_written_before_the_network_call(client, submitted, monkeypatch):
    """Ordering is the durability guarantee, so assert it directly."""
    import supabase_client
    seen = {}

    def post_and_check(row):
        with open(config.SUBMISSION_FALLBACK_PATH, encoding="utf-8") as handle:
            seen["already_on_disk"] = len([line for line in handle if line.strip()])
        submitted.append(row)
        return None

    monkeypatch.setattr(supabase_client, "post_row", post_and_check)
    complete_study(client)
    assert seen["already_on_disk"] == 1


# --- Ops -------------------------------------------------------------------

def test_health_reports_the_queue_arms_and_the_dropout_blind_spot(client, submitted):
    complete_study(client)
    # A second participant who saw the response and wandered off.
    reach_response(client)

    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert sum(body["submitted_by_arm"].values()) == 1
    assert body["saw_response_never_submitted"] == 1
    assert body["queue"]["done"] == 2
