"""Helpers that drive a test participant through the study.

The client owns the flow now, so most of what these helpers do is BUILD THE
PAYLOAD a browser would have accumulated locally — there are no per-page
endpoints left to call. The only real HTTP steps are session creation, the
generation request, and the final submit.

Shared by test_server.py and test_export.py so the two cannot drift on what a
valid submission looks like.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import questionnaires
import scenarios

PERSPECTIVE_TEXT = "My side of it is that I only had room for a few people. " * 2
DESCRIPTION_TEXT = "d" * 60


def headers(token):
    return {"X-Session-Token": token}


def start_session(client):
    """POST /api/session. Returns (token, body)."""
    body = client.post("/api/session", json={}).json()
    return body["session_token"], body


def answers_for_page(page_id, value_index=0):
    """A complete, valid answer set for one questionnaire page.

    Picks each question's `value_index`-th option, so calling it with a
    different index produces a genuinely different submission.
    """
    page = questionnaires.get_page(page_id)
    answers = {}
    for question in page["questions"]:
        option = question["options"][min(value_index, len(question["options"]) - 1)]
        if question["type"] == "matrix":
            for item in question["items"]:
                answers[item["id"]] = option["value"]
        elif question["type"] == "multi":
            answers[question["id"]] = [option["value"]]
        else:
            answers[question["id"]] = option["value"]
    return answers


def all_answers(value_index=0):
    """A complete answer set across every questionnaire page."""
    answers = {}
    for page_id in questionnaires.QUESTIONNAIRE_PAGE_IDS:
        answers.update(answers_for_page(page_id, value_index))
    return answers


def eligible_payload(screening_order, selected=None, **overrides):
    """The payload a browser would hold for a participant who finished.

    `selected` defaults to the first scenario in the presentation order, i.e.
    someone who matched immediately. Screening answers before the match are
    "not really", matching how the client records them.
    """
    selected = selected or screening_order[0]
    index = screening_order.index(selected)
    screening = []
    for position, key in enumerate(screening_order):
        if position < index:
            screening.append({"scenario_key": key, "similarity": "not_really",
                              "description": None})
        elif position == index:
            screening.append({"scenario_key": key, "similarity": "very",
                              "description": DESCRIPTION_TEXT})
    payload = {
        "consent": "agree",
        "prolific_pid": "TESTPID",
        "eligibility": "eligible",
        "selected_scenario": selected,
        "screening": screening,
        "perspective_text": PERSPECTIVE_TEXT,
        "additional_thoughts": "Thanks!",
        "answers": all_answers(),
        "client_timings": {"consent": "2026-07-28T00:00:00.000Z"},
    }
    payload.update(overrides)
    return payload


def screened_out_payload(screening_order, **overrides):
    """The payload for someone who answered "not really" to all four."""
    payload = {
        "consent": "agree",
        "prolific_pid": "TESTPID",
        "eligibility": "screened_out",
        "selected_scenario": None,
        "screening": [{"scenario_key": key, "similarity": "not_really",
                       "description": None} for key in screening_order],
        "perspective_text": "",
        "additional_thoughts": "",
        "answers": {},
        "client_timings": {},
    }
    payload.update(overrides)
    return payload


def request_response(client, token, payload):
    """POST /api/response/start with the two texts the model needs."""
    selected = [entry for entry in payload["screening"]
                if entry["scenario_key"] == payload["selected_scenario"]][0]
    return client.post("/api/response/start", json={
        "scenario_key": payload["selected_scenario"],
        "description": selected["description"],
        "perspective": payload["perspective_text"],
    }, headers=headers(token))


def drain_queue(client, text="Here is the model's reply."):
    """Act as the GPU worker: claim every pending job and complete it."""
    worker = {"X-Worker-Token": config.WORKER_TOKEN}
    completed = 0
    while True:
        job = client.post("/api/worker/claim", json={"worker_id": "test"},
                          headers=worker).json()["job"]
        if job is None:
            return completed
        client.post("/api/worker/complete", json={
            "job_id": job["job_id"], "worker_id": "test",
            "response_text": text,
            "stages": [{"name": "generate", "output": text}],
        }, headers=worker)
        completed += 1


def reach_response(client, text="Here is the model's reply."):
    """Session created, generation requested and completed.

    Returns (token, payload, response_state) with the payload the client would
    still be holding.
    """
    token, body = start_session(client)
    payload = eligible_payload(body["screening_order"])
    request_response(client, token, payload)
    drain_queue(client, text)
    state = client.get("/api/response/status", headers=headers(token)).json()["response"]
    assert state["status"] == "done", state
    return token, payload, state


def complete_study(client, text="Here is the model's reply.", **overrides):
    """Walk a participant all the way through and submit. Returns (token, body)."""
    token, payload, _ = reach_response(client, text)
    payload.update(overrides)
    response = client.post("/api/submit", json={"payload": payload},
                           headers=headers(token))
    assert response.status_code == 200, response.text
    return token, response.json()
