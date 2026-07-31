"""The study's behaviour, driven against the real database.

This is the suite that replaces test_server.py, test_flow.py and test_store.py.
Those tested Python that no longer runs in production — a FastAPI app nothing
deploys, and validation that now lives in plpgsql — and they passed while the
live system was broken, twice over. See conftest.py for the two specific bugs.

Everything a participant does is exercised with the PUBLIC anon key, because
that is what a browser holds. The service key appears only to read back what
landed and to clean up.

Organised by what could actually go wrong:
  1. the happy path, end to end
  2. randomization — the arm and the screening order
  2b. the flow fork — which arms show the review-and-edit page, and the fact
      that asking is the only thing the browser learns about its arm
  2c. the /debug_<key> links — the one sanctioned way to choose an arm, and the
      flag that keeps a hand-walked session out of the dataset
  3. the two integrity rules: the arm is unforgeable, one job per session
  4. validation, and the deliberate difference between "reject" and "record"
  5. the queue under concurrency
  6. the permission boundary
"""

import concurrent.futures as cf
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import questionnaires
import scenarios
from prompts import ARMS, DEBUG_CONDITIONS, SKIP_REVIEW_ARMS

DESC = "I did not invite my sibling to my birthday party because I only had six seats."
PERS = "My side is that the table was genuinely too small and I meant no slight by it."


def all_answers(value_index=0):
    """A complete answer set: each question's `value_index`-th option."""
    answers = {}
    for question in questionnaires.all_questions():
        option = question["options"][min(value_index, len(question["options"]) - 1)]
        for key in questionnaires.response_keys(question):
            answers[key] = option["value"]
    return answers


def payload(session, **overrides):
    """What a browser would submit for an eligible, completing participant."""
    body = {
        "consent": "agree", "prolific_pid": "PYTEST", "eligibility": "eligible",
        "selected_scenario": session["screening_order"][0],
        "screening": [{"scenario_key": session["screening_order"][0],
                       "similarity": "very", "description": DESC}],
        "perspective_text": PERS, "additional_thoughts": "",
        "answers": all_answers(), "client_timings": {},
    }
    body.update(overrides)
    return body


def start(anon):
    return anon.rpc("start_session", {"p_user_agent": "pytest"})


def ask_flow(anon, session):
    """What the browser asks on the response_intro page: review page or not?"""
    return anon.rpc("response_flow", {"p_token": session["session_token"]})


def arm_of(service, session):
    return service.get("/study_sessions?session_id=eq.%s&select=arm"
                       % session["session_id"])[0]["arm"]


def request_generation(anon, session, description=DESC, perspective=PERS):
    """Confirm the review page and enqueue the generation.

    start_response takes the FINISHED user turn, not the two halves — the
    review page composes it in the browser so the participant's edit is what
    gets sent. This mirrors what screens.js does.
    """
    return anon.rpc("start_response", {
        "p_token": session["session_token"],
        "p_scenario_key": session["screening_order"][0],
        "p_user_text": description + "\n\n" + perspective})


def finish_generation(anon, session, worker_token, text="the model reply",
                      worker="pytest-w", timeout=120):
    """Get this session's job to `done`, and return the text it ended up with.

    SHARED-DATABASE HAZARD. These tests run against the same project a live GPU
    worker pool may be polling, so the pool will sometimes claim a test's job
    first. Asserting "I claimed it" would then fail for a reason that has
    nothing to do with the behaviour under test — which it did, on three tests,
    the first time this suite ran against a live pool.

    So: claim it if it is still pending, otherwise wait for whoever did claim it
    to finish. The caller compares against the RETURNED text rather than a
    literal, so the assertions hold either way.

    The real fix is a separate Supabase project for tests; see README.
    """
    claimed = anon.rpc("worker_claim", {"p_worker_token": worker_token,
                                        "p_worker_id": worker})["job"]
    if claimed is not None:
        anon.rpc("worker_complete", {
            "p_worker_token": worker_token, "p_job_id": claimed["job_id"],
            "p_worker_id": worker, "p_response_text": text,
            "p_stages": [{"name": "generate", "output": text}]})
        return text

    deadline = time.time() + timeout
    while time.time() < deadline:
        state = anon.rpc("response_status", {"p_token": session["session_token"]})
        if state["status"] == "done":
            return state["text"]
        assert state["status"] != "failed", "the live pool failed this job: %s" % (state,)
        time.sleep(2)
    raise AssertionError("job never completed within %ds" % timeout)


# --- 1. The happy path -----------------------------------------------------

def test_a_participant_can_complete_the_whole_study(anon, service, worker_token, clean):
    session = start(anon)
    request_generation(anon, session)
    shown = finish_generation(anon, session, worker_token, text="A generated reply.")

    state = anon.rpc("response_status", {"p_token": session["session_token"]})
    assert state["status"] == "done" and state["text"] == shown

    result = anon.rpc("submit_study", {"p_token": session["session_token"],
                                       "p_payload": payload(session)})
    assert result["ok"] is True and result["duplicate"] is False

    row = service.get("/study_submissions?session_id=eq.%s&select=*"
                      % session["session_id"])[0]
    assert row["consent"] == "agree" and row["eligibility"] == "eligible"
    assert row["shown_response"] == shown
    assert row["user_text"] == DESC + "\n\n" + PERS
    assert row["perspective_text"] == PERS
    assert len(row["answers"]) == len(questionnaires.all_response_keys())
    assert row["missing_keys"] == []
    assert row["stages"][0]["name"] == "generate"


def test_a_screened_out_participant_submits_with_no_arm(anon, service, clean):
    """The case a `not null` constraint silently broke, and mocks did not catch."""
    session = start(anon)
    result = anon.rpc("submit_study", {"p_token": session["session_token"],
        "p_payload": payload(session, eligibility="screened_out",
                             selected_scenario=None, answers={},
                             screening=[{"scenario_key": k, "similarity": "not_really",
                                         "description": None}
                                        for k in session["screening_order"]])})
    assert result["ok"] is True
    row = service.get("/study_submissions?session_id=eq.%s&select=*"
                      % session["session_id"])[0]
    assert row["arm"] is None, "a screen-out never reached a generation"
    assert len(row["screening"]) == 4, "the screening funnel must survive"
    assert len(row["missing_keys"]) == len(questionnaires.all_response_keys())


def test_a_decliner_submits(anon, service, clean):
    session = start(anon)
    anon.rpc("submit_study", {"p_token": session["session_token"],
        "p_payload": payload(session, consent="decline", eligibility="declined",
                             selected_scenario=None, screening=[], answers={})})
    row = service.get("/study_submissions?session_id=eq.%s&select=arm,consent"
                      % session["session_id"])[0]
    assert row["consent"] == "decline" and row["arm"] is None


# --- 2. Randomization ------------------------------------------------------

def test_screening_order_is_a_permutation_and_varies(anon, service, clean):
    orders = set()
    for _ in range(12):
        session = start(anon)
        assert sorted(session["screening_order"]) == sorted(scenarios.scenario_keys())
        orders.add(tuple(session["screening_order"]))
    assert len(orders) > 1, "screening order never changed across 12 sessions"


def test_every_arm_is_reachable(anon, service, worker_token, clean):
    # 60 draws for 6 arms: missing one has probability 6*(5/6)^60 ≈ 1e-4, so a
    # failure here means an arm is genuinely unreachable, not bad luck.
    drawn = set()
    for _ in range(60):
        session = start(anon)
        request_generation(anon, session)
        drawn.add(arm_of(service, session))
        if drawn == set(ARMS):
            break
    assert drawn == set(ARMS), "only drew %s in 60 sessions" % sorted(drawn)


# --- 2b. The flow fork: which arms show the review page --------------------

def test_the_flow_answer_matches_the_arm_that_was_drawn(anon, service, clean):
    """The whole contract of response_flow, checked against the arm it drew.

    Two of the six arms drop the review-and-edit page. Loops until both answers
    have been observed, so this cannot pass by only ever seeing `true`.
    """
    seen = set()
    for _ in range(60):
        session = start(anon)
        flow = ask_flow(anon, session)
        arm = arm_of(service, session)
        assert arm is not None, "response_flow returned without drawing an arm"
        assert flow["show_review"] is (arm not in SKIP_REVIEW_ARMS), (
            "arm %s got show_review=%s" % (arm, flow["show_review"]))
        seen.add(flow["show_review"])
        if seen == {True, False}:
            break
    assert seen == {True, False}, "only ever saw show_review=%s" % sorted(seen)


def test_the_flow_call_leaks_nothing_but_the_bit(anon, clean):
    """It is the one call that tells the browser anything about its arm, so what
    it does NOT contain is the property worth pinning."""
    session = start(anon)
    flow = ask_flow(anon, session)
    assert set(flow) == {"show_review"}, sorted(flow)
    body = json.dumps(flow)
    for arm in ARMS:
        assert arm not in body, "arm %r leaked to the client" % arm


def test_the_flow_is_sticky_and_cannot_be_re_rolled(anon, service, clean):
    """The integrity rule the new call could have broken: a participant who
    presses Continue repeatedly must not be redrawing their condition."""
    session = start(anon)
    first = ask_flow(anon, session)
    arm = arm_of(service, session)
    for _ in range(8):
        assert ask_flow(anon, session) == first
        assert arm_of(service, session) == arm


def test_the_arm_the_flow_drew_is_the_arm_the_job_gets(anon, service, clean):
    """start_response must reuse the draw rather than making its own."""
    session = start(anon)
    ask_flow(anon, session)
    arm = arm_of(service, session)
    request_generation(anon, session)
    assert arm_of(service, session) == arm
    job = service.get("/study_jobs?session_id=eq.%s&select=arm"
                      % session["session_id"])[0]
    assert job["arm"] == arm


def test_asking_the_flow_creates_no_job(anon, service, clean):
    """It runs one page before the generation; someone who stops there must not
    leave a job for the worker to burn a GPU on."""
    session = start(anon)
    ask_flow(anon, session)
    assert service.get("/study_jobs?session_id=eq.%s&select=job_id"
                       % session["session_id"]) == []


def test_a_client_that_never_asks_the_flow_still_gets_an_arm(anon, service, clean):
    """A tampered or stale client can skip response_flow. start_response must
    still draw, or the job would carry a null arm."""
    session = start(anon)
    request_generation(anon, session)
    assert arm_of(service, session) in set(ARMS)


def test_the_flow_rejects_an_unknown_token(anon, clean):
    assert anon.try_rpc("response_flow", {"p_token": "made-up"}).status_code >= 400


# --- 2c. Walking a condition by hand (the /debug_<key> links) --------------
# The one sanctioned way to choose an arm. It exists because the flow arms are
# the only ones a person can verify by looking, and random assignment is exactly
# what stops you looking. Everything here is about the trade being contained:
# the arm is forced, the session is FLAGGED, and nothing else changes.

def test_a_debug_link_forces_exactly_the_condition_it_names(anon, service, clean):
    """The whole point: no restarting the study until the coin lands."""
    for key, expected in DEBUG_CONDITIONS.items():
        session = anon.rpc("start_session", {"p_user_agent": "pytest",
                                             "p_debug_condition": key})
        assert session["is_debug"] is True
        assert arm_of(service, session) == expected, (
            "condition %r did not force %r" % (key, expected))


def test_a_debug_session_walks_the_flow_its_arm_implies(anon, service, clean):
    """Forcing the arm is only useful if response_flow then agrees with it —
    otherwise the page under test is still decided by the draw."""
    for key, arm in DEBUG_CONDITIONS.items():
        session = anon.rpc("start_session", {"p_user_agent": "pytest",
                                             "p_debug_condition": key})
        flow = ask_flow(anon, session)
        assert flow["show_review"] is (arm not in SKIP_REVIEW_ARMS)
        assert arm_of(service, session) == arm, "asking the flow re-rolled a forced arm"


def test_a_forced_arm_survives_to_the_submission_row_with_its_flag(
        anon, service, worker_token, clean):
    """The containment guarantee, end to end. A hand-walked session must be
    identifiable in the exported data, or it counts as a participant in whichever
    cell it landed in — which is worse than the arm being forced in the first
    place."""
    key, arm = sorted(DEBUG_CONDITIONS.items())[0]
    session = anon.rpc("start_session", {"p_user_agent": "pytest",
                                         "p_debug_condition": key})
    request_generation(anon, session)
    finish_generation(anon, session, worker_token, text="A generated reply.")
    anon.rpc("submit_study", {"p_token": session["session_token"],
                              "p_payload": payload(session)})
    row = service.get("/study_submissions?session_id=eq.%s&select=arm,is_debug"
                      % session["session_id"])[0]
    assert row["arm"] == arm
    assert row["is_debug"] is True, "a forced arm reached the dataset unflagged"


def test_a_real_participant_is_never_flagged(anon, service, worker_token, clean):
    """The flag has to be a real discriminator: if an ordinary session set it,
    the export's default filter would throw away the study."""
    session = start(anon)
    assert session["is_debug"] is False
    request_generation(anon, session)
    finish_generation(anon, session, worker_token)
    anon.rpc("submit_study", {"p_token": session["session_token"],
                              "p_payload": payload(session)})
    assert service.get("/study_sessions?session_id=eq.%s&select=is_debug"
                       % session["session_id"])[0]["is_debug"] is False
    assert service.get("/study_submissions?session_id=eq.%s&select=is_debug"
                       % session["session_id"])[0]["is_debug"] is False


def test_a_debug_link_still_never_returns_the_arm(anon, clean):
    """It tells the browser THAT the condition was honoured, not which one it is.
    The key came from the URL; the arm name must stay in the database."""
    session = anon.rpc("start_session", {"p_user_agent": "pytest",
                                         "p_debug_condition": "c1"})
    body = json.dumps(session)
    for arm in ARMS:
        assert arm not in body, "arm %r leaked to the client" % (arm,)


def test_an_unknown_condition_is_refused_rather_than_randomized(anon, clean):
    """A mistyped link must fail loudly. Falling back to a normal draw would let
    someone believe they had seen a condition they never saw."""
    for bad in ("c9", "ours", "", "ours_noreview"):
        response = anon.try_rpc("start_session", {"p_user_agent": "pytest",
                                                  "p_debug_condition": bad})
        if bad == "":
            # Empty is the absence of a condition, not a bad one: a plain visit
            # with a stray `?debug=` must still start a normal session.
            assert response.status_code == 200 and response.json()["is_debug"] is False
        else:
            assert response.status_code >= 400, (
                "%r was accepted as a debug condition" % (bad,))


def test_an_ordinary_session_needs_no_debug_argument(anon, service, clean):
    """PostgREST resolves overloads by the keys in the body, so the argument
    being OPTIONAL is what keeps a participant's first call working."""
    session = anon.rpc("start_session", {"p_user_agent": "pytest"})
    assert session["is_debug"] is False
    assert arm_of(service, session) is None, "a session start must draw no arm"


# --- 3. The two integrity rules --------------------------------------------

def test_the_arm_is_never_returned_to_the_browser(anon, service, clean):
    """A client that could read its arm could re-roll until it liked one."""
    session = start(anon)
    body = json.dumps(request_generation(anon, session))
    for arm in ARMS:
        assert '"%s"' % arm not in body, "arm %r leaked to the client" % arm
    assert json.dumps(session).count("arm") == 0


def test_a_client_cannot_forge_its_arm_or_response(anon, service, worker_token, clean):
    session = start(anon)
    request_generation(anon, session)
    shown = finish_generation(anon, session, worker_token, text="THE REAL REPLY")
    actual = service.get("/study_sessions?session_id=eq.%s&select=arm"
                         % session["session_id"])[0]["arm"]

    anon.rpc("submit_study", {"p_token": session["session_token"],
        "p_payload": payload(session, arm="syco", shown_response="FORGED",
                             user_text="FORGED", stages=[{"name": "fake"}])})
    row = service.get("/study_submissions?session_id=eq.%s&select=*"
                      % session["session_id"])[0]
    assert row["arm"] == actual
    assert row["shown_response"] == shown and row["shown_response"] != "FORGED"
    assert row["user_text"] == DESC + "\n\n" + PERS
    assert row["stages"][0]["name"] == "generate"


def test_one_job_per_session_forever(anon, service, clean):
    """The one-way door. Re-posting different text must not regenerate."""
    session = start(anon)
    first = request_generation(anon, session)
    arm = service.get("/study_sessions?session_id=eq.%s&select=arm"
                      % session["session_id"])[0]["arm"]

    again = request_generation(
        anon, session,
        description="a completely different story about something else entirely",
        perspective="and a completely different perspective on it, written out here")
    assert again["job_id"] == first["job_id"]

    jobs = service.get("/study_jobs?session_id=eq.%s&select=user_text"
                       % session["session_id"])
    assert len(jobs) == 1
    assert jobs[0]["user_text"] == DESC + "\n\n" + PERS
    assert service.get("/study_sessions?session_id=eq.%s&select=arm"
                       % session["session_id"])[0]["arm"] == arm


def test_no_vignette_text_reaches_the_model(anon, service, clean):
    """The vignette is a canned example; prepending it once made the model
    answer about the wrong story entirely."""
    session = start(anon)
    request_generation(anon, session)
    user_text = service.get("/study_jobs?session_id=eq.%s&select=user_text"
                            % session["session_id"])[0]["user_text"]
    for scenario in scenarios.SCENARIOS:
        assert scenario["vignette"] not in user_text
        assert scenario["question"] not in user_text


def test_submitting_twice_stores_one_row(anon, service, worker_token, clean):
    session = start(anon)
    request_generation(anon, session)
    finish_generation(anon, session, worker_token)
    first = anon.rpc("submit_study", {"p_token": session["session_token"],
                                      "p_payload": payload(session)})
    second = anon.rpc("submit_study", {"p_token": session["session_token"],
                                       "p_payload": payload(session)})
    assert first["duplicate"] is False and second["duplicate"] is True
    assert len(service.get("/study_submissions?session_id=eq.%s&select=session_id"
                           % session["session_id"])) == 1


# --- 4. Validation ---------------------------------------------------------

@pytest.mark.parametrize("description,perspective,why", [
    ("tiny", "also tiny", "combined text far below the floor"),
    ("a", "b", "essentially empty"),
])
def test_short_free_text_is_rejected(anon, clean, description, perspective, why):
    """The floor is min_describe + min_perspective on the CONFIRMED text, so
    editing on the review page cannot be used to get under the requirement."""
    session = start(anon)
    response = anon.try_rpc("start_response", {
        "p_token": session["session_token"],
        "p_scenario_key": session["screening_order"][0],
        "p_user_text": description + "\n\n" + perspective})
    assert response.status_code >= 400, why


def test_the_longest_composition_the_client_can_produce_is_accepted(anon, service, clean):
    """The ceiling must clear the WORST CASE, not the per-field cap.

    On a no-review arm the participant never sees a page holding the composed
    text, so a rejection here is a dead end — nothing to shorten, no way
    forward. Two maxed-out fields joined by a blank line is exactly what the
    client can hand over.
    """
    import config
    description = "d" * config.MAX_TEXT_CHARS
    perspective = "p" * config.MAX_TEXT_CHARS
    session = start(anon)
    request_generation(anon, session, description=description, perspective=perspective)
    stored = service.get("/study_jobs?session_id=eq.%s&select=user_text"
                         % session["session_id"])[0]["user_text"]
    assert len(stored) == config.MAX_MODEL_INPUT_CHARS
    assert stored == description + "\n\n" + perspective, "the input was altered"


def test_a_composition_over_the_ceiling_is_still_rejected(anon, clean):
    """The cap is raised to the real worst case, not removed."""
    import config
    session = start(anon)
    response = anon.try_rpc("start_response", {
        "p_token": session["session_token"],
        "p_scenario_key": session["screening_order"][0],
        "p_user_text": "x" * (config.MAX_MODEL_INPUT_CHARS + 1)})
    assert response.status_code >= 400


def test_unknown_scenario_and_unknown_token_are_rejected(anon, clean):
    session = start(anon)
    assert anon.try_rpc("start_response", {
        "p_token": session["session_token"], "p_scenario_key": "not_a_scenario",
        "p_user_text": DESC + "\n\n" + PERS}).status_code >= 400
    assert anon.try_rpc("response_status",
                        {"p_token": "made-up"}).status_code >= 400


def test_an_out_of_domain_answer_is_rejected(anon, service, worker_token, clean):
    """A code no option declares must never enter the dataset."""
    session = start(anon)
    request_generation(anon, session)
    finish_generation(anon, session, worker_token)
    answers = all_answers()
    answers["fk_1"] = 999
    response = anon.try_rpc("submit_study", {"p_token": session["session_token"],
                                             "p_payload": payload(session, answers=answers)})
    assert response.status_code >= 400
    assert service.get("/study_submissions?session_id=eq.%s&select=session_id"
                       % session["session_id"]) == []


def test_a_missing_answer_is_recorded_not_rejected(anon, service, worker_token, clean):
    """Opposite of the above, on purpose: this runs after the participant has
    finished, and discarding a completed session over one blank cell is worse
    than recording the gap."""
    session = start(anon)
    request_generation(anon, session)
    finish_generation(anon, session, worker_token)
    answers = all_answers()
    del answers["fk_1"]
    result = anon.rpc("submit_study", {"p_token": session["session_token"],
                                       "p_payload": payload(session, answers=answers)})
    assert result["ok"] is True
    assert result["missing_keys"] == ["fk_1"]
    row = service.get("/study_submissions?session_id=eq.%s&select=missing_keys"
                      % session["session_id"])[0]
    assert row["missing_keys"] == ["fk_1"]


def test_skipped_qsf_choice_ids_are_accepted_and_gaps_are_not(anon, service,
                                                              worker_token, clean):
    """response_objectivity's options are 1,2,3,4,5,7,8 — 6 is not a valid code."""
    values = [o["value"] for q in questionnaires.all_questions()
              if q["id"] == "response_objectivity" for o in q["options"]]
    assert values == [1, 2, 3, 4, 5, 7, 8]

    session = start(anon)
    request_generation(anon, session)
    finish_generation(anon, session, worker_token)
    good = all_answers()
    good["response_objectivity"] = 8
    assert anon.rpc("submit_study", {"p_token": session["session_token"],
                                     "p_payload": payload(session, answers=good)})["ok"]

    other = start(anon)
    bad = all_answers()
    bad["response_objectivity"] = 6
    assert anon.try_rpc("submit_study", {
        "p_token": other["session_token"],
        "p_payload": payload(other, eligibility="screened_out",
                             selected_scenario=None, answers=bad)}).status_code >= 400


@pytest.mark.parametrize("bad", [
    {"consent": "maybe"},
    {"eligibility": "nonsense"},
    {"selected_scenario": "not_a_scenario"},
    {"eligibility": "eligible", "selected_scenario": None},
])
def test_malformed_payloads_are_rejected(anon, clean, bad):
    session = start(anon)
    assert anon.try_rpc("submit_study", {
        "p_token": session["session_token"],
        "p_payload": payload(session, **bad)}).status_code >= 400


# --- 5. The queue under concurrency ----------------------------------------

def test_concurrent_workers_never_double_claim(anon, service, worker_token, tag, clean):
    """The property the four-worker pool depends on.

    NOTE ON ISOLATION. This runs against the same database a live worker pool
    may be polling, so some of these jobs will be claimed by that pool rather
    than by this test's claimers — which is why the assertion is "no job was
    taken twice" rather than "this test took all of them". The weaker form is
    still the property that matters: FOR UPDATE SKIP LOCKED must hand each job
    to exactly one claimer, whoever is competing.
    """
    session_ids = []
    for _ in range(8):
        session = start(anon)
        request_generation(anon, session)
        session_ids.append(session["session_id"])

    def drain(index):
        taken = []
        while True:
            job = anon.rpc("worker_claim", {"p_worker_token": worker_token,
                                            "p_worker_id": "%s-w%d" % (tag, index)})["job"]
            if job is None:
                return taken
            taken.append(job["job_id"])

    with cf.ThreadPoolExecutor(6) as pool:
        claimed = [j for batch in pool.map(drain, range(6)) for j in batch]

    assert len(claimed) == len(set(claimed)), (
        "the same job was handed to two claimers: %s" % (claimed,))

    # Whoever took them, every job this test created must now be held by exactly
    # one worker — no job left pending and unclaimable, none held by two.
    rows = service.get("/study_jobs?session_id=in.(%s)&select=job_id,status,worker_id"
                       % ",".join(session_ids))
    assert len(rows) == 8
    for row in rows:
        assert row["status"] in ("running", "done"), row
        assert row["worker_id"], "claimed job with no worker_id: %s" % (row,)


def test_a_worker_that_lost_its_lease_cannot_overwrite_the_winner(
        anon, service, worker_token, clean):
    """The guard that stops a slow worker clobbering a result already produced.

    Set up by writing the job's state directly rather than racing to claim it:
    a live pool sharing this database will otherwise take it first, and the test
    would fail for a reason unrelated to the guard. What is under test is
    complete_job's worker_id check, and that is exercised either way.
    """
    import requests
    session = start(anon)
    request_generation(anon, session)
    job_id = service.get("/study_jobs?session_id=eq.%s&select=job_id"
                         % session["session_id"])[0]["job_id"]

    # 'slow' holds it, with a lease that has already expired.
    requests.patch("%s/study_jobs?job_id=eq.%s" % (service.base, job_id),
                   headers=service.headers, timeout=30,
                   json={"status": "running", "worker_id": "slow",
                         "lease_expires_at": "2000-01-01T00:00:00Z"})
    # Deliberately NOT asserting that the reap leaves it 'pending' here: it
    # does, and then a live worker pool re-claims it within its 2s poll, so the
    # intermediate state is unobservable on a shared database. What this test
    # is for is complete_job's worker_id check, below.
    anon.rpc("response_status", {"p_token": session["session_token"]})  # drives the reap

    requests.patch("%s/study_jobs?job_id=eq.%s" % (service.base, job_id),
                   headers=service.headers, timeout=30,
                   json={"status": "running", "worker_id": "fast",
                         "lease_expires_at": "2099-01-01T00:00:00Z"})

    assert anon.rpc("worker_complete", {
        "p_worker_token": worker_token, "p_job_id": job_id,
        "p_worker_id": "fast", "p_response_text": "winner",
        "p_stages": []})["accepted"] is True
    assert anon.rpc("worker_complete", {
        "p_worker_token": worker_token, "p_job_id": job_id,
        "p_worker_id": "slow", "p_response_text": "loser",
        "p_stages": []})["accepted"] is False
    assert service.get("/study_jobs?job_id=eq.%s&select=response_text"
                       % job_id)[0]["response_text"] == "winner"


def test_retry_is_capped(anon, service, worker_token, clean):
    import requests
    session = start(anon)
    request_generation(anon, session)
    # Written directly rather than claimed: a live pool would otherwise take the
    # job first. What is under test is the manual-retry cap, not who claimed it.
    job_id = service.get("/study_jobs?session_id=eq.%s&select=job_id"
                         % session["session_id"])[0]["job_id"]
    requests.patch("%s/study_jobs?job_id=eq.%s" % (service.base, job_id),
                   headers=service.headers, timeout=30, json={"status": "failed"})
    state = anon.rpc("response_status", {"p_token": session["session_token"]})
    assert state["status"] == "failed" and state["can_retry"] is True

    anon.rpc("retry_response", {"p_token": session["session_token"]})
    requests.patch("%s/study_jobs?job_id=eq.%s" % (service.base, job_id),
                   headers=service.headers, timeout=30, json={"status": "failed"})
    state = anon.rpc("response_status", {"p_token": session["session_token"]})
    assert state["can_retry"] is False, "manual retries must be capped"
    assert anon.try_rpc("retry_response",
                        {"p_token": session["session_token"]}).status_code >= 400


# --- 6. The permission boundary --------------------------------------------

@pytest.mark.parametrize("table", [
    "study_sessions", "study_jobs", "study_submissions",
    "study_config", "questionnaire_options", "study_scenarios",
])
def test_the_public_key_cannot_read_any_table(anon, table):
    """RLS with no policies. study_config is the sharp one — the worker secret
    lives in it, and it WAS readable until 2026-07-28."""
    response = anon.try_get("/%s?select=*&limit=1" % table)
    assert response.status_code in (401, 403) or response.json() == [], (
        "%s is readable with the public key: %s" % (table, response.text[:200]))


@pytest.mark.parametrize("name,args", [
    ("claim_job", {"p_worker_id": "x", "p_lease_seconds": 60}),
    ("complete_job", {"p_job_id": "x", "p_worker_id": "y",
                      "p_response_text": "z", "p_stages": []}),
    ("fail_job_attempt", {"p_job_id": "x", "p_worker_id": "y",
                          "p_error": "e", "p_max_attempts": 4}),
    ("extend_lease", {"p_job_id": "x", "p_worker_id": "y", "p_lease_seconds": 60}),
    ("requeue_job", {"p_job_id": "x", "p_deadline_seconds": 60}),
    ("mark_submitted", {"p_session_id": "x"}),
    ("reap_expired_jobs", {"p_max_attempts": 4}),
    ("queue_stats", {}),
    # The arm draw itself, which RETURNS the arm. Reachable from the browser it
    # would hand a participant their condition outright — response_flow is the
    # only way to it, and that answers with one bit.
    ("session_arm", {"p_session_id": "x"}),
])
def test_the_public_key_cannot_call_internal_functions(anon, name, args):
    """Postgres grants EXECUTE to PUBLIC by default and Supabase additionally
    grants to anon BY NAME, so these need explicit revokes from both. They did
    not have them until 2026-07-28."""
    assert anon.try_rpc(name, args).status_code >= 400, (
        "%s is callable with the public key" % name)


def test_worker_rpcs_require_the_shared_secret(anon):
    for name, args in (("worker_claim", {"p_worker_id": "x"}),
                       ("worker_heartbeat", {"p_job_id": "x", "p_worker_id": "y"}),
                       ("worker_complete", {"p_job_id": "x", "p_worker_id": "y",
                                            "p_response_text": "z", "p_stages": []}),
                       ("worker_fail", {"p_job_id": "x", "p_worker_id": "y",
                                        "p_error": "e"})):
        args = dict(args, p_worker_token="not-the-secret")
        assert anon.try_rpc(name, args).status_code >= 400, (
            "%s accepted a wrong worker token" % name)


def test_the_worker_secret_is_not_readable(anon):
    response = anon.try_get("/study_config?key=eq.worker_token&select=value")
    leaked = response.status_code == 200 and response.json() not in ([], None)
    assert not leaked, "the worker token is readable with the public key"
