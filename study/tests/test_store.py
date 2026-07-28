"""Contract tests for the datastore, run against BOTH backends.

The answer tables are gone (answers live in the browser then Supabase), so what
is left to test is the queue's concurrency behaviour — leases, double-claims,
retries — plus the two session facts the server still owns.

Every test here runs twice: once against store_sqlite (a temp file) and once
against store_supabase (the live project), because the two implement the same
22-function contract and nothing else would catch them drifting apart. The
Supabase pass is skipped when credentials are absent, so the suite still runs
offline — but it is the pass that matters, since Supabase is what production
uses.

`force_job` exists because several tests need to put a job into a state the API
cannot produce (an expired lease, a passed deadline). That is raw SQL on one
backend and a PATCH on the other, so it is abstracted rather than inlined.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import store


import supabase_client


def _supabase_available():
    return supabase_client.is_configured()


@pytest.fixture(params=["sqlite", "supabase"])
def db(request, monkeypatch):
    """A clean datastore on each backend in turn.

    SQLite gets a throwaway file. Supabase gets the real project, with every
    row this test creates deleted afterwards — identified by diffing the
    session ids present before and after, since ids are generated inside the
    code under test and cannot be predicted here.
    """
    if request.param == "sqlite":
        handle, path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.unlink(path)
        monkeypatch.setattr(config, "DATABASE_URL", "sqlite:///" + path)
        store.init_db()
        yield path
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(path + suffix):
                os.unlink(path + suffix)
        return

    if not _supabase_available():
        pytest.skip("Supabase not configured; set STUDY_SUPABASE_URL/_SERVICE_KEY")
    monkeypatch.setattr(config, "DATABASE_URL", "supabase://")
    import store_supabase
    before = {row["session_id"] for row in
              store_supabase._request("GET", "/study_sessions?select=session_id")}
    yield "supabase"
    after = {row["session_id"] for row in
             store_supabase._request("GET", "/study_sessions?select=session_id")}
    for session_id in after - before:
        # jobs cascade on the session's foreign key.
        store_supabase._request("DELETE", "/study_sessions?session_id=eq.%s" % session_id)


def force_job(job_id, **fields):
    """Force job columns into a state the public API cannot produce.

    Needed to simulate an expired lease or a passed deadline without waiting
    out a real timeout. Backend-specific by necessity; the tests above it stay
    backend-agnostic.
    """
    if store.backend_name() == "sqlite":
        assignments = ", ".join("%s = ?" % k for k in fields)
        with store.transaction() as conn:
            conn.execute("UPDATE jobs SET %s WHERE job_id = ?" % assignments,
                         list(fields.values()) + [job_id])
        return
    import store_supabase
    store_supabase._request("PATCH", "/study_jobs?job_id=eq.%s" % job_id, json=fields)


def make_session(order=None):
    session_id, token = store.create_session(order or ["a", "b"], "pytest")
    return session_id, token


def make_job(session_id, arm="ours"):
    return store.create_job(session_id, arm, "Topic", "the user's story")


# --- Sessions --------------------------------------------------------------

def test_a_new_session_has_no_arm_and_is_unsubmitted(db):
    session_id, token = make_session()
    session = store.get_session_by_token(token)
    assert session["session_id"] == session_id
    assert session["arm"] is None
    assert session["submitted_at"] is None


def test_update_session_refuses_columns_outside_its_allowlist(db):
    session_id, _ = make_session()
    store.update_session(session_id, arm="syco")
    assert store.get_session(session_id)["arm"] == "syco"
    with pytest.raises(AssertionError):
        store.update_session(session_id, perspective_text="should not exist")


def test_mark_submitted_succeeds_once_and_then_refuses(db):
    """The idempotency guard behind /api/submit."""
    session_id, _ = make_session()
    assert store.mark_submitted(session_id) is True
    assert store.mark_submitted(session_id) is False
    assert store.get_session(session_id)["submitted_at"] is not None


def test_arm_counts_only_counts_submitted_sessions(db):
    first, _ = make_session()
    second, _ = make_session()
    store.update_session(first, arm="ours")
    store.update_session(second, arm="ours")
    store.mark_submitted(first)
    assert store.arm_counts() == {"ours": 1}


def test_unsubmitted_sessions_with_done_jobs_counts_the_blind_spot(db):
    session_id, _ = make_session()
    job_id = make_job(session_id)
    store.claim_job("w")
    store.complete_job(job_id, "w", "text", [])
    assert store.unsubmitted_sessions_with_done_jobs() == 1
    store.mark_submitted(session_id)
    assert store.unsubmitted_sessions_with_done_jobs() == 0


# --- Queue -----------------------------------------------------------------

def test_claiming_moves_a_job_to_running_and_burns_an_attempt(db):
    session_id, _ = make_session()
    make_job(session_id)
    job = store.claim_job("worker-1")
    assert job["status"] == "running" and job["attempts"] == 1
    assert store.claim_job("worker-2") is None


def test_two_workers_cannot_claim_the_same_job(db):
    session_id, _ = make_session()
    make_job(session_id)
    first = store.claim_job("worker-1")
    second = store.claim_job("worker-2")
    assert first is not None and second is None


def test_completing_records_the_text_and_the_stages(db):
    session_id, _ = make_session()
    job_id = make_job(session_id)
    store.claim_job("w")
    assert store.complete_job(job_id, "w", "final", [{"name": "generate", "output": "final"}])
    job = store.get_job(job_id)
    assert job["status"] == "done" and job["response_text"] == "final"
    assert '"generate"' in job["stages_json"]


def test_a_worker_that_lost_its_lease_cannot_overwrite_the_winner(db):
    session_id, _ = make_session()
    job_id = make_job(session_id)
    store.claim_job("slow")
    force_job(job_id, lease_expires_at="2000-01-01T00:00:00Z")
    store.reap_expired_jobs()
    store.claim_job("fast")
    assert store.complete_job(job_id, "fast", "winner", []) is True
    assert store.complete_job(job_id, "slow", "loser", []) is False
    assert store.get_job(job_id)["response_text"] == "winner"


def test_an_expired_lease_returns_the_job_to_the_queue(db):
    session_id, _ = make_session()
    job_id = make_job(session_id)
    store.claim_job("gone")
    force_job(job_id, lease_expires_at="2000-01-01T00:00:00Z")
    requeued, failed = store.reap_expired_jobs()
    assert (requeued, failed) == (1, 0)
    assert store.get_job(job_id)["status"] == "pending"


def test_a_passed_deadline_fails_the_job_so_nobody_spins_forever(db):
    session_id, _ = make_session()
    job_id = make_job(session_id)
    force_job(job_id, deadline_at="2000-01-01T00:00:00Z")
    requeued, failed = store.reap_expired_jobs()
    assert (requeued, failed) == (0, 1)
    assert store.get_job(job_id)["status"] == "failed"


def test_exhausting_attempts_makes_the_failure_terminal(db, monkeypatch):
    monkeypatch.setattr(config, "JOB_MAX_ATTEMPTS", 2)
    session_id, _ = make_session()
    job_id = make_job(session_id)
    store.claim_job("w")
    assert store.fail_job_attempt(job_id, "w", "boom") == "pending"
    store.claim_job("w")
    assert store.fail_job_attempt(job_id, "w", "boom") == "failed"


def test_requeue_resets_attempts_but_counts_the_manual_retry(db):
    """manual_retries is the only durable record that the button was pressed."""
    session_id, _ = make_session()
    job_id = make_job(session_id)
    store.claim_job("w")
    store.fail_job_attempt(job_id, "w", "boom")
    force_job(job_id, status="failed")

    assert store.requeue_job(job_id) is True
    job = store.get_job(job_id)
    assert job["status"] == "pending"
    assert job["attempts"] == 0
    assert job["manual_retries"] == 1

    force_job(job_id, status="failed")
    store.requeue_job(job_id)
    assert store.get_job(job_id)["manual_retries"] == 2


def test_requeue_refuses_a_job_that_is_not_failed(db):
    session_id, _ = make_session()
    job_id = make_job(session_id)
    assert store.requeue_job(job_id) is False


def test_completing_with_empty_text_is_a_programming_error(db):
    session_id, _ = make_session()
    job_id = make_job(session_id)
    store.claim_job("w")
    with pytest.raises(AssertionError):
        store.complete_job(job_id, "w", "   ", [])


def test_latest_job_for_session_is_the_most_recent(db):
    session_id, _ = make_session()
    first = make_job(session_id)
    second = make_job(session_id)
    assert store.latest_job_for_session(session_id)["job_id"] == second
    assert first != second


def test_advisory_precheck_matches_the_reaper(db):
    """has_reapable_jobs exists to skip the write lock; it must not lie."""
    session_id, _ = make_session()
    job_id = make_job(session_id)
    assert store.has_reapable_jobs() is False
    force_job(job_id, deadline_at="2000-01-01T00:00:00Z")
    assert store.has_reapable_jobs() is True
