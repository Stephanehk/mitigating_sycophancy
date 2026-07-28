"""Supabase-backed store: the production datastore for sessions and the queue.

Speaks PostgREST over HTTPS with the service-role key, NOT psycopg. Three
reasons, all of which matter behind Lambda:

  * No connection pool. A Lambda fleet opening Postgres connections exhausts
    the server's slots; HTTPS requests do not have that failure mode.
  * No database password. The service-role key we already hold is enough.
  * Cold starts stay cheap — no driver, no TLS handshake to a Postgres port,
    no pool warm-up.

The four operations that must be atomic against concurrent workers are plpgsql
functions rather than table writes, because PostgREST cannot express
`FOR UPDATE SKIP LOCKED` or a guarded `UPDATE ... RETURNING`. See
supabase_queue_schema.sql, which must be applied before this module works.

CONTRACT. Every function here mirrors store_sqlite.py exactly, including return
shapes, so server.py cannot tell which backend it is talking to. Two places
need active translation:

  * `stages` is a real jsonb column here but a TEXT column named `stages_json`
    in SQLite. Job dicts are normalized to carry `stages_json` as a JSON string,
    matching what server.py already parses.
  * Postgres returns timestamptz as "...+00:00"; SQLite returns "...Z". Callers
    only ever pass these through to the export, which parses both.
"""

import json

import requests

import config
from store_common import in_seconds, new_id, utcnow  # noqa: F401  (re-exported)

#: Long enough to ride out a slow Supabase response, short enough that a
#: participant's request cannot hang past their patience.
_TIMEOUT = 20


def _base():
    assert config.SUPABASE_URL, "STUDY_SUPABASE_URL is not set"
    assert config.SUPABASE_SERVICE_KEY, "STUDY_SUPABASE_SERVICE_KEY is not set"
    return config.SUPABASE_URL.rstrip("/") + "/rest/v1"


def _headers(extra=None):
    headers = {
        "apikey": config.SUPABASE_SERVICE_KEY,
        "Authorization": "Bearer " + config.SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _request(method, path, **kwargs):
    """One HTTP call to PostgREST. Raises on anything but success.

    Fail-fast by design: a 4xx here means a schema mismatch or a bad key, both
    of which are programming/config errors that must not be papered over.
    """
    response = requests.request(method, _base() + path, headers=_headers(kwargs.pop("headers", None)),
                                timeout=_TIMEOUT, **kwargs)
    assert response.status_code in (200, 201, 204), (
        "supabase %s %s -> %d %s" % (method, path, response.status_code, response.text[:400]))
    return response.json() if response.text.strip() else None


def _rpc(name, payload):
    """Call a plpgsql function via POST /rest/v1/rpc/<name>."""
    return _request("POST", "/rpc/" + name, json=payload)


def _one(rows):
    """First row of a PostgREST list response, or None."""
    if not rows:
        return None
    return rows[0]


def _normalize_job(row):
    """Present a job row in the shape store_sqlite.py returns.

    The only real difference is `stages` (jsonb) versus `stages_json` (TEXT).
    server.py does `json.loads(job["stages_json"])`, so serializing here keeps
    that call site backend-agnostic rather than making it branch.
    """
    if row is None:
        return None
    job = dict(row)
    stages = job.pop("stages", None)
    job["stages_json"] = None if stages is None else json.dumps(stages)
    return job


def init_db():
    """Verify the schema is reachable. The tables themselves are created by
    supabase_queue_schema.sql, run once by hand in the SQL editor — a Lambda
    must never try to run DDL on cold start."""
    _request("GET", "/study_jobs?select=job_id&limit=1")


# --- Sessions --------------------------------------------------------------

def create_session(screening_order, user_agent):
    """Insert a new session. Returns (session_id, session_token)."""
    assert screening_order, "screening_order must be non-empty"
    session_id, token = new_id(), new_id()
    _request("POST", "/study_sessions", json={
        "session_id": session_id, "session_token": token,
        "screening_order": screening_order, "user_agent": user_agent})
    return session_id, token


def get_session(session_id):
    """Fetch one session row as a dict, or None."""
    return _one(_request("GET", "/study_sessions?session_id=eq.%s&select=*" % session_id))


def get_session_by_token(token):
    """Fetch a session by its browser-held token, or None.

    The authenticated lookup behind every participant request. Returns None
    rather than raising on an unknown token, because that is an ordinary
    "browser lost its session" case, not an error.
    """
    if not token:
        return None
    return _one(_request("GET", "/study_sessions?session_token=eq.%s&select=*"
                         % requests.utils.quote(str(token), safe="")))


def update_session(session_id, **fields):
    """Patch named columns on a session and bump `updated_at`.

    Same short allowlist as the SQLite backend: the only mutable server-side
    facts are the assigned arm and whether the payload has been submitted.
    """
    allowed = {"arm", "submitted_at"}
    unknown = set(fields) - allowed
    assert not unknown, "update_session got unknown columns: %s" % (sorted(unknown),)
    assert fields, "update_session called with no fields"
    payload = dict(fields)
    payload["updated_at"] = utcnow()
    rows = _request("PATCH", "/study_sessions?session_id=eq.%s" % session_id,
                    json=payload, headers={"Prefer": "return=representation"})
    assert rows, "no session updated for id %r" % (session_id,)


def mark_submitted(session_id):
    """Record that the final payload was accepted. True the first time only."""
    return bool(_rpc("mark_submitted", {"p_session_id": session_id}))


def arm_counts():
    """Submitted-session count per arm, as a dict. Used for monitoring."""
    return _rpc("queue_stats", {})["submitted_by_arm"]


def unsubmitted_sessions_with_done_jobs():
    """Sessions that saw a response but never submitted.

    The completers-only design's blind spot, made visible: each of these is a
    participant whose answers exist nowhere.
    """
    return _rpc("queue_stats", {})["saw_response_never_submitted"]


# --- Job queue -------------------------------------------------------------

def create_job(session_id, arm, topic_title, user_text):
    """Enqueue a generation job and return its id."""
    job_id = new_id()
    _request("POST", "/study_jobs", json={
        "job_id": job_id, "session_id": session_id, "arm": arm,
        "topic_title": topic_title, "user_text": user_text, "status": "pending",
        "deadline_at": in_seconds(config.JOB_DEADLINE_SECONDS)})
    return job_id


def get_job(job_id):
    """Fetch one job row as a dict, or None."""
    return _normalize_job(_one(_request("GET", "/study_jobs?job_id=eq.%s&select=*" % job_id)))


def latest_job_for_session(session_id):
    """The most recently created job for a session, or None.

    Used on resume: a participant who refreshes the loading page must be
    reattached to their in-flight job rather than triggering a second one.
    """
    return _normalize_job(_one(_request(
        "GET", "/study_jobs?session_id=eq.%s&select=*&order=created_at.desc&limit=1" % session_id)))


def has_pending_job():
    """Whether any job is waiting to be claimed. READ-ONLY advisory check."""
    return bool(_request("GET", "/study_jobs?status=eq.pending&select=job_id&limit=1"))


def has_reapable_jobs():
    """Whether reap_expired_jobs would change a row. READ-ONLY advisory check.

    Kept for contract parity with the SQLite backend, where it exists to avoid
    taking a write lock. Here it saves a round trip rather than a lock, so it is
    a smaller win — reap_expired_jobs is cheap and safe to call unconditionally.
    """
    now = utcnow()
    query = ("/study_jobs?select=job_id&limit=1&or=("
             "and(status.eq.running,lease_expires_at.lt.%s),"
             "and(status.in.(pending,running),attempts.gte.%d),"
             "and(status.in.(pending,running),deadline_at.lt.%s))"
             % (now, config.JOB_MAX_ATTEMPTS, now))
    return bool(_request("GET", query))


def reap_expired_jobs():
    """Return timed-out jobs to the queue, fail jobs that are out of road.

    Returns (requeued_count, failed_count). Safe to call on every poll.
    """
    row = _one(_rpc("reap_expired_jobs", {"p_max_attempts": config.JOB_MAX_ATTEMPTS}))
    if row is None:
        return 0, 0
    return int(row["requeued"]), int(row["failed"])


def claim_job(worker_id):
    """Atomically claim the oldest pending job, or return None.

    Serialized by `FOR UPDATE SKIP LOCKED` inside the function: two workers
    polling simultaneously take DIFFERENT rows rather than blocking or
    double-claiming. Verified against the live database with 8 concurrent
    claimers over 12 jobs, zero duplicates.
    """
    return _normalize_job(_one(_rpc("claim_job", {
        "p_worker_id": worker_id, "p_lease_seconds": config.JOB_LEASE_SECONDS})))


def extend_lease(job_id, worker_id):
    """Push a running job's lease out. Only the holder may extend."""
    return bool(_rpc("extend_lease", {
        "p_job_id": job_id, "p_worker_id": worker_id,
        "p_lease_seconds": config.JOB_LEASE_SECONDS}))


def complete_job(job_id, worker_id, response_text, stages):
    """Mark a job done with its final text. True if this worker won the race."""
    assert response_text is not None and response_text.strip(), (
        "refusing to complete job %r with empty response text" % (job_id,))
    return bool(_rpc("complete_job", {
        "p_job_id": job_id, "p_worker_id": worker_id,
        "p_response_text": response_text, "p_stages": stages}))


def fail_job_attempt(job_id, worker_id, error):
    """Record a failed attempt. Returns the resulting status, or None if this
    worker no longer held the job."""
    return _rpc("fail_job_attempt", {
        "p_job_id": job_id, "p_worker_id": worker_id, "p_error": str(error)[:2000],
        "p_max_attempts": config.JOB_MAX_ATTEMPTS})


def requeue_job(job_id):
    """Put a terminally failed job back in the queue with a fresh budget."""
    return bool(_rpc("requeue_job", {
        "p_job_id": job_id, "p_deadline_seconds": config.JOB_DEADLINE_SECONDS}))


def queue_depth():
    """Count of jobs per status, as a dict. For the ops endpoint."""
    return _rpc("queue_stats", {})["queue"]
