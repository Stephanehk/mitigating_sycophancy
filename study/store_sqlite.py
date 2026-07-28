"""SQLite backend for the datastore: local development and the test suite.

One of two backends behind store.py (the other is store_supabase.py). Selected
whenever STUDY_DATABASE_URL is not a supabase:// or postgres:// URL.

SINGLE-PROCESS ONLY. This cannot back the production Lambda deployment, where
invocations share no filesystem — that is what store_supabase.py is for.

This module is the ONLY place that talks SQL. It holds two tables — thin
`sessions` rows and the `jobs` queue the GPU worker polls (see schema.sql).

IT DOES NOT STORE PARTICIPANT ANSWERS. Those live in the browser for the
duration of the session and go to Supabase in one payload at the end; see
supabase_client.py. That is a deliberate change: answers used to be persisted
per page, which put a synchronous write on the critical path of every Continue
press, and on this cluster's NFS mount that cost ~213ms typical and produced
sporadic 10-14 second stalls under worker-poll contention.

Concurrency model. SQLite in WAL mode with a short busy timeout. The one
genuinely contended operation is `claim_job`, where two workers could otherwise
take the same job; it is serialized with an IMMEDIATE transaction and re-checks
the row's status inside that transaction, so a double-claim is impossible
rather than merely unlikely.

Timestamps are ISO-8601 UTC strings with a 'Z' suffix. They sort
lexicographically in the same order they sort chronologically, which is what
lets the lease and deadline comparisons happen in SQL.
"""

import contextlib
import json
import os
import sqlite3
import threading

import config

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


from store_common import in_seconds, new_id, utcnow  # noqa: F401


def _sqlite_path(database_url):
    """Extract the filesystem path from a sqlite:/// URL.

    Assumption: only the three-slash (absolute path) form is used, which is
    what config.DATABASE_URL builds. Anything else is a configuration error.
    """
    prefix = "sqlite:///"
    assert database_url.startswith(prefix), (
        "store.py currently supports only sqlite:/// URLs, got %r" % (database_url,))
    return database_url[len(prefix):]


_keepalive_lock = threading.Lock()
_keepalive = None


def _ensure_keepalive(path):
    """Hold one connection to `path` open for the life of the process.

    Not an optimization — a correctness-adjacent fix for WAL mode. When the LAST
    connection to a WAL database closes, SQLite checkpoints the entire log and
    unlinks the -wal/-shm files. Because every function here opens a connection
    and closes it again, that "last close" happened on essentially every call,
    so each read paid for a full checkpoint plus file creation and unlink. On the
    NFS-backed trial database that showed up as participant requests stalling for
    seconds whenever the worker was polling (73ms median, 17s worst case).

    Keeping one idle connection open means the WAL is never torn down and
    checkpointing follows the normal autocheckpoint schedule. The connection is
    never used for queries, only held, so it pins no read snapshot and blocks no
    checkpoint.

    Re-opened when `path` changes, which is what lets each test point
    config.DATABASE_URL at its own temporary file.
    """
    global _keepalive
    with _keepalive_lock:
        if _keepalive is not None and _keepalive[0] == path:
            return
        if _keepalive is not None:
            _keepalive[1].close()
            _keepalive = None
        # check_same_thread=False because this is created by whichever thread
        # happens to connect first and merely held open afterwards.
        conn = sqlite3.connect(path, timeout=15.0, isolation_level=None,
                               check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        _keepalive = (path, conn)


def connect():
    """Open a connection to the configured database, creating its directory.

    Row factory is sqlite3.Row so callers index by column name. The busy
    timeout lets a concurrent writer wait rather than immediately raising
    "database is locked".
    """
    path = _sqlite_path(config.DATABASE_URL)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    _ensure_keepalive(path)
    conn = sqlite3.connect(path, timeout=15.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    # NORMAL rather than the FULL default: with WAL, NORMAL still gives
    # atomic, crash-safe commits and only risks the very last transactions on
    # an OS/host crash (not on a process crash). It matters here because the
    # trial database lives on NFS, where each FULL commit is a synchronous
    # network round-trip — that cost is what turned worker poll contention into
    # multi-second participant stalls.
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextlib.contextmanager
def transaction(immediate=False):
    """Run a block inside one transaction, committing on success.

    `immediate=True` takes the write lock up front (BEGIN IMMEDIATE), which is
    what `claim_job` needs to serialize competing workers. The connection is
    closed on the way out either way; on an exception the transaction rolls
    back and the exception propagates, per the repo's fail-fast policy.
    """
    conn = connect()
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        conn.close()
        raise
    conn.commit()
    conn.close()


def init_db():
    """Create the schema if it does not exist. Idempotent."""
    with open(_SCHEMA_PATH) as handle:
        script = handle.read()
    conn = connect()
    conn.executescript(script)
    conn.close()


# --- Sessions --------------------------------------------------------------

def create_session(screening_order, user_agent):
    """Insert a new session.

    The row carries no stage and no answers — the browser owns both. It exists
    so the generation job has something to hang off and so the randomized
    screening order is recorded server-side.

    Inputs: `screening_order` — list of scenario keys in presentation order;
            `user_agent` — the browser UA string, for debugging only.
    Returns (session_id, session_token).
    """
    assert screening_order, "screening_order must be non-empty"
    session_id = new_id()
    token = new_id()
    now = utcnow()
    with transaction() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, session_token, screening_order,"
            " user_agent, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, token, json.dumps(screening_order), user_agent, now, now))
    return session_id, token


def get_session(session_id):
    """Fetch one session row as a dict, or None if it does not exist."""
    conn = connect()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row is not None else None


def get_session_by_token(token):
    """Fetch a session by its browser-held token, or None.

    This is the authenticated lookup used by every participant-facing request.
    """
    conn = connect()
    row = conn.execute("SELECT * FROM sessions WHERE session_token = ?", (token,)).fetchone()
    conn.close()
    return dict(row) if row is not None else None


def update_session(session_id, **fields):
    """Patch named columns on a session and bump `updated_at`.

    Column names are validated against an allowlist rather than interpolated
    blind, so a caller typo cannot become SQL. The allowlist is short by design:
    the only mutable server-side facts left are the assigned arm and whether the
    final payload has been submitted.
    """
    allowed = {"arm", "submitted_at"}
    unknown = set(fields) - allowed
    assert not unknown, "update_session got unknown columns: %s" % (sorted(unknown),)
    assert fields, "update_session called with no fields"
    assignments = ", ".join("%s = ?" % name for name in fields)
    values = list(fields.values()) + [utcnow(), session_id]
    with transaction() as conn:
        cursor = conn.execute(
            "UPDATE sessions SET %s, updated_at = ? WHERE session_id = ?" % assignments, values)
        assert cursor.rowcount == 1, "no session updated for id %r" % (session_id,)


def mark_submitted(session_id):
    """Record that this session's final payload was accepted. Returns True once.

    The UPDATE is guarded on `submitted_at IS NULL`, so a duplicate submit (a
    double-click, or a client retry after a slow response) matches zero rows and
    the caller learns it was already done rather than writing Supabase twice.
    """
    with transaction(immediate=True) as conn:
        cursor = conn.execute(
            "UPDATE sessions SET submitted_at = ?, updated_at = ?"
            " WHERE session_id = ? AND submitted_at IS NULL",
            (utcnow(), utcnow(), session_id))
        return cursor.rowcount == 1


def arm_counts():
    """Submitted-session count per arm, as a dict. Used for monitoring."""
    conn = connect()
    rows = conn.execute(
        "SELECT arm, COUNT(*) AS n FROM sessions"
        " WHERE arm IS NOT NULL AND submitted_at IS NOT NULL GROUP BY arm").fetchall()
    conn.close()
    return {row["arm"]: row["n"] for row in rows}


# --- Job queue -------------------------------------------------------------

def create_job(session_id, arm, topic_title, user_text):
    """Enqueue a generation job and return its id."""
    job_id = new_id()
    now = utcnow()
    with transaction() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, session_id, arm, topic_title, user_text, status,"
            " attempts, deadline_at, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)",
            (job_id, session_id, arm, topic_title, user_text,
             in_seconds(config.JOB_DEADLINE_SECONDS), now, now))
    return job_id


def get_job(job_id):
    """Fetch one job row as a dict, or None."""
    conn = connect()
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row is not None else None


def latest_job_for_session(session_id):
    """The most recently created job for a session, or None.

    Used on resume: a participant who refreshes the loading page must be
    reattached to their in-flight job rather than triggering a second one.
    """
    conn = connect()
    row = conn.execute(
        "SELECT * FROM jobs WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (session_id,)).fetchone()
    conn.close()
    return dict(row) if row is not None else None


def _expiry_predicate(now):
    """SQL and params matching every job reap_expired_jobs would change."""
    return ("(status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?)"
            " OR (status IN ('pending', 'running') AND (attempts >= ? OR deadline_at < ?))",
            (now, config.JOB_MAX_ATTEMPTS, now))


def has_reapable_jobs():
    """Whether reap_expired_jobs would actually change a row. READ-ONLY.

    A cheap advisory pre-check so the common case — nothing to reap — costs one
    read instead of a BEGIN IMMEDIATE write lock. It is called on every worker
    poll tick and every participant status poll, so on a network filesystem the
    difference is the whole ballgame.

    Racy by design: a job can expire immediately after this returns False. That
    is harmless because the next poll (at most a second later) catches it, and
    because the authoritative check still happens inside reap's transaction.
    """
    where, params = _expiry_predicate(utcnow())
    conn = connect()
    row = conn.execute("SELECT 1 FROM jobs WHERE %s LIMIT 1" % where, params).fetchone()
    conn.close()
    return row is not None


def has_pending_job():
    """Whether any job is waiting to be claimed. READ-ONLY.

    The same advisory-pre-check trick as has_reapable_jobs, for claim_job: an
    idle worker long-polling an empty queue should not be taking the write lock
    twice a second. claim_job re-checks inside its transaction, so a job that
    arrives just after this returns False is simply picked up on the next tick.
    """
    conn = connect()
    row = conn.execute("SELECT 1 FROM jobs WHERE status = 'pending' LIMIT 1").fetchone()
    conn.close()
    return row is not None


def reap_expired_jobs():
    """Return timed-out jobs to the queue, and fail jobs that are out of road.

    Two distinct expiries, both checked here so every call site gets the same
    semantics:
      - lease expiry: a 'running' job whose worker went away becomes 'pending'
        again so another worker can pick it up.
      - deadline / attempt exhaustion: a job that has burned all its attempts
        or passed its wall-clock deadline becomes 'failed', which is what
        releases the participant from the loading screen.

    Returns (requeued_count, failed_count). Safe to call on every poll — and
    cheap to, because it short-circuits on a read when there is nothing to do.
    """
    if not has_reapable_jobs():
        return 0, 0
    now = utcnow()
    with transaction(immediate=True) as conn:
        requeued = conn.execute(
            "UPDATE jobs SET status = 'pending', worker_id = NULL, lease_expires_at = NULL,"
            " updated_at = ?"
            " WHERE status = 'running' AND lease_expires_at IS NOT NULL"
            " AND lease_expires_at < ?", (now, now)).rowcount
        failed = conn.execute(
            "UPDATE jobs SET status = 'failed', updated_at = ?, completed_at = ?,"
            " error = COALESCE(error, 'exhausted: attempts or deadline')"
            " WHERE status IN ('pending', 'running')"
            " AND (attempts >= ? OR deadline_at < ?)",
            (now, now, config.JOB_MAX_ATTEMPTS, now)).rowcount
    return requeued, failed


def claim_job(worker_id):
    """Atomically claim the oldest pending job for `worker_id`, or return None.

    Serialized with BEGIN IMMEDIATE so two workers polling simultaneously
    cannot both take the same row: the second one blocks until the first
    commits, then sees status='running' and selects a different job.

    Increments `attempts` at claim time, not at completion, so a worker that
    crashes mid-generation still burns an attempt. Without that, a job that
    reliably kills its worker would be retried forever.

    Short-circuits on a read when the queue is empty, so an idle long-poll does
    not hold the write lock (see has_pending_job).
    """
    if not has_pending_job():
        return None
    with transaction(immediate=True) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at, rowid LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE jobs SET status = 'running', worker_id = ?, attempts = attempts + 1,"
            " lease_expires_at = ?, updated_at = ? WHERE job_id = ?",
            (worker_id, in_seconds(config.JOB_LEASE_SECONDS), utcnow(), row["job_id"]))
        claimed = dict(row)
    claimed["status"] = "running"
    claimed["attempts"] = row["attempts"] + 1
    return claimed


def extend_lease(job_id, worker_id):
    """Push a running job's lease out by another JOB_LEASE_SECONDS.

    Lets a worker hold a genuinely slow generation without the reaper stealing
    it. Only the lease holder may extend, so a worker that already lost the
    job to expiry cannot reclaim it behind the new holder's back.
    Returns True if the lease was extended.
    """
    with transaction(immediate=True) as conn:
        cursor = conn.execute(
            "UPDATE jobs SET lease_expires_at = ?, updated_at = ?"
            " WHERE job_id = ? AND worker_id = ? AND status = 'running'",
            (in_seconds(config.JOB_LEASE_SECONDS), utcnow(), job_id, worker_id))
        return cursor.rowcount == 1


def complete_job(job_id, worker_id, response_text, stages):
    """Mark a job done with its final text. Returns True if this worker won.

    The `worker_id` guard means a worker whose lease expired mid-generation
    cannot overwrite a result that a second worker already produced: its write
    matches zero rows and it learns it lost.
    """
    assert response_text is not None and response_text.strip(), (
        "refusing to complete job %r with empty response text" % (job_id,))
    now = utcnow()
    with transaction(immediate=True) as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'done', response_text = ?, stages_json = ?,"
            " error = NULL, updated_at = ?, completed_at = ?, lease_expires_at = NULL"
            " WHERE job_id = ? AND worker_id = ? AND status = 'running'",
            (response_text, json.dumps(stages), now, now, job_id, worker_id))
        return cursor.rowcount == 1


def fail_job_attempt(job_id, worker_id, error):
    """Record a failed attempt, returning the job to the queue or failing it.

    A job goes back to 'pending' if it has attempts left and time on its
    deadline; otherwise it is terminal 'failed'. Returns the resulting status,
    or None if this worker no longer held the job.
    """
    now = utcnow()
    with transaction(immediate=True) as conn:
        row = conn.execute(
            "SELECT attempts, deadline_at FROM jobs"
            " WHERE job_id = ? AND worker_id = ? AND status = 'running'",
            (job_id, worker_id)).fetchone()
        if row is None:
            return None
        exhausted = row["attempts"] >= config.JOB_MAX_ATTEMPTS or row["deadline_at"] < now
        status = "failed" if exhausted else "pending"
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, updated_at = ?,"
            " completed_at = CASE WHEN ? = 'failed' THEN ? ELSE NULL END,"
            " worker_id = NULL, lease_expires_at = NULL WHERE job_id = ?",
            (status, str(error)[:2000], now, status, now, job_id))
    return status


def requeue_job(job_id):
    """Put a terminally failed job back in the queue with a fresh budget.

    Backs the participant-facing "Try again" button: the brief asks for a
    recoverable failure state, and resetting attempts is what makes the retry
    meaningful rather than an instant re-fail.

    `manual_retries` is incremented and deliberately NOT reset, because it is
    the only durable record that the participant pressed the button — the row is
    reused, so its status cannot distinguish one manual retry from two. This
    replaces the old `events` table, which existed solely to count them.
    """
    now = utcnow()
    with transaction(immediate=True) as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'pending', attempts = 0, error = NULL,"
            " worker_id = NULL, lease_expires_at = NULL, completed_at = NULL,"
            " manual_retries = manual_retries + 1,"
            " deadline_at = ?, updated_at = ? WHERE job_id = ? AND status = 'failed'",
            (in_seconds(config.JOB_DEADLINE_SECONDS), now, job_id))
        return cursor.rowcount == 1


def queue_depth():
    """Count of pending and running jobs, as a dict. For the ops endpoint."""
    conn = connect()
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
    conn.close()
    return {row["status"]: row["n"] for row in rows}


def unsubmitted_sessions_with_done_jobs():
    """Sessions that saw a response but never submitted. For the ops endpoint.

    This is the cost of the completers-only design made visible: each of these
    is a participant who reached the response and then abandoned, whose answers
    exist nowhere. Worth watching during a live run, because a sudden rise means
    something is broken after the response page rather than people wandering off.
    """
    conn = connect()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM sessions s JOIN jobs j ON j.session_id = s.session_id"
        " WHERE s.submitted_at IS NULL AND j.status = 'done'").fetchone()
    conn.close()
    return row["n"]
