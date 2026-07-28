-- Schema for the sycophancy user study's LOCAL database.
--
-- This database is now only the generation queue plus the thin session rows the
-- queue needs. Participant answers are NOT stored here: the browser accumulates
-- them for the whole session and posts one JSON payload to /api/submit at the
-- end, which the server forwards to Supabase (see supabase_client.py and
-- supabase_schema.sql).
--
-- What the server still owns, and why it cannot move to the client:
--   * `arm` — drawn server-side at job creation and never redrawn, so a
--     participant cannot re-randomize their condition from devtools.
--   * `jobs` — one row per requested generation; the GPU worker polls it.
--     One job per session, so the response cannot be regenerated until the
--     participant likes it.
-- Both guarantees are what keep the DVs interpretable, so they stay on the
-- server even though everything else moved to the browser.
--
-- Keep this file on LOCAL DISK. See config.check_production_config().

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- One participant's run through the study. Deliberately thin: no answers, no
-- stage cursor (the client owns navigation now), no free text beyond what the
-- model was given.
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    -- Opaque token the browser holds; separate from session_id so a leaked
    -- id in a log cannot be used to resume or mutate someone's session.
    session_token     TEXT NOT NULL UNIQUE,
    -- JSON array of scenario keys: the randomized screening order, drawn
    -- server-side so the randomization is recorded even though the client
    -- renders the pages.
    screening_order   TEXT NOT NULL,
    -- One of prompts.ARMS. Assigned at job creation (the moment of
    -- eligibility, so screened-out participants consume no arm slot) and
    -- never reassigned.
    arm               TEXT,
    user_agent        TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    -- Set when /api/submit has accepted this session's final payload. Also the
    -- idempotency guard: a second submit for the same session is refused.
    submitted_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_arm ON sessions (arm);

-- The generation queue. One row per requested model response.
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions (session_id) ON DELETE CASCADE,
    arm           TEXT NOT NULL,
    topic_title   TEXT NOT NULL,
    user_text     TEXT NOT NULL,
    -- 'pending' -> 'running' -> 'done' | 'failed'.
    status        TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    -- How many times the PARTICIPANT pressed "Try again". Distinct from
    -- `attempts`, which the worker burns; requeue_job resets attempts but not
    -- this, so views.retry_exhausted can still cap manual retries. Replaces the
    -- `events` table, which existed only to count these.
    manual_retries INTEGER NOT NULL DEFAULT 0,
    -- Set while status='running'; a job whose lease has passed is reclaimable.
    lease_expires_at TEXT,
    -- Identifies the worker holding the lease, for debugging.
    worker_id     TEXT,
    -- Wall-clock cutoff; past this the job fails even with attempts remaining.
    deadline_at   TEXT NOT NULL,
    -- Final text shown to the participant (post-edit for the neutral arm).
    response_text TEXT,
    -- JSON: every stage's raw output, so the neutral arm's pre-edit Stage 1
    -- text is recoverable for analysis.
    stages_json   TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    completed_at  TEXT
);

-- The worker's claim query filters on exactly this pair.
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs (status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs (session_id);
