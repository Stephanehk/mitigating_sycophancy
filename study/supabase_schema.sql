-- Supabase table for the sycophancy user study.
--
-- Run this once in the Supabase SQL editor before launching. One row per
-- COMPLETED participant — by design, sessions that abandon mid-study never
-- reach here (the browser holds the payload until the end; see
-- static/app.js and study/README.md "Where the data lives").
--
-- Shape: a few promoted columns for the things you will filter and group by,
-- plus the whole payload as JSONB so nothing is lost and adding a questionnaire
-- item needs no migration. `answers` is split out of the payload as its own
-- JSONB column because it is the one part analysis always touches.

create table if not exists public.study_submissions (
    id                bigint generated always as identity primary key,

    -- Server-issued. UNIQUE is the idempotency guard: the server also refuses a
    -- second submit per session, but this makes a double-insert impossible even
    -- if that check is bypassed.
    session_id        text not null unique,

    -- Server-owned facts. NEVER sent to the browser, so they cannot be forged.
    -- NULLABLE ON PURPOSE. Screened-out and declining participants submit too
    -- (they reached the end of their path, and the screening funnel is a
    -- reportable result) but never reached a generation, so they have no arm.
    -- This was `not null` and would have rejected every one of them.
    arm               text check (arm in ('ours', 'untrained', 'syco', 'neutral')),
    job_id            text,
    topic_title       text,
    -- Exactly the string the model received (prompts.build_user_turn output).
    user_text         text,
    -- Exactly the text rendered on screen.
    shown_response    text,
    shown_response_chars integer,
    -- Per-stage raw outputs; the neutral arm's pre-edit Stage 1 text lives here.
    stages            jsonb,
    worker_attempts   integer,
    manual_retries    integer,

    -- Participant-reported.
    prolific_pid      text,
    consent           text not null check (consent in ('agree', 'decline')),
    eligibility       text not null check (eligibility in ('eligible', 'screened_out', 'declined')),
    selected_scenario text,
    screening_order   jsonb,
    screening         jsonb,
    perspective_text  text,
    additional_thoughts text,

    -- Questionnaire answers, keyed by response key (see
    -- questionnaires.all_response_keys). Values are the raw QSF choice ids.
    answers           jsonb not null default '{}'::jsonb,
    -- Response keys that arrived blank. Should always be empty: every question
    -- is required and the client blocks Continue. A non-empty array is a client
    -- bug worth investigating, not imputable missing data.
    missing_keys      jsonb not null default '[]'::jsonb,

    -- Per-stage client-side timestamps, for reading-time / dropout analysis.
    client_timings    jsonb,

    -- The browser's payload EXACTLY as received, before validation. A backstop
    -- against a validator bug quietly dropping something: the columns above are
    -- the normalized view, this is the original.
    payload           jsonb not null,

    session_created_at timestamptz,
    -- Stamped by the server, not by `default now()`, so the fallback JSONL
    -- carries the same value and both export sources agree on durations.
    submitted_at      timestamptz not null default now()
);

create index if not exists idx_submissions_arm on public.study_submissions (arm);
create index if not exists idx_submissions_prolific on public.study_submissions (prolific_pid);
create index if not exists idx_submissions_submitted on public.study_submissions (submitted_at);

-- The server writes with the SERVICE-ROLE key, which bypasses RLS. RLS is
-- enabled with no policies so that the anon/public key — which is what a
-- browser would ever hold — can neither read nor write this table.
alter table public.study_submissions enable row level security;
