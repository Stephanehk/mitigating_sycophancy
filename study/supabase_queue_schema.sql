-- Supabase schema for the sycophancy study's SESSION + JOB QUEUE.
--
-- Run this once in the Supabase SQL editor, alongside supabase_schema.sql
-- (which defines study_submissions_v2, the finished-participant table).
--
-- THE `_v2` SUFFIX (2026-08-07) is config.TABLE_SUFFIX. The 2026-08-07 revision
-- is a different instrument — new screening scenarios, and a draw over the
-- review-page arms — so it writes to its own tables and leaves the previous
-- run's rows where they are. The FUNCTION names are deliberately unsuffixed:
-- `create or replace` repoints them at the new tables, so the browser and the
-- GPU worker call exactly what they called before. Anything that renames a
-- table here must also change config.TABLE_SUFFIX;
-- tests/test_views.py::test_the_sql_files_and_config_agree_about_the_table_suffix
-- pins the two together.
--
-- WHY THIS EXISTS. The queue used to be SQLite on the study server's local
-- disk. That works for one always-on process; it cannot work behind Lambda,
-- where each invocation gets its own ephemeral filesystem and no two
-- invocations share state. Moving the queue into Postgres makes the API
-- stateless, which is what lets it run serverless.
--
-- WHY RPC FUNCTIONS RATHER THAN PLAIN TABLE ACCESS. Four of these operations
-- must be atomic against concurrent workers, and PostgREST cannot express
-- "SELECT ... FOR UPDATE SKIP LOCKED" or a guarded UPDATE ... RETURNING. Each
-- one is therefore a plpgsql function called via POST /rest/v1/rpc/<name>.
-- That also means the server needs only the service-role key over HTTPS — no
-- database password, and no connection pool to exhaust from Lambda.
--
-- SECURITY. RLS is enabled with no policies on both tables, so the anon /
-- publishable key can neither read nor write them. The functions are
-- SECURITY DEFINER so they still work, and are owned by the definer rather
-- than the caller; they take only the arguments they need and never accept
-- arbitrary SQL.

-- =========================================================================
-- Tables
-- =========================================================================

-- One participant's run. Deliberately thin: no answers, no stage cursor. The
-- browser owns navigation and posts everything at the end (see README §1b).
create table if not exists public.study_sessions_v2 (
    session_id        text primary key,
    -- Opaque token the browser holds; separate from session_id so a leaked id
    -- in a log cannot be used to resume or mutate someone's session.
    session_token     text not null unique,
    -- JSON array of scenario keys: the randomized screening order, drawn
    -- server-side so the randomization is recorded even though the client
    -- renders the pages.
    screening_order   jsonb not null,
    -- Assigned at the moment of eligibility (so screened-out participants
    -- consume no arm slot) and never reassigned.
    --
    -- NO INLINE CHECK. The set of legal arms lives in prompts.py and is emitted
    -- into supabase_app_schema.sql, which re-asserts the constraint every time
    -- it is run. Hardcoding the list here is what broke on 2026-07-30: two arms
    -- were added, the generated file was re-run, and every session in a new arm
    -- was rejected by a four-name check constraint this file had created months
    -- earlier — `create table if not exists` never revisits it.
    arm               text,
    -- True when the arm above was SET by a /debug_<key> link rather than drawn
    -- (start_session's p_debug_condition; see prompts.DEBUG_CONDITIONS). Those
    -- sessions exist so a condition can be walked by hand, and the export drops
    -- them, which is what keeps a link that needs no secret out of the dataset.
    --
    -- Declared here for a FRESH install only. `create table if not exists` never
    -- revisits an existing table, so supabase_app_schema.sql also emits an
    -- `add column if not exists` — that is the one that reaches a live database.
    is_debug          boolean not null default false,
    user_agent        text,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now(),
    -- Set when /api/submit has accepted the final payload; also the
    -- idempotency guard against a double submit.
    submitted_at      timestamptz
);

create index if not exists idx_sessions_v2_token on public.study_sessions_v2 (session_token);
create index if not exists idx_sessions_v2_arm on public.study_sessions_v2 (arm);

-- The generation queue. One row per requested model response.
create table if not exists public.study_jobs_v2 (
    job_id            text primary key,
    session_id        text not null references public.study_sessions_v2 (session_id) on delete cascade,
    arm               text not null,
    topic_title       text not null,
    user_text         text not null,
    status            text not null check (status in ('pending', 'running', 'done', 'failed')),
    attempts          integer not null default 0,
    -- Participant "Try again" presses. requeue_job increments and never resets
    -- it, because the row is reused and status cannot distinguish one from two.
    manual_retries    integer not null default 0,
    lease_expires_at  timestamptz,
    worker_id         text,
    deadline_at       timestamptz not null,
    response_text     text,
    -- Per-stage raw outputs; the neutral arm's pre-edit Stage 1 text lives here.
    stages            jsonb,
    error             text,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now(),
    completed_at      timestamptz
);

-- claim_job filters on exactly this pair, in this order.
create index if not exists idx_jobs_v2_claim on public.study_jobs_v2 (status, created_at);
create index if not exists idx_jobs_v2_session on public.study_jobs_v2 (session_id, created_at desc);

alter table public.study_sessions_v2 enable row level security;
alter table public.study_jobs_v2 enable row level security;


-- =========================================================================
-- Atomic queue operations
-- =========================================================================

-- Claim the oldest pending job. THE ONE OPERATION THAT MUST NOT RACE.
--
-- FOR UPDATE SKIP LOCKED is the Postgres primitive for a work queue: two
-- workers calling this simultaneously take DIFFERENT rows rather than
-- blocking or double-claiming. This replaces SQLite's BEGIN IMMEDIATE, and is
-- strictly better — the SQLite version serialized every claim globally.
--
-- Attempts increment at CLAIM time, not completion, so a worker that crashes
-- mid-generation still burns an attempt. Without that, a job that reliably
-- kills its worker would be retried forever.
--
-- DROPPED FIRST because its return type is a TABLE type. `create or replace`
-- cannot change a function's return type, so re-running this file against a
-- database where claim_job already returns `setof public.study_jobs` (the
-- pre-2026-08-07 table) fails with "cannot change return type of existing
-- function" rather than repointing it. The drop makes the rename idempotent.
drop function if exists public.claim_job(text, integer);

create or replace function public.claim_job(
    p_worker_id text,
    p_lease_seconds integer
) returns setof public.study_jobs_v2
language plpgsql security definer
set search_path = public, pg_temp as $$
declare
    v_job_id text;
begin
    select job_id into v_job_id
    from public.study_jobs_v2
    where status = 'pending'
    order by created_at, job_id
    for update skip locked
    limit 1;

    if v_job_id is null then
        return;
    end if;

    return query
    update public.study_jobs_v2
    set status = 'running',
        worker_id = p_worker_id,
        attempts = attempts + 1,
        lease_expires_at = now() + make_interval(secs => p_lease_seconds),
        updated_at = now()
    where job_id = v_job_id
    returning *;
end;
$$;

-- Return timed-out jobs to the queue, and fail jobs that are out of road.
--
-- Two distinct expiries, both here so every call site gets the same semantics:
--   lease expiry   — a 'running' job whose worker vanished becomes 'pending'.
--   deadline/attempts — becomes 'failed', which releases the participant from
--                       the loading screen rather than leaving them spinning.
create or replace function public.reap_expired_jobs(
    p_max_attempts integer
) returns table (requeued integer, failed integer)
language plpgsql security definer
set search_path = public, pg_temp as $$
declare
    v_requeued integer;
    v_failed integer;
begin
    with r as (
        update public.study_jobs_v2
        set status = 'pending', worker_id = null, lease_expires_at = null,
            updated_at = now()
        where status = 'running' and lease_expires_at is not null
          and lease_expires_at < now()
        returning 1
    ) select count(*)::integer into v_requeued from r;

    with f as (
        update public.study_jobs_v2
        set status = 'failed', updated_at = now(), completed_at = now(),
            error = coalesce(error, 'exhausted: attempts or deadline')
        where status in ('pending', 'running')
          and (attempts >= p_max_attempts or deadline_at < now())
        returning 1
    ) select count(*)::integer into v_failed from f;

    return query select v_requeued, v_failed;
end;
$$;

-- Mark a job done. Returns true only if THIS worker still held the lease, so a
-- worker whose lease expired mid-generation cannot overwrite a result a second
-- worker already produced — its update matches zero rows and it learns it lost.
create or replace function public.complete_job(
    p_job_id text,
    p_worker_id text,
    p_response_text text,
    p_stages jsonb
) returns boolean
language plpgsql security definer
set search_path = public, pg_temp as $$
declare
    v_rows integer;
begin
    if p_response_text is null or btrim(p_response_text) = '' then
        raise exception 'refusing to complete job % with empty response text', p_job_id;
    end if;
    update public.study_jobs_v2
    set status = 'done', response_text = p_response_text, stages = p_stages,
        error = null, updated_at = now(), completed_at = now(),
        lease_expires_at = null
    where job_id = p_job_id and worker_id = p_worker_id and status = 'running';
    get diagnostics v_rows = row_count;
    return v_rows = 1;
end;
$$;

-- Record a failed attempt. Back to 'pending' if attempts and deadline remain,
-- otherwise terminal 'failed'. Returns the resulting status, or null if this
-- worker no longer held the job.
create or replace function public.fail_job_attempt(
    p_job_id text,
    p_worker_id text,
    p_error text,
    p_max_attempts integer
) returns text
language plpgsql security definer
set search_path = public, pg_temp as $$
declare
    v_attempts integer;
    v_deadline timestamptz;
    v_status text;
begin
    select attempts, deadline_at into v_attempts, v_deadline
    from public.study_jobs_v2
    where job_id = p_job_id and worker_id = p_worker_id and status = 'running'
    for update;

    if not found then
        return null;
    end if;

    v_status := case when v_attempts >= p_max_attempts or v_deadline < now()
                     then 'failed' else 'pending' end;

    update public.study_jobs_v2
    set status = v_status,
        error = left(p_error, 2000),
        updated_at = now(),
        completed_at = case when v_status = 'failed' then now() else null end,
        worker_id = null,
        lease_expires_at = null
    where job_id = p_job_id;

    return v_status;
end;
$$;

-- Extend a running job's lease. Only the lease holder may extend, so a worker
-- that already lost the job to expiry cannot reclaim it behind the new holder.
create or replace function public.extend_lease(
    p_job_id text,
    p_worker_id text,
    p_lease_seconds integer
) returns boolean
language plpgsql security definer
set search_path = public, pg_temp as $$
declare
    v_rows integer;
begin
    update public.study_jobs_v2
    set lease_expires_at = now() + make_interval(secs => p_lease_seconds),
        updated_at = now()
    where job_id = p_job_id and worker_id = p_worker_id and status = 'running';
    get diagnostics v_rows = row_count;
    return v_rows = 1;
end;
$$;

-- Put a terminally failed job back in the queue with a fresh attempt budget,
-- backing the participant's "Try again". manual_retries is incremented and
-- never reset — it is the only durable record that the button was pressed.
create or replace function public.requeue_job(
    p_job_id text,
    p_deadline_seconds integer
) returns boolean
language plpgsql security definer
set search_path = public, pg_temp as $$
declare
    v_rows integer;
begin
    update public.study_jobs_v2
    set status = 'pending', attempts = 0, error = null, worker_id = null,
        lease_expires_at = null, completed_at = null,
        manual_retries = manual_retries + 1,
        deadline_at = now() + make_interval(secs => p_deadline_seconds),
        updated_at = now()
    where job_id = p_job_id and status = 'failed';
    get diagnostics v_rows = row_count;
    return v_rows = 1;
end;
$$;

-- Record that a session's final payload was accepted. Guarded on
-- submitted_at IS NULL, so a duplicate submit matches zero rows and the caller
-- learns it was already done rather than writing the submission twice.
create or replace function public.mark_submitted(
    p_session_id text
) returns boolean
language plpgsql security definer
set search_path = public, pg_temp as $$
declare
    v_rows integer;
begin
    update public.study_sessions_v2
    set submitted_at = now(), updated_at = now()
    where session_id = p_session_id and submitted_at is null;
    get diagnostics v_rows = row_count;
    return v_rows = 1;
end;
$$;

-- Ops counters for /api/health, as one round trip instead of four.
create or replace function public.queue_stats()
returns jsonb
language sql security definer
set search_path = public, pg_temp as $$
    select jsonb_build_object(
        'queue', coalesce((select jsonb_object_agg(status, n)
                           from (select status, count(*)::int as n
                                 from public.study_jobs_v2 group by status) q), '{}'::jsonb),
        'submitted_by_arm', coalesce((select jsonb_object_agg(arm, n)
                           from (select arm, count(*)::int as n
                                 from public.study_sessions_v2
                                 where arm is not null and submitted_at is not null
                                 group by arm) a), '{}'::jsonb),
        'saw_response_never_submitted', (
            select count(*)::int from public.study_sessions_v2 s
            join public.study_jobs_v2 j on j.session_id = s.session_id
            where s.submitted_at is null and j.status = 'done')
    );
$$;

-- The service-role key bypasses RLS, but PostgREST still requires the function
-- to be executable by the role it authenticates as.
grant execute on function public.claim_job(text, integer) to service_role;
grant execute on function public.reap_expired_jobs(integer) to service_role;
grant execute on function public.complete_job(text, text, text, jsonb) to service_role;
grant execute on function public.fail_job_attempt(text, text, text, integer) to service_role;
grant execute on function public.extend_lease(text, text, integer) to service_role;
grant execute on function public.requeue_job(text, integer) to service_role;
grant execute on function public.mark_submitted(text) to service_role;
grant execute on function public.queue_stats() to service_role;
