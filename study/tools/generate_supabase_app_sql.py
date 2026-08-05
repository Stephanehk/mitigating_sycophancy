"""Generate supabase_app_schema.sql — the RPCs the browser and worker call.

WHY GENERATED RATHER THAN HAND-WRITTEN. Dropping the API server moves four
things out of Python and into Postgres: the arm draw, the user-turn
composition, the free-text length checks and the answer-domain validation. The
last of those needs every questionnaire item's allowed values, and hand-copying
42 response keys into SQL would make questionnaires.py stop being the source of
truth the first time someone edited one and forgot the other. So the data
tables are emitted from questionnaires.py / scenarios.py / config.py, and
regenerating is a one-liner:

    python tools/generate_supabase_app_sql.py > supabase_app_schema.sql

AUTHENTICATION MODEL. There is no server to hold a privileged key, so nothing
deployed holds one. Both the browser and the GPU worker use the ANON key, which
is public and can do nothing on its own — RLS blocks every table. Each RPC is
SECURITY DEFINER and authenticates its own caller:

    participants  an opaque session_token, issued by start_session
    the worker    a shared secret, compared against study_config inside the RPC

service_role is then only ever used from a developer machine, for exports.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import config
import questionnaires
import scenarios
import views
from prompts import ARMS, DEBUG_CONDITIONS, RANDOMIZED_ARMS, SKIP_REVIEW_ARMS


def sql_str(value):
    """A single-quoted SQL literal, or NULL."""
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def emit_scenarios():
    """The four dilemmas, for topic_title lookup and scenario-key validation.

    Only `key` and `topic_title` are needed server-side — the participant-facing
    prose ships to the browser in the static content bundle, not from here.
    """
    lines = ["-- Generated from scenarios.py. Only what the queue needs: the key",
             "-- (validated on start_response) and topic_title (stored on the job).",
             "create table if not exists public.study_scenarios (",
             "    key         text primary key,",
             "    topic_title text not null,",
             "    position    integer not null",
             ");",
             "truncate public.study_scenarios;",
             "insert into public.study_scenarios (key, topic_title, position) values"]
    rows = ["    (%s, %s, %d)" % (sql_str(s["key"]), sql_str(s["topic_title"]), i)
            for i, s in enumerate(scenarios.SCENARIOS)]
    lines.append(",\n".join(rows) + ";")
    return "\n".join(lines)


def emit_questionnaire_options():
    """Every (response_key, allowed value) pair, for submit-time validation.

    This is the check that stops a tampered client writing a code no option maps
    to. It is also why this file is generated: the pairs come straight from
    questionnaires.py, so the two cannot drift.
    """
    pairs = []
    for question in questionnaires.all_questions():
        for key in questionnaires.response_keys(question):
            for option in question["options"]:
                pairs.append((key, question["id"], option["value"]))
    lines = ["-- Generated from questionnaires.py: %d (response_key, value) pairs"
             " across %d response keys." % (len(pairs), len(questionnaires.all_response_keys())),
             "create table if not exists public.questionnaire_options (",
             "    response_key text not null,",
             "    question_id  text not null,",
             "    value        integer not null,",
             "    primary key (response_key, value)",
             ");",
             "truncate public.questionnaire_options;",
             "insert into public.questionnaire_options (response_key, question_id, value) values"]
    rows = ["    (%s, %s, %d)" % (sql_str(k), sql_str(q), v) for k, q, v in pairs]
    lines.append(",\n".join(rows) + ";")
    return "\n".join(lines)


def emit_arm_domain():
    """Re-assert `arm in (...)` on both arm columns, from prompts.ARMS.

    THE DOMAIN IS ARMS, NOT RANDOMIZED_ARMS, and that is deliberate: it is what
    a row may legally contain, not what the draw produces. Retired arms stay in
    it so the rows already collected under them remain valid, and so a debug
    link can still force one.

    THIS IS A REPAIR AS WELL AS A DEFINITION. The two hand-written schema files
    used to carry the list inline, and both use `create table if not exists` —
    so when prompts.py gained two arms on 2026-07-30, re-running everything left
    the OLD four-name constraint in place and the database rejected every
    session drawn into a new arm. Emitting it here, dropped and re-added, means
    the constraint tracks prompts.py on every deploy instead of being frozen at
    whenever the table happened to be created.

    Idempotent: `drop constraint if exists` then `add constraint`. Widening is
    always safe; narrowing would fail loudly against rows already outside the
    new set, which is the right way for that to go wrong.
    """
    values = ", ".join(sql_str(a) for a in ARMS)
    lines = ["-- Generated from prompts.ARMS: the %d legal arm values." % len(ARMS)]
    for table in ("study_sessions", "study_submissions"):
        lines.append(
            "alter table public.%s drop constraint if exists %s_arm_check;" % (table, table))
        lines.append(
            "alter table public.%s add constraint %s_arm_check\n    check (arm in (%s));"
            % (table, table, values))
    return "\n".join(lines)


def emit_debug_column():
    """Add `is_debug` to both session-bearing tables, idempotently.

    EMITTED RATHER THAN DECLARED for the same reason the arm constraint is: both
    hand-written schema files use `create table if not exists`, so a column added
    to their CREATE TABLE never appears in a database that already exists. The
    tables declare it too, for a fresh install; this is what reaches an old one.

    Marks a session whose arm was SET by a debug link instead of drawn (see
    prompts.DEBUG_CONDITIONS). It is the whole safety story for those links:
    analysis/export_responses.py drops these rows by default, so a forced arm
    cannot enter the dataset even though the link needs no secret.
    """
    lines = ["-- Sessions whose arm was forced by a /debug_<key> link rather than drawn.",
             "-- Excluded from the export by default; see prompts.DEBUG_CONDITIONS."]
    for table in ("study_sessions", "study_submissions"):
        lines.append("alter table public.%s add column if not exists "
                     "is_debug boolean not null default false;" % (table,))
    return "\n".join(lines)


def emit_config():
    """Tunables the RPCs need, so thresholds live in one place.

    `worker_token` is set separately by the operator (see the note in the
    generated file) — it is a secret and must not be committed.
    """
    values = {
        "min_describe_chars": config.MIN_DESCRIBE_CHARS,
        "min_perspective_chars": config.MIN_PERSPECTIVE_CHARS,
        "max_text_chars": config.MAX_TEXT_CHARS,
        "max_model_input_chars": config.MAX_MODEL_INPUT_CHARS,
        "job_deadline_seconds": config.JOB_DEADLINE_SECONDS,
        "job_lease_seconds": config.JOB_LEASE_SECONDS,
        "job_max_attempts": config.JOB_MAX_ATTEMPTS,
        "max_manual_retries": views.MAX_MANUAL_RETRIES,
        "client_poll_interval_ms": config.CLIENT_POLL_INTERVAL_MS,
        "client_slow_notice_ms": config.CLIENT_SLOW_NOTICE_MS,
    }
    lines = ["-- Generated from config.py. Thresholds the RPCs enforce.",
             "create table if not exists public.study_config (",
             "    key   text primary key,",
             "    value text not null",
             ");"]
    for key, value in values.items():
        lines.append("insert into public.study_config (key, value) values (%s, %s)"
                     "\n    on conflict (key) do update set value = excluded.value;"
                     % (sql_str(key), sql_str(value)))
    lines.append("")
    lines.append("-- The worker's shared secret. NOT generated — set it yourself, once:")
    lines.append("--   insert into public.study_config (key, value)")
    lines.append("--   values ('worker_token', '<the same secret the worker exports>')")
    lines.append("--   on conflict (key) do update set value = excluded.value;")
    return "\n".join(lines)


#: What session_arm samples. prompts.RANDOMIZED_ARMS, NOT prompts.ARMS — the
#: retired arms stay legal values (emit_arm_domain) without being drawn.
RANDOMIZED_ARRAY = "array[" + ", ".join(sql_str(a) for a in RANDOMIZED_ARMS) + "]"
SKIP_REVIEW_ARRAY = "array[" + ", ".join(sql_str(a) for a in SKIP_REVIEW_ARMS) + "]"

#: The /debug_<key> → arm mapping, as the branches of a CASE. Generated from
#: prompts.DEBUG_CONDITIONS so the browser never has to know an arm name: it
#: sends the key from its URL and the database does the lookup.
DEBUG_CASE = "\n".join(
    "                      when %s then %s" % (sql_str(key), sql_str(arm))
    for key, arm in sorted(DEBUG_CONDITIONS.items()))

RPCS = """
-- =========================================================================
-- Helpers
-- =========================================================================

create or replace function public.cfg(p_key text) returns integer
language sql stable security definer set search_path = public, pg_temp as $$
    select value::integer from public.study_config where key = p_key;
$$;

-- The client-facing view of a job. Shared by start_response, response_status
-- and retry_response so the three cannot describe the same job differently.
create or replace function public.job_state(p_job public.study_jobs) returns jsonb
language sql stable security definer set search_path = public, pg_temp as $$
    select jsonb_strip_nulls(jsonb_build_object(
        'job_id', p_job.job_id,
        'status', p_job.status,
        'poll_interval_ms', cfg('client_poll_interval_ms'),
        'slow_notice_ms', cfg('client_slow_notice_ms'),
        'text', case when p_job.status = 'done' then p_job.response_text end,
        'can_retry', case when p_job.status = 'failed'
                          then p_job.manual_retries < cfg('max_manual_retries') end));
$$;

-- Resolve a participant's opaque token. Raises rather than returning null, so
-- every caller fails the same way on a lost session.
create or replace function public.session_for_token(p_token text)
returns public.study_sessions
language plpgsql stable security definer set search_path = public, pg_temp as $$
declare v_session public.study_sessions;
begin
    select * into v_session from public.study_sessions where session_token = p_token;
    if not found then
        raise exception 'unknown session' using errcode = 'P0002';
    end if;
    return v_session;
end;
$$;

-- Draw this session's arm, ONCE, and return it.
--
-- A UNIFORM TWO-WAY DRAW over prompts.RANDOMIZED_ARMS (ours_noreview /
-- untrained_noreview): our model vs the untrained base, both on the flow where
-- the participant never edits the model's input. The four retired arms are still
-- legal column values but are never sampled, and no debug link forces one, so
-- nothing new can land in them.
--
-- Sticky by construction: the update only fires while `arm` is null, and the
-- value is re-read afterwards so a caller that lost a concurrent race still
-- gets the arm that actually landed. That is what stops a participant re-rolling
-- their condition by calling response_flow or start_response again.
--
-- Called at the moment of eligibility (from response_flow, or from
-- start_response if the client skipped it), never at session creation, so
-- screen-outs and decliners keep a null arm.
--
-- A debug session (start_session's p_debug_condition) already has one, so the
-- same stickiness that stops a re-roll is what makes a forced arm hold: this
-- function reads it and draws nothing.
create or replace function public.session_arm(p_session_id text) returns text
language plpgsql security definer set search_path = public, pg_temp as $$
declare v_arm text;
begin
    select arm into v_arm from public.study_sessions where session_id = p_session_id;
    if v_arm is null then
        update public.study_sessions
           set arm = (__RANDOMIZED_ARRAY__)[1 + floor(random() * __RANDOMIZED_COUNT__)],
               updated_at = now()
         where session_id = p_session_id and arm is null;
        select arm into v_arm from public.study_sessions where session_id = p_session_id;
    end if;
    return v_arm;
end;
$$;

-- The worker's shared secret. Every worker RPC calls this first; the anon key
-- alone gets nothing.
create or replace function public.assert_worker(p_worker_token text) returns void
language plpgsql stable security definer set search_path = public, pg_temp as $$
declare v_expected text;
begin
    select value into v_expected from public.study_config where key = 'worker_token';
    if v_expected is null or v_expected = '' then
        raise exception 'worker_token is not configured' using errcode = 'P0001';
    end if;
    if p_worker_token is distinct from v_expected then
        raise exception 'bad worker token' using errcode = '28000';
    end if;
end;
$$;


-- =========================================================================
-- Participant RPCs  (anon key + session_token)
-- =========================================================================

-- Start a session and draw this participant's screening order.
--
-- p_debug_condition IS THE ONE WAY TO CHOOSE AN ARM, and it is for walking a
-- condition by hand (see prompts.DEBUG_CONDITIONS and the /debug_<key> pages).
-- Random assignment is precisely what you cannot work around when you need to
-- SEE a condition. Since the draw narrowed to the two no-review arms these links
-- reach the same flow a participant does; what they add is knowing which model
-- is answering, which the draw hides on purpose.
--
-- WHAT KEEPS IT FROM CORRUPTING THE DATA. The session is stamped is_debug, and
-- the export drops those rows by default, so a forced arm is quarantined rather
-- than counted. That guarantee does not depend on the link staying secret —
-- which is why it is not secret. The randomization itself is untouched: a
-- participant who does not pass this argument gets a null arm here and a drawn
-- one at eligibility, exactly as before.
--
-- The KEY is sent, never an arm name, so the browser still has no vocabulary for
-- the conditions; the mapping below is generated from prompts.py.
drop function if exists public.start_session(text);

create or replace function public.start_session(
    p_user_agent text, p_debug_condition text default null)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_id text := replace(gen_random_uuid()::text, '-', '');
    v_token text := replace(gen_random_uuid()::text, '-', '');
    v_order jsonb;
    v_arm text := null;
begin
    if coalesce(p_debug_condition, '') <> '' then
        v_arm := case p_debug_condition
__DEBUG_CASE__
                 end;
        -- A mistyped link must not silently become a normally randomized
        -- session: someone testing a condition would then believe they had
        -- seen it.
        if v_arm is null then
            raise exception 'unknown debug condition: %', p_debug_condition
                using errcode = 'P0001';
        end if;
    end if;
    -- Randomized per participant, drawn here so the order is recorded even
    -- though the browser renders the pages.
    select jsonb_agg(key order by random()) into v_order from public.study_scenarios;
    insert into public.study_sessions (session_id, session_token, screening_order,
                                       user_agent, arm, is_debug)
    values (v_id, v_token, v_order, left(coalesce(p_user_agent, ''), 500),
            v_arm, v_arm is not null);
    -- is_debug, not the arm: enough for the client to confirm the link worked
    -- and to show its banner, and it still names no condition.
    return jsonb_build_object('session_id', v_id, 'session_token', v_token,
                              'screening_order', v_order, 'has_job', false,
                              'submitted', false, 'is_debug', v_arm is not null);
end;
$$;

-- Which flow this participant walks: with the review-and-edit page, or without.
--
-- WHY THE CLIENT HAS TO ASK. Two arms (SKIP_REVIEW_ARMS) remove the page that
-- shows the participant their concatenated writing and offers it for editing.
-- Skipping a visible page cannot be hidden from the browser, so the arm is
-- drawn here — one page before it is needed — and exactly one bit of it is
-- returned: `show_review`. The arm ITSELF is still never sent.
--
-- SINCE THE DRAW NARROWED TO THE TWO NO-REVIEW ARMS (2026-08-05) this bit is
-- constant for real participants — every one of them gets `false` — so it
-- partitions nothing and leaks nothing. It is still what makes the flow happen:
-- `false` is the instruction to drop the page and enqueue the generation
-- straight from response_intro. The within-session guarantees are unchanged: the
-- draw is sticky (session_arm), the job is one-per-session, and WHICH of the two
-- arms a participant got is unknowable to the browser.
create or replace function public.response_flow(p_token text)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.study_sessions;
    v_arm text;
begin
    v_session := public.session_for_token(p_token);
    v_arm := public.session_arm(v_session.session_id);
    return jsonb_build_object(
        'show_review', not (v_arm = any (__SKIP_REVIEW_ARRAY__)));
end;
$$;

-- Enqueue the generation from the text the participant CONFIRMED.
--
-- Takes the finished user turn, not the two halves. On the no-review arms — which
-- since 2026-08-05 is every arm a participant can be drawn into — the browser
-- composes the two halves and sends the result without showing it. On the review
-- arms (retired) a page showed them exactly what would be sent and let them edit
-- it before pressing Confirm, which is why the composition lives in the browser
-- at all: composing it server-side afterwards would have discarded those edits.
-- The string is the same either way.
--
-- The floor is min_describe + min_perspective, i.e. the same total the two
-- separate pages already enforced, so editing cannot be used to get past the
-- length requirement by deleting most of the text.
--
-- THE ARM IS DRAWN HERE AND NEVER RETURNED. That is the point of doing this in
-- the database rather than the browser: a client that could pick its own arm
-- could re-roll until it got the one it wanted, which would corrupt every DV.
--
-- IDEMPOTENT, AND THAT IS A DATA-INTEGRITY RULE. A session gets one job,
-- forever. Re-posting with different text returns the existing job unchanged,
-- so the response cannot be regenerated until the participant likes it.
drop function if exists public.start_response(text, text, text, text);

create or replace function public.start_response(
    p_token text, p_scenario_key text, p_user_text text)
returns jsonb
language plpgsql security definer
set search_path = public, pg_temp as $$
declare
    v_session public.study_sessions;
    v_job public.study_jobs;
    v_topic text;
    v_arm text;
    v_text text := btrim(coalesce(p_user_text, ''));
    v_min integer;
begin
    v_session := public.session_for_token(p_token);

    select * into v_job from public.study_jobs
    where session_id = v_session.session_id
    order by created_at desc limit 1;
    if found then
        return public.job_state(v_job);
    end if;

    select topic_title into v_topic from public.study_scenarios where key = p_scenario_key;
    if v_topic is null then
        raise exception 'Unknown scenario.' using errcode = 'P0001';
    end if;

    v_min := cfg('min_describe_chars') + cfg('min_perspective_chars');
    if length(v_text) < v_min then
        raise exception 'Please write a bit more — at least % characters.', v_min
            using errcode = 'P0001';
    end if;
    -- The COMPOSED cap, not the per-field one: this text is two capped fields
    -- joined by a blank line. On a no-review arm the participant has no page
    -- holding it, so a rejection here would be a dead end rather than something
    -- they could shorten.
    if length(v_text) > cfg('max_model_input_chars') then
        raise exception 'Your answer is too long (limit % characters).',
            cfg('max_model_input_chars') using errcode = 'P0001';
    end if;

    -- Drawn at the moment of eligibility, so screened-out participants never
    -- consume an arm slot. Normally response_flow has already drawn it one page
    -- earlier; session_arm is sticky, so this returns that same arm rather than
    -- re-rolling, and still works for a client that never called response_flow.
    v_arm := public.session_arm(v_session.session_id);

    insert into public.study_jobs (job_id, session_id, arm, topic_title, user_text,
                                   status, deadline_at)
    values (replace(gen_random_uuid()::text, '-', ''), v_session.session_id, v_arm,
            v_topic, v_text, 'pending',
            now() + make_interval(secs => cfg('job_deadline_seconds')))
    returning * into v_job;

    return public.job_state(v_job);
end;
$$;

-- Poll the generation. Reaps expired jobs first, so a participant on the
-- loading screen is what drives lease recovery — no cron needed.
create or replace function public.response_status(p_token text)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.study_sessions;
    v_job public.study_jobs;
begin
    v_session := public.session_for_token(p_token);
    perform public.reap_expired_jobs(cfg('job_max_attempts'));
    select * into v_job from public.study_jobs
    where session_id = v_session.session_id order by created_at desc limit 1;
    if not found then
        raise exception 'no generation for this session' using errcode = 'P0002';
    end if;
    return public.job_state(v_job);
end;
$$;

-- Requeue a failed generation, backing the participant's "Try again".
create or replace function public.retry_response(p_token text)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.study_sessions;
    v_job public.study_jobs;
begin
    v_session := public.session_for_token(p_token);
    select * into v_job from public.study_jobs
    where session_id = v_session.session_id order by created_at desc limit 1;
    if not found or v_job.status <> 'failed' then
        raise exception 'no failed generation to retry' using errcode = 'P0001';
    end if;
    if v_job.manual_retries >= cfg('max_manual_retries') then
        raise exception 'retries exhausted' using errcode = 'P0001';
    end if;
    perform public.requeue_job(v_job.job_id, cfg('job_deadline_seconds'));
    select * into v_job from public.study_jobs where job_id = v_job.job_id;
    return public.job_state(v_job);
end;
$$;

-- Accept the whole study in one payload.
--
-- The client's answers are validated against questionnaire_options, then
-- MERGED WITH SERVER-OWNED FACTS — arm, job id, the exact user_text the model
-- received, the response text, per-stage outputs, attempt counts — none of
-- which is taken from the payload, because the browser was never told the arm
-- and must not be able to claim one.
--
-- Two failure modes, handled differently on purpose: an out-of-domain answer
-- value is a tampered client and RAISES, but a MISSING answer is recorded in
-- missing_keys rather than rejected, because this runs after the participant
-- has finished and throwing away a completed session over one blank cell is
-- worse than recording the gap.
create or replace function public.submit_study(p_token text, p_payload jsonb)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.study_sessions;
    v_job public.study_jobs;
    v_answers jsonb := coalesce(p_payload -> 'answers', '{}'::jsonb);
    v_key text;
    v_raw text;
    v_val integer;
    v_missing jsonb;
    v_eligibility text := p_payload ->> 'eligibility';
    v_consent text := p_payload ->> 'consent';
    v_selected text := p_payload ->> 'selected_scenario';
begin
    v_session := public.session_for_token(p_token);

    if v_consent is null or v_consent not in ('agree', 'decline') then
        raise exception 'Missing or invalid consent.' using errcode = 'P0001';
    end if;
    if v_eligibility is null or v_eligibility not in ('eligible', 'screened_out', 'declined') then
        raise exception 'Missing or invalid eligibility.' using errcode = 'P0001';
    end if;
    if v_selected is not null
       and not exists (select 1 from public.study_scenarios where key = v_selected) then
        raise exception 'Unknown selected scenario.' using errcode = 'P0001';
    end if;
    if v_eligibility = 'eligible' and v_selected is null then
        raise exception 'An eligible submission must name a selected scenario.'
            using errcode = 'P0001';
    end if;

    -- Every supplied answer must be a code some option actually declares.
    for v_key, v_raw in select * from jsonb_each_text(v_answers) loop
        if v_raw is null or v_raw = '' then
            continue;
        end if;
        begin
            v_val := v_raw::integer;
        exception when others then
            raise exception 'Invalid answer for %.', v_key using errcode = 'P0001';
        end;
        if not exists (select 1 from public.questionnaire_options
                       where response_key = v_key and value = v_val) then
            raise exception 'Invalid answer for %.', v_key using errcode = 'P0001';
        end if;
    end loop;

    select coalesce(jsonb_agg(q.response_key order by q.response_key), '[]'::jsonb)
    into v_missing
    from (select distinct response_key from public.questionnaire_options) q
    where not (v_answers ? q.response_key);

    select * into v_job from public.study_jobs
    where session_id = v_session.session_id order by created_at desc limit 1;
    if v_eligibility = 'eligible' and v_job.job_id is null then
        raise exception 'Eligible submission with no generation job.' using errcode = 'P0001';
    end if;

    if v_session.submitted_at is not null then
        return jsonb_build_object('ok', true, 'duplicate', true);
    end if;
    if not public.mark_submitted(v_session.session_id) then
        return jsonb_build_object('ok', true, 'duplicate', true);
    end if;

    -- is_debug is carried from the SESSION, not the payload, for the same reason
    -- the arm is: a debug walk-through must be identifiable at analysis time
    -- even though the browser has no say in it.
    insert into public.study_submissions (
        session_id, arm, is_debug, job_id, topic_title, user_text, shown_response,
        shown_response_chars, stages, worker_attempts, manual_retries,
        prolific_pid, consent, eligibility, selected_scenario, screening_order,
        screening, perspective_text, additional_thoughts, answers, missing_keys,
        client_timings, payload, session_created_at, submitted_at)
    values (
        v_session.session_id, v_session.arm, v_session.is_debug,
        v_job.job_id, v_job.topic_title,
        v_job.user_text, v_job.response_text, length(v_job.response_text),
        v_job.stages, v_job.attempts, v_job.manual_retries,
        left(coalesce(p_payload ->> 'prolific_pid', ''), 200),
        v_consent, v_eligibility, v_selected, v_session.screening_order,
        coalesce(p_payload -> 'screening', '[]'::jsonb),
        left(coalesce(p_payload ->> 'perspective_text', ''), cfg('max_text_chars')),
        left(coalesce(p_payload ->> 'additional_thoughts', ''), cfg('max_text_chars')),
        v_answers, v_missing,
        coalesce(p_payload -> 'client_timings', '{}'::jsonb),
        p_payload, v_session.created_at, now());

    return jsonb_build_object('ok', true, 'duplicate', false,
                              'missing_keys', v_missing);
end;
$$;


-- =========================================================================
-- Worker RPCs  (anon key + the shared worker token)
-- =========================================================================

create or replace function public.worker_claim(p_worker_token text, p_worker_id text)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare v_job public.study_jobs;
begin
    perform public.assert_worker(p_worker_token);
    perform public.reap_expired_jobs(cfg('job_max_attempts'));
    select * into v_job from public.claim_job(left(p_worker_id, 120),
                                              cfg('job_lease_seconds'));
    if not found then
        return jsonb_build_object('job', null);
    end if;
    return jsonb_build_object('job', jsonb_build_object(
        'job_id', v_job.job_id, 'arm', v_job.arm, 'topic_title', v_job.topic_title,
        'user_text', v_job.user_text, 'attempts', v_job.attempts));
end;
$$;

create or replace function public.worker_heartbeat(
    p_worker_token text, p_job_id text, p_worker_id text) returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
begin
    perform public.assert_worker(p_worker_token);
    return jsonb_build_object('extended',
        public.extend_lease(p_job_id, p_worker_id, cfg('job_lease_seconds')));
end;
$$;

create or replace function public.worker_complete(
    p_worker_token text, p_job_id text, p_worker_id text,
    p_response_text text, p_stages jsonb) returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
begin
    perform public.assert_worker(p_worker_token);
    return jsonb_build_object('accepted',
        public.complete_job(p_job_id, p_worker_id, p_response_text, p_stages));
end;
$$;

create or replace function public.worker_fail(
    p_worker_token text, p_job_id text, p_worker_id text, p_error text) returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
begin
    perform public.assert_worker(p_worker_token);
    return jsonb_build_object('status',
        public.fail_job_attempt(p_job_id, p_worker_id, p_error,
                                cfg('job_max_attempts')));
end;
$$;


-- =========================================================================
-- Grants
-- =========================================================================
-- anon is the ONLY key deployed anywhere — it sits in the SPA's page source
-- and on the GPU worker. It can execute exactly these functions and touch no
-- table directly (RLS is on with no policies), and every function authenticates
-- its own caller. service_role stays on a developer machine, for exports.

grant execute on function public.start_session(text, text) to anon, authenticated;
grant execute on function public.response_flow(text) to anon, authenticated;
grant execute on function public.start_response(text, text, text) to anon, authenticated;
grant execute on function public.response_status(text) to anon, authenticated;
grant execute on function public.retry_response(text) to anon, authenticated;
grant execute on function public.submit_study(text, jsonb) to anon, authenticated;
grant execute on function public.worker_claim(text, text) to anon, authenticated;
grant execute on function public.worker_heartbeat(text, text, text) to anon, authenticated;
grant execute on function public.worker_complete(text, text, text, text, jsonb) to anon, authenticated;
grant execute on function public.worker_fail(text, text, text, text) to anon, authenticated;

-- NOT ENOUGH ON ITS OWN. Postgres grants EXECUTE on every function to PUBLIC by
-- default, so "we simply did not grant it to anon" denies nothing — anon
-- inherits PUBLIC. Verified the hard way on 2026-07-28: with only the grants
-- above in place, the anon key could call claim_job, complete_job,
-- mark_submitted, requeue_job, reap_expired_jobs and queue_stats, and could
-- SELECT study_config and read the worker token straight out of it. Anyone who
-- loaded the study page could have poisoned the dataset with fabricated
-- responses.
--
-- REVOKING FROM `public` IS NOT ENOUGH EITHER. Supabase's project bootstrap
-- runs `alter default privileges in schema public grant all on functions to
-- anon, authenticated, service_role`, so every function created here is
-- auto-granted to `anon` BY NAME. That explicit grant survives a revoke from
-- the PUBLIC pseudo-role — confirmed on 2026-07-28, when the first round of
-- revokes locked the tables but left claim_job and friends wide open. So each
-- function is revoked from public AND anon AND authenticated.
--
-- Revoking per-name rather than "all functions in schema public" on purpose —
-- the blanket form would also strip extension functions like gen_random_uuid().

revoke execute on function public.cfg(text) from public, anon, authenticated;
revoke execute on function public.claim_job(text, integer) from public, anon, authenticated;
revoke execute on function public.reap_expired_jobs(integer) from public, anon, authenticated;
revoke execute on function public.complete_job(text, text, text, jsonb) from public, anon, authenticated;
revoke execute on function public.fail_job_attempt(text, text, text, integer) from public, anon, authenticated;
revoke execute on function public.extend_lease(text, text, integer) from public, anon, authenticated;
revoke execute on function public.requeue_job(text, integer) from public, anon, authenticated;
revoke execute on function public.mark_submitted(text) from public, anon, authenticated;
revoke execute on function public.queue_stats() from public, anon, authenticated;
revoke execute on function public.job_state(public.study_jobs) from public, anon, authenticated;
revoke execute on function public.session_for_token(text) from public, anon, authenticated;
-- session_arm is the arm draw itself. Callable with the anon key it would let a
-- participant burn draws on other sessions, and — worse — it RETURNS the arm.
revoke execute on function public.session_arm(text) from public, anon, authenticated;
revoke execute on function public.assert_worker(text) from public, anon, authenticated;

-- The internal helpers and raw queue functions are reachable ONLY through the
-- SECURITY DEFINER wrappers above, which run as the owner and so do not need
-- the caller to hold EXECUTE. service_role keeps them for exports and ops.
grant execute on function public.claim_job(text, integer) to service_role;
grant execute on function public.reap_expired_jobs(integer) to service_role;
grant execute on function public.complete_job(text, text, text, jsonb) to service_role;
grant execute on function public.fail_job_attempt(text, text, text, integer) to service_role;
grant execute on function public.extend_lease(text, text, integer) to service_role;
grant execute on function public.requeue_job(text, integer) to service_role;
grant execute on function public.mark_submitted(text) to service_role;
grant execute on function public.queue_stats() to service_role;
grant execute on function public.session_arm(text) to service_role;

-- RLS on the generated data tables too. Without this the anon key could read
-- study_config, which is where the worker's shared secret lives.
alter table public.study_scenarios enable row level security;
alter table public.questionnaire_options enable row level security;
alter table public.study_config enable row level security;
"""


def main():
    print("-- GENERATED FILE — do not edit by hand.")
    print("-- Regenerate with: python tools/generate_supabase_app_sql.py > "
          "supabase_app_schema.sql")
    print("--")
    print("-- Run AFTER supabase_schema.sql and supabase_queue_schema.sql.")
    print("--")
    print(__doc__.strip().replace("\n", "\n-- ").join(["-- ", ""]))
    print()
    print(emit_scenarios())
    print()
    print(emit_questionnaire_options())
    print()
    print(emit_arm_domain())
    print()
    print(emit_debug_column())
    print()
    print(emit_config())
    print(RPCS.replace("__RANDOMIZED_ARRAY__", RANDOMIZED_ARRAY)
              .replace("__RANDOMIZED_COUNT__", str(len(RANDOMIZED_ARMS)))
              .replace("__SKIP_REVIEW_ARRAY__", SKIP_REVIEW_ARRAY)
              .replace("__DEBUG_CASE__", DEBUG_CASE))


if __name__ == "__main__":
    main()
