-- GENERATED FILE — do not edit by hand.
-- Regenerate with: python tools/generate_supabase_app_sql.py > supabase_app_schema.sql
--
-- Run AFTER supabase_schema.sql and supabase_queue_schema.sql.
--
-- Generate supabase_app_schema.sql — the RPCs the browser and worker call.
-- 
-- WHY GENERATED RATHER THAN HAND-WRITTEN. Dropping the API server moves four
-- things out of Python and into Postgres: the arm draw, the user-turn
-- composition, the free-text length checks and the answer-domain validation. The
-- last of those needs every questionnaire item's allowed values, and hand-copying
-- 42 response keys into SQL would make questionnaires.py stop being the source of
-- truth the first time someone edited one and forgot the other. So the data
-- tables are emitted from questionnaires.py / scenarios.py / config.py, and
-- regenerating is a one-liner:
-- 
--     python tools/generate_supabase_app_sql.py > supabase_app_schema.sql
-- 
-- AUTHENTICATION MODEL. There is no server to hold a privileged key, so nothing
-- deployed holds one. Both the browser and the GPU worker use the ANON key, which
-- is public and can do nothing on its own — RLS blocks every table. Each RPC is
-- SECURITY DEFINER and authenticates its own caller:
-- 
--     participants  an opaque session_token, issued by start_session
--     the worker    a shared secret, compared against study_config inside the RPC
-- 
-- service_role is then only ever used from a developer machine, for exports.

-- Generated from scenarios.py. Only what the queue needs: the key
-- (validated on start_response) and topic_title (stored on the job).
create table if not exists public.study_scenarios (
    key         text primary key,
    topic_title text not null,
    position    integer not null
);
truncate public.study_scenarios;
insert into public.study_scenarios (key, topic_title, position) values
    ('relationship', 'Relationship Boundaries', 0),
    ('others', 'Involving Yourself in Someone Else’s Business', 1),
    ('exclusion', 'Excluding Someone', 2),
    ('uncomfortable', 'Making Someone Uncomfortable', 3);

-- Generated from questionnaires.py: 284 (response_key, value) pairs across 42 response keys.
create table if not exists public.questionnaire_options (
    response_key text not null,
    question_id  text not null,
    value        integer not null,
    primary key (response_key, value)
);
truncate public.questionnaire_options;
insert into public.questionnaire_options (response_key, question_id, value) values
    ('fk_1', 'fk', 1),
    ('fk_1', 'fk', 2),
    ('fk_1', 'fk', 3),
    ('fk_1', 'fk', 4),
    ('fk_1', 'fk', 5),
    ('fk_1', 'fk', 6),
    ('fk_1', 'fk', 7),
    ('fk_2', 'fk', 1),
    ('fk_2', 'fk', 2),
    ('fk_2', 'fk', 3),
    ('fk_2', 'fk', 4),
    ('fk_2', 'fk', 5),
    ('fk_2', 'fk', 6),
    ('fk_2', 'fk', 7),
    ('fk_3', 'fk', 1),
    ('fk_3', 'fk', 2),
    ('fk_3', 'fk', 3),
    ('fk_3', 'fk', 4),
    ('fk_3', 'fk', 5),
    ('fk_3', 'fk', 6),
    ('fk_3', 'fk', 7),
    ('fk_4', 'fk', 1),
    ('fk_4', 'fk', 2),
    ('fk_4', 'fk', 3),
    ('fk_4', 'fk', 4),
    ('fk_4', 'fk', 5),
    ('fk_4', 'fk', 6),
    ('fk_4', 'fk', 7),
    ('fk_8', 'fk', 1),
    ('fk_8', 'fk', 2),
    ('fk_8', 'fk', 3),
    ('fk_8', 'fk', 4),
    ('fk_8', 'fk', 5),
    ('fk_8', 'fk', 6),
    ('fk_8', 'fk', 7),
    ('fk_9', 'fk', 1),
    ('fk_9', 'fk', 2),
    ('fk_9', 'fk', 3),
    ('fk_9', 'fk', 4),
    ('fk_9', 'fk', 5),
    ('fk_9', 'fk', 6),
    ('fk_9', 'fk', 7),
    ('MDMT_1', 'MDMT', 1),
    ('MDMT_1', 'MDMT', 2),
    ('MDMT_1', 'MDMT', 3),
    ('MDMT_1', 'MDMT', 4),
    ('MDMT_1', 'MDMT', 5),
    ('MDMT_1', 'MDMT', 6),
    ('MDMT_1', 'MDMT', 7),
    ('MDMT_2', 'MDMT', 1),
    ('MDMT_2', 'MDMT', 2),
    ('MDMT_2', 'MDMT', 3),
    ('MDMT_2', 'MDMT', 4),
    ('MDMT_2', 'MDMT', 5),
    ('MDMT_2', 'MDMT', 6),
    ('MDMT_2', 'MDMT', 7),
    ('MDMT_3', 'MDMT', 1),
    ('MDMT_3', 'MDMT', 2),
    ('MDMT_3', 'MDMT', 3),
    ('MDMT_3', 'MDMT', 4),
    ('MDMT_3', 'MDMT', 5),
    ('MDMT_3', 'MDMT', 6),
    ('MDMT_3', 'MDMT', 7),
    ('MDMT_4', 'MDMT', 1),
    ('MDMT_4', 'MDMT', 2),
    ('MDMT_4', 'MDMT', 3),
    ('MDMT_4', 'MDMT', 4),
    ('MDMT_4', 'MDMT', 5),
    ('MDMT_4', 'MDMT', 6),
    ('MDMT_4', 'MDMT', 7),
    ('MDMT_10', 'MDMT', 1),
    ('MDMT_10', 'MDMT', 2),
    ('MDMT_10', 'MDMT', 3),
    ('MDMT_10', 'MDMT', 4),
    ('MDMT_10', 'MDMT', 5),
    ('MDMT_10', 'MDMT', 6),
    ('MDMT_10', 'MDMT', 7),
    ('MDMT_11', 'MDMT', 1),
    ('MDMT_11', 'MDMT', 2),
    ('MDMT_11', 'MDMT', 3),
    ('MDMT_11', 'MDMT', 4),
    ('MDMT_11', 'MDMT', 5),
    ('MDMT_11', 'MDMT', 6),
    ('MDMT_11', 'MDMT', 7),
    ('MDMT_8', 'MDMT', 1),
    ('MDMT_8', 'MDMT', 2),
    ('MDMT_8', 'MDMT', 3),
    ('MDMT_8', 'MDMT', 4),
    ('MDMT_8', 'MDMT', 5),
    ('MDMT_8', 'MDMT', 6),
    ('MDMT_8', 'MDMT', 7),
    ('MDMT_12', 'MDMT', 1),
    ('MDMT_12', 'MDMT', 2),
    ('MDMT_12', 'MDMT', 3),
    ('MDMT_12', 'MDMT', 4),
    ('MDMT_12', 'MDMT', 5),
    ('MDMT_12', 'MDMT', 6),
    ('MDMT_12', 'MDMT', 7),
    ('MDMT_13', 'MDMT', 1),
    ('MDMT_13', 'MDMT', 2),
    ('MDMT_13', 'MDMT', 3),
    ('MDMT_13', 'MDMT', 4),
    ('MDMT_13', 'MDMT', 5),
    ('MDMT_13', 'MDMT', 6),
    ('MDMT_13', 'MDMT', 7),
    ('MDMT_14', 'MDMT', 1),
    ('MDMT_14', 'MDMT', 2),
    ('MDMT_14', 'MDMT', 3),
    ('MDMT_14', 'MDMT', 4),
    ('MDMT_14', 'MDMT', 5),
    ('MDMT_14', 'MDMT', 6),
    ('MDMT_14', 'MDMT', 7),
    ('response_objectivity', 'response_objectivity', 1),
    ('response_objectivity', 'response_objectivity', 2),
    ('response_objectivity', 'response_objectivity', 3),
    ('response_objectivity', 'response_objectivity', 4),
    ('response_objectivity', 'response_objectivity', 5),
    ('response_objectivity', 'response_objectivity', 7),
    ('response_objectivity', 'response_objectivity', 8),
    ('response_quality', 'response_quality', 1),
    ('response_quality', 'response_quality', 2),
    ('response_quality', 'response_quality', 3),
    ('response_quality', 'response_quality', 4),
    ('response_quality', 'response_quality', 5),
    ('response_quality', 'response_quality', 7),
    ('response_quality', 'response_quality', 8),
    ('rightorwrong', 'rightorwrong', 7),
    ('rightorwrong', 'rightorwrong', 14),
    ('rightorwrong', 'rightorwrong', 15),
    ('rightorwrong', 'rightorwrong', 16),
    ('rightorwrong', 'rightorwrong', 17),
    ('rightorwrong', 'rightorwrong', 18),
    ('rightorwrong', 'rightorwrong', 19),
    ('repair_1', 'repair', 1),
    ('repair_1', 'repair', 2),
    ('repair_1', 'repair', 3),
    ('repair_1', 'repair', 4),
    ('repair_1', 'repair', 5),
    ('repair_1', 'repair', 6),
    ('repair_1', 'repair', 7),
    ('repair_2', 'repair', 1),
    ('repair_2', 'repair', 2),
    ('repair_2', 'repair', 3),
    ('repair_2', 'repair', 4),
    ('repair_2', 'repair', 5),
    ('repair_2', 'repair', 6),
    ('repair_2', 'repair', 7),
    ('repair_3', 'repair', 1),
    ('repair_3', 'repair', 2),
    ('repair_3', 'repair', 3),
    ('repair_3', 'repair', 4),
    ('repair_3', 'repair', 5),
    ('repair_3', 'repair', 6),
    ('repair_3', 'repair', 7),
    ('return_likelihood', 'return_likelihood', 1),
    ('return_likelihood', 'return_likelihood', 2),
    ('return_likelihood', 'return_likelihood', 3),
    ('return_likelihood', 'return_likelihood', 4),
    ('return_likelihood', 'return_likelihood', 5),
    ('return_likelihood', 'return_likelihood', 6),
    ('return_likelihood', 'return_likelihood', 7),
    ('attention_check_1', 'attention_check', 1),
    ('attention_check_1', 'attention_check', 2),
    ('attention_check_1', 'attention_check', 3),
    ('attention_check_1', 'attention_check', 4),
    ('attention_check_1', 'attention_check', 5),
    ('attention_check_1', 'attention_check', 6),
    ('attention_check_2', 'attention_check', 1),
    ('attention_check_2', 'attention_check', 2),
    ('attention_check_2', 'attention_check', 3),
    ('attention_check_2', 'attention_check', 4),
    ('attention_check_2', 'attention_check', 5),
    ('attention_check_2', 'attention_check', 6),
    ('attention_check_3', 'attention_check', 1),
    ('attention_check_3', 'attention_check', 2),
    ('attention_check_3', 'attention_check', 3),
    ('attention_check_3', 'attention_check', 4),
    ('attention_check_3', 'attention_check', 5),
    ('attention_check_3', 'attention_check', 6),
    ('frequency', 'frequency', 1),
    ('frequency', 'frequency', 2),
    ('frequency', 'frequency', 3),
    ('frequency', 'frequency', 4),
    ('frequency', 'frequency', 5),
    ('AIAS_1', 'AIAS', 1),
    ('AIAS_1', 'AIAS', 2),
    ('AIAS_1', 'AIAS', 3),
    ('AIAS_1', 'AIAS', 4),
    ('AIAS_1', 'AIAS', 5),
    ('AIAS_1', 'AIAS', 6),
    ('AIAS_1', 'AIAS', 7),
    ('AIAS_1', 'AIAS', 8),
    ('AIAS_1', 'AIAS', 9),
    ('AIAS_1', 'AIAS', 10),
    ('AIAS_2', 'AIAS', 1),
    ('AIAS_2', 'AIAS', 2),
    ('AIAS_2', 'AIAS', 3),
    ('AIAS_2', 'AIAS', 4),
    ('AIAS_2', 'AIAS', 5),
    ('AIAS_2', 'AIAS', 6),
    ('AIAS_2', 'AIAS', 7),
    ('AIAS_2', 'AIAS', 8),
    ('AIAS_2', 'AIAS', 9),
    ('AIAS_2', 'AIAS', 10),
    ('AIAS_3', 'AIAS', 1),
    ('AIAS_3', 'AIAS', 2),
    ('AIAS_3', 'AIAS', 3),
    ('AIAS_3', 'AIAS', 4),
    ('AIAS_3', 'AIAS', 5),
    ('AIAS_3', 'AIAS', 6),
    ('AIAS_3', 'AIAS', 7),
    ('AIAS_3', 'AIAS', 8),
    ('AIAS_3', 'AIAS', 9),
    ('AIAS_3', 'AIAS', 10),
    ('AIAS_4', 'AIAS', 1),
    ('AIAS_4', 'AIAS', 2),
    ('AIAS_4', 'AIAS', 3),
    ('AIAS_4', 'AIAS', 4),
    ('AIAS_4', 'AIAS', 5),
    ('AIAS_4', 'AIAS', 6),
    ('AIAS_4', 'AIAS', 7),
    ('AIAS_4', 'AIAS', 8),
    ('AIAS_4', 'AIAS', 9),
    ('AIAS_4', 'AIAS', 10),
    ('AIAS_5', 'AIAS', 1),
    ('AIAS_5', 'AIAS', 2),
    ('AIAS_5', 'AIAS', 3),
    ('AIAS_5', 'AIAS', 4),
    ('AIAS_5', 'AIAS', 5),
    ('AIAS_5', 'AIAS', 6),
    ('AIAS_5', 'AIAS', 7),
    ('AIAS_5', 'AIAS', 8),
    ('AIAS_5', 'AIAS', 9),
    ('AIAS_5', 'AIAS', 10),
    ('big_five_1', 'big_five', 1),
    ('big_five_1', 'big_five', 2),
    ('big_five_1', 'big_five', 3),
    ('big_five_1', 'big_five', 4),
    ('big_five_1', 'big_five', 5),
    ('big_five_2', 'big_five', 1),
    ('big_five_2', 'big_five', 2),
    ('big_five_2', 'big_five', 3),
    ('big_five_2', 'big_five', 4),
    ('big_five_2', 'big_five', 5),
    ('big_five_3', 'big_five', 1),
    ('big_five_3', 'big_five', 2),
    ('big_five_3', 'big_five', 3),
    ('big_five_3', 'big_five', 4),
    ('big_five_3', 'big_five', 5),
    ('big_five_4', 'big_five', 1),
    ('big_five_4', 'big_five', 2),
    ('big_five_4', 'big_five', 3),
    ('big_five_4', 'big_five', 4),
    ('big_five_4', 'big_five', 5),
    ('big_five_5', 'big_five', 1),
    ('big_five_5', 'big_five', 2),
    ('big_five_5', 'big_five', 3),
    ('big_five_5', 'big_five', 4),
    ('big_five_5', 'big_five', 5),
    ('big_five_6', 'big_five', 1),
    ('big_five_6', 'big_five', 2),
    ('big_five_6', 'big_five', 3),
    ('big_five_6', 'big_five', 4),
    ('big_five_6', 'big_five', 5),
    ('big_five_7', 'big_five', 1),
    ('big_five_7', 'big_five', 2),
    ('big_five_7', 'big_five', 3),
    ('big_five_7', 'big_five', 4),
    ('big_five_7', 'big_five', 5),
    ('big_five_8', 'big_five', 1),
    ('big_five_8', 'big_five', 2),
    ('big_five_8', 'big_five', 3),
    ('big_five_8', 'big_five', 4),
    ('big_five_8', 'big_five', 5),
    ('big_five_9', 'big_five', 1),
    ('big_five_9', 'big_five', 2),
    ('big_five_9', 'big_five', 3),
    ('big_five_9', 'big_five', 4),
    ('big_five_9', 'big_five', 5),
    ('big_five_10', 'big_five', 1),
    ('big_five_10', 'big_five', 2),
    ('big_five_10', 'big_five', 3),
    ('big_five_10', 'big_five', 4),
    ('big_five_10', 'big_five', 5);

-- Generated from config.py. Thresholds the RPCs enforce.
create table if not exists public.study_config (
    key   text primary key,
    value text not null
);
insert into public.study_config (key, value) values ('min_describe_chars', '20')
    on conflict (key) do update set value = excluded.value;
insert into public.study_config (key, value) values ('min_perspective_chars', '50')
    on conflict (key) do update set value = excluded.value;
insert into public.study_config (key, value) values ('max_text_chars', '4000')
    on conflict (key) do update set value = excluded.value;
insert into public.study_config (key, value) values ('job_deadline_seconds', '420')
    on conflict (key) do update set value = excluded.value;
insert into public.study_config (key, value) values ('job_lease_seconds', '180')
    on conflict (key) do update set value = excluded.value;
insert into public.study_config (key, value) values ('job_max_attempts', '4')
    on conflict (key) do update set value = excluded.value;
insert into public.study_config (key, value) values ('max_manual_retries', '1')
    on conflict (key) do update set value = excluded.value;
insert into public.study_config (key, value) values ('client_poll_interval_ms', '1200')
    on conflict (key) do update set value = excluded.value;
insert into public.study_config (key, value) values ('client_slow_notice_ms', '12000')
    on conflict (key) do update set value = excluded.value;

-- The worker's shared secret. NOT generated — set it yourself, once:
--   insert into public.study_config (key, value)
--   values ('worker_token', '<the same secret the worker exports>')
--   on conflict (key) do update set value = excluded.value;

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
create or replace function public.start_session(p_user_agent text)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_id text := replace(gen_random_uuid()::text, '-', '');
    v_token text := replace(gen_random_uuid()::text, '-', '');
    v_order jsonb;
begin
    -- Randomized per participant, drawn here so the order is recorded even
    -- though the browser renders the pages.
    select jsonb_agg(key order by random()) into v_order from public.study_scenarios;
    insert into public.study_sessions (session_id, session_token, screening_order, user_agent)
    values (v_id, v_token, v_order, left(coalesce(p_user_agent, ''), 500));
    return jsonb_build_object('session_id', v_id, 'session_token', v_token,
                              'screening_order', v_order, 'has_job', false,
                              'submitted', false);
end;
$$;

-- Enqueue the generation from the participant's two free-text answers.
--
-- THE ARM IS DRAWN HERE AND NEVER RETURNED. That is the point of doing this in
-- the database rather than the browser: a client that could pick its own arm
-- could re-roll until it got the one it wanted, which would corrupt every DV.
--
-- IDEMPOTENT, AND THAT IS A DATA-INTEGRITY RULE. A session gets one job,
-- forever. Re-posting with different text returns the existing job unchanged,
-- so the response cannot be regenerated until the participant likes it.
create or replace function public.start_response(
    p_token text, p_scenario_key text, p_description text, p_perspective text)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.study_sessions;
    v_job public.study_jobs;
    v_topic text;
    v_arm text;
    v_desc text := btrim(coalesce(p_description, ''));
    v_persp text := btrim(coalesce(p_perspective, ''));
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
    if length(v_desc) < cfg('min_describe_chars') then
        raise exception 'Please write a bit more — at least % characters.',
            cfg('min_describe_chars') using errcode = 'P0001';
    end if;
    if length(v_persp) < cfg('min_perspective_chars') then
        raise exception 'Please write a bit more — at least % characters.',
            cfg('min_perspective_chars') using errcode = 'P0001';
    end if;
    if length(v_desc) > cfg('max_text_chars') or length(v_persp) > cfg('max_text_chars') then
        raise exception 'Your answer is too long (limit % characters).',
            cfg('max_text_chars') using errcode = 'P0001';
    end if;

    -- Drawn at the moment of eligibility, so screened-out participants never
    -- consume an arm slot. Sticky: the early return above means this line runs
    -- at most once per session.
    v_arm := coalesce(v_session.arm, (array['ours', 'untrained', 'syco', 'neutral'])[1 + floor(random() * 4)]);
    if v_session.arm is distinct from v_arm then
        update public.study_sessions set arm = v_arm, updated_at = now()
        where session_id = v_session.session_id;
    end if;

    insert into public.study_jobs (job_id, session_id, arm, topic_title, user_text,
                                   status, deadline_at)
    values (replace(gen_random_uuid()::text, '-', ''), v_session.session_id, v_arm,
            v_topic,
            -- prompts.build_user_turn: the participant's own account of their
            -- incident, then their side of it, joined by a blank line. NO
            -- vignette text — it is a canned example, and prepending it once
            -- made the model answer about the wrong story.
            v_desc || E'\n\n' || v_persp,
            'pending', now() + make_interval(secs => cfg('job_deadline_seconds')))
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

    insert into public.study_submissions (
        session_id, arm, job_id, topic_title, user_text, shown_response,
        shown_response_chars, stages, worker_attempts, manual_retries,
        prolific_pid, consent, eligibility, selected_scenario, screening_order,
        screening, perspective_text, additional_thoughts, answers, missing_keys,
        client_timings, payload, session_created_at, submitted_at)
    values (
        v_session.session_id, v_session.arm, v_job.job_id, v_job.topic_title,
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

grant execute on function public.start_session(text) to anon, authenticated;
grant execute on function public.start_response(text, text, text, text) to anon, authenticated;
grant execute on function public.response_status(text) to anon, authenticated;
grant execute on function public.retry_response(text) to anon, authenticated;
grant execute on function public.submit_study(text, jsonb) to anon, authenticated;
grant execute on function public.worker_claim(text, text) to anon, authenticated;
grant execute on function public.worker_heartbeat(text, text, text) to anon, authenticated;
grant execute on function public.worker_complete(text, text, text, text, jsonb) to anon, authenticated;
grant execute on function public.worker_fail(text, text, text, text) to anon, authenticated;

-- Deliberately NOT granted to anon: claim_job, complete_job, fail_job_attempt,
-- extend_lease, requeue_job, mark_submitted, reap_expired_jobs, queue_stats.
-- Those take no token and are only reachable through the wrappers above.

