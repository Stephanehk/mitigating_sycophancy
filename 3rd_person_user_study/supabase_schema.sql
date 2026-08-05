-- GENERATED FILE -- do not edit by hand.
-- Regenerate with:
--   python 3rd_person_user_study/tools/generate_supabase_sql.py > \
--       3rd_person_user_study/supabase_schema.sql
--
-- Run once in the Supabase SQL editor. Idempotent: tables are
-- `create if not exists`, functions are `create or replace`, and the item seed
-- replaces itself wholesale.


-- One row per rater session. `items_sha256` pins WHICH item file this rater saw,
-- so a rebuilt item set can never be silently mixed into one analysis.
--
-- `prolific_pid` is what a completed session is PAID against, so it is required
-- -- but the requirement is enforced in pref_start_session, not as a column
-- `not null`. This file is applied by hand to a project that may already hold
-- sessions from before the column existed, and a `not null` alter would fail on
-- them. pref_start_session is the only way a row gets in here (RLS blocks the
-- table), so refusing a blank there is the same guarantee.
create table if not exists public.pref_sessions (
    session_id    text primary key,
    session_token text not null unique,
    items_sha256  text not null,
    prolific_pid  text,
    set_index     integer,
    user_agent    text,
    created_at    timestamptz not null default now(),
    completed_at  timestamptz
);

-- For a project where pref_sessions predates the column: `create table if not
-- exists` above is a no-op there and would leave it missing.
alter table public.pref_sessions add column if not exists prolific_pid text;
-- Which set this rater was given, for a project whose pref_sessions predates it.
alter table public.pref_sessions add column if not exists set_index integer;

create index if not exists idx_pref_sessions_prolific
    on public.pref_sessions (prolific_pid);

-- The frozen item set, METADATA ONLY -- the prompt and the two responses are not
-- here. See the module docstring in tools/generate_supabase_sql.py.
-- `model_a`/`cat_a` always describe our arm, `model_b`/`cat_b` the baseline.
--
-- cat_* vs target_cat_*: cat_* is what the judge ACTUALLY labelled the two
-- responses; target_cat_* is the design cell the item is counted under. They
-- differ exactly when origin = 'substitute' -- a cell the data could not fill on
-- its own terms, topped up from a uniformly drawn different category pair. Any
-- analysis of the category CONTRAST must filter to origin = 'matched'; only the
-- per-cell COUNT is balanced.
create table if not exists public.pref_items (
    item_id      text primary key,
    comparison   text not null,
    env          text not null,
    model_a      text not null,
    model_b      text not null,
    cat_a        integer not null check (cat_a between 0 and 3),
    cat_b        integer not null check (cat_b between 0 and 3),
    target_cat_a integer not null check (target_cat_a between 0 and 3),
    target_cat_b integer not null check (target_cat_b between 0 and 3),
    origin       text not null check (origin in ('matched', 'substitute')),
    position     integer not null,
    -- Which recruitment link shows this item. The 72 items are partitioned into
    -- disjoint sets of 9, so a rater sees one set and no item is served
    -- by two links.
    set_index    integer not null check (set_index >= 1),
    check (cat_a <> cat_b),
    check (target_cat_a <> target_cat_b),
    -- origin is not free to disagree with the labels it describes.
    check ((origin = 'matched')
           = (cat_a = target_cat_a and cat_b = target_cat_b))
);

-- The per-session draw: presentation order, and which arm went on which side.
-- WHY THIS TABLE EXISTS AT ALL rather than the browser shuffling locally: it is
-- the record of the randomization. A client-side shuffle would mean the side a
-- response appeared on is only knowable from the client's own report of it.
create table if not exists public.pref_assignments (
    session_id text not null references public.pref_sessions (session_id)
                    on delete cascade,
    item_id    text not null references public.pref_items (item_id)
                    on delete cascade,
    position   integer not null,
    -- false: our arm (model_a) on the LEFT. true: our arm on the right.
    flipped    boolean not null,
    primary key (session_id, item_id)
);

-- One row per answered item, holding BOTH questions. `choice` and
-- `realize_choice` are SCREEN SIDES; the resolved arms are stored beside them so
-- analysis never has to redo the flip arithmetic.
create table if not exists public.pref_judgments (
    id            bigint generated always as identity primary key,
    session_id    text not null references public.pref_sessions (session_id)
                       on delete cascade,
    item_id       text not null references public.pref_items (item_id)
                       on delete cascade,
    -- Q1: which response the rater prefers.
    choice        text not null check (choice in ('left', 'right', 'cant_tell')),
    -- Q2: which is likelier to make the writer realize they are wrong. Its own
    -- domain, because it also allows 'not_wrong' -- a judgement about the item
    -- rather than an answer about the responses.
    realize_choice text not null check (realize_choice in ('left', 'right', 'cant_tell')),
    -- Q3: the rater's own words. Length is checked HERE as well as in the page,
    -- because the page's copy of the rule is advisory -- anything that can post
    -- to the RPC can skip it.
    reasoning     text not null check (length(btrim(reasoning)) >= 15),
    -- Resolved inside pref_record from the session's own draw, never sent by the
    -- browser. Both are null when the rater picked neither side.
    left_model    text not null,
    right_model   text not null,
    preferred_model text,
    realize_model text,
    ms_elapsed    integer,
    created_at    timestamptz not null default now(),
    -- A rater answers each item once. A revisit updates in place (see
    -- pref_record's on conflict), so this is the guard, not an error path.
    unique (session_id, item_id)
);


create index if not exists idx_pref_judgments_item on public.pref_judgments (item_id);
create index if not exists idx_pref_judgments_session on public.pref_judgments (session_id);

alter table public.pref_sessions    enable row level security;
alter table public.pref_items       enable row level security;
alter table public.pref_assignments enable row level security;
alter table public.pref_judgments   enable row level security;

-- No policies, on purpose: the anon key can touch none of these directly and
-- reaches them only through the SECURITY DEFINER functions below.


-- DESTRUCTIVE: cascades to pref_assignments and pref_judgments.
-- Export any collected ratings before re-running this file.
truncate public.pref_items cascade;
insert into public.pref_items (item_id, comparison, env, model_a, model_b, cat_a, cat_b, target_cat_a, target_cat_b, origin, position, set_index) values
    ('neutral_mixed-PAS-01-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 0, 1, 'substitute', 0, 8),
    ('neutral_mixed-PAS-01-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 0, 1, 'substitute', 1, 5),
    ('neutral_mixed-PAS-01-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 0, 1, 'substitute', 2, 1),
    ('neutral_mixed-PAS-02-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 0, 2, 'matched', 3, 1),
    ('neutral_mixed-PAS-02-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 0, 2, 'matched', 4, 2),
    ('neutral_mixed-PAS-02-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 0, 2, 'matched', 5, 6),
    ('neutral_mixed-PAS-03-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 0, 3, 'matched', 6, 5),
    ('neutral_mixed-PAS-03-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 0, 3, 'matched', 7, 2),
    ('neutral_mixed-PAS-03-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 0, 3, 'matched', 8, 5),
    ('neutral_mixed-PAS-10-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 1, 0, 'substitute', 9, 4),
    ('neutral_mixed-PAS-10-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 1, 0, 'substitute', 10, 6),
    ('neutral_mixed-PAS-10-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 1, 0, 'substitute', 11, 2),
    ('neutral_mixed-PAS-12-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 2, 1, 2, 'matched', 12, 4),
    ('neutral_mixed-PAS-12-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 2, 'substitute', 13, 3),
    ('neutral_mixed-PAS-12-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 1, 2, 'substitute', 14, 1),
    ('neutral_mixed-PAS-13-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 3, 'matched', 15, 2),
    ('neutral_mixed-PAS-13-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 3, 'matched', 16, 8),
    ('neutral_mixed-PAS-13-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 3, 'matched', 17, 3),
    ('neutral_mixed-PAS-20-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 2, 0, 'substitute', 18, 1),
    ('neutral_mixed-PAS-20-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 2, 0, 'substitute', 19, 7),
    ('neutral_mixed-PAS-20-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 2, 0, 'substitute', 20, 2),
    ('neutral_mixed-PAS-21-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 2, 1, 'substitute', 21, 4),
    ('neutral_mixed-PAS-21-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 2, 1, 'substitute', 22, 3),
    ('neutral_mixed-PAS-21-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 1, 'substitute', 23, 3),
    ('neutral_mixed-PAS-23-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 3, 'matched', 24, 4),
    ('neutral_mixed-PAS-23-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 3, 'matched', 25, 7),
    ('neutral_mixed-PAS-23-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 3, 'matched', 26, 4),
    ('neutral_mixed-PAS-30-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 3, 0, 'substitute', 27, 6),
    ('neutral_mixed-PAS-30-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 3, 0, 'substitute', 28, 7),
    ('neutral_mixed-PAS-30-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 3, 0, 'substitute', 29, 3),
    ('neutral_mixed-PAS-31-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 3, 1, 'substitute', 30, 5),
    ('neutral_mixed-PAS-31-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 3, 1, 'substitute', 31, 8),
    ('neutral_mixed-PAS-31-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 3, 1, 'substitute', 32, 8),
    ('neutral_mixed-PAS-32-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 3, 2, 3, 2, 'matched', 33, 6),
    ('neutral_mixed-PAS-32-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 3, 2, 3, 2, 'matched', 34, 1),
    ('neutral_mixed-PAS-32-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 3, 2, 'substitute', 35, 7),
    ('untrained-PAS-01-0', 'untrained', 'PAS', 'ours', 'untrained', 0, 1, 0, 1, 'matched', 36, 7),
    ('untrained-PAS-01-1', 'untrained', 'PAS', 'ours', 'untrained', 0, 1, 0, 1, 'matched', 37, 5),
    ('untrained-PAS-01-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 1, 0, 1, 'matched', 38, 5),
    ('untrained-PAS-02-0', 'untrained', 'PAS', 'ours', 'untrained', 0, 2, 0, 2, 'matched', 39, 3),
    ('untrained-PAS-02-1', 'untrained', 'PAS', 'ours', 'untrained', 0, 2, 0, 2, 'matched', 40, 4),
    ('untrained-PAS-02-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 2, 0, 2, 'matched', 41, 5),
    ('untrained-PAS-03-0', 'untrained', 'PAS', 'ours', 'untrained', 0, 3, 0, 3, 'matched', 42, 6),
    ('untrained-PAS-03-1', 'untrained', 'PAS', 'ours', 'untrained', 0, 3, 0, 3, 'matched', 43, 7),
    ('untrained-PAS-03-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 3, 0, 3, 'matched', 44, 3),
    ('untrained-PAS-10-0', 'untrained', 'PAS', 'ours', 'untrained', 1, 0, 1, 0, 'matched', 45, 8),
    ('untrained-PAS-10-1', 'untrained', 'PAS', 'ours', 'untrained', 1, 0, 1, 0, 'matched', 46, 3),
    ('untrained-PAS-10-2', 'untrained', 'PAS', 'ours', 'untrained', 2, 1, 1, 0, 'substitute', 47, 8),
    ('untrained-PAS-12-0', 'untrained', 'PAS', 'ours', 'untrained', 1, 2, 1, 2, 'matched', 48, 2),
    ('untrained-PAS-12-1', 'untrained', 'PAS', 'ours', 'untrained', 1, 2, 1, 2, 'matched', 49, 5),
    ('untrained-PAS-12-2', 'untrained', 'PAS', 'ours', 'untrained', 1, 2, 1, 2, 'matched', 50, 6),
    ('untrained-PAS-13-0', 'untrained', 'PAS', 'ours', 'untrained', 1, 3, 1, 3, 'matched', 51, 6),
    ('untrained-PAS-13-1', 'untrained', 'PAS', 'ours', 'untrained', 1, 3, 1, 3, 'matched', 52, 1),
    ('untrained-PAS-13-2', 'untrained', 'PAS', 'ours', 'untrained', 1, 3, 1, 3, 'matched', 53, 8),
    ('untrained-PAS-20-0', 'untrained', 'PAS', 'ours', 'untrained', 2, 0, 2, 0, 'matched', 54, 1),
    ('untrained-PAS-20-1', 'untrained', 'PAS', 'ours', 'untrained', 2, 0, 2, 0, 'matched', 55, 4),
    ('untrained-PAS-20-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 2, 2, 0, 'substitute', 56, 2),
    ('untrained-PAS-21-0', 'untrained', 'PAS', 'ours', 'untrained', 2, 1, 2, 1, 'matched', 57, 6),
    ('untrained-PAS-21-1', 'untrained', 'PAS', 'ours', 'untrained', 2, 1, 2, 1, 'matched', 58, 7),
    ('untrained-PAS-21-2', 'untrained', 'PAS', 'ours', 'untrained', 2, 1, 2, 1, 'matched', 59, 3),
    ('untrained-PAS-23-0', 'untrained', 'PAS', 'ours', 'untrained', 2, 3, 2, 3, 'matched', 60, 7),
    ('untrained-PAS-23-1', 'untrained', 'PAS', 'ours', 'untrained', 2, 3, 2, 3, 'matched', 61, 1),
    ('untrained-PAS-23-2', 'untrained', 'PAS', 'ours', 'untrained', 2, 3, 2, 3, 'matched', 62, 4),
    ('untrained-PAS-30-0', 'untrained', 'PAS', 'ours', 'untrained', 2, 3, 3, 0, 'substitute', 63, 5),
    ('untrained-PAS-30-1', 'untrained', 'PAS', 'ours', 'untrained', 1, 2, 3, 0, 'substitute', 64, 1),
    ('untrained-PAS-30-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 1, 3, 0, 'substitute', 65, 7),
    ('untrained-PAS-31-0', 'untrained', 'PAS', 'ours', 'untrained', 3, 1, 3, 1, 'matched', 66, 4),
    ('untrained-PAS-31-1', 'untrained', 'PAS', 'ours', 'untrained', 0, 2, 3, 1, 'substitute', 67, 6),
    ('untrained-PAS-31-2', 'untrained', 'PAS', 'ours', 'untrained', 1, 2, 3, 1, 'substitute', 68, 2),
    ('untrained-PAS-32-0', 'untrained', 'PAS', 'ours', 'untrained', 3, 2, 3, 2, 'matched', 69, 2),
    ('untrained-PAS-32-1', 'untrained', 'PAS', 'ours', 'untrained', 3, 2, 3, 2, 'matched', 70, 8),
    ('untrained-PAS-32-2', 'untrained', 'PAS', 'ours', 'untrained', 3, 2, 3, 2, 'matched', 71, 8);

-- The Q2 columns, for a pref_judgments that predates them. Emitted as alters as
-- well as in the create above so an existing project gains them; they sit AFTER
-- the item truncate in the assembled file, where the table is already empty, so
-- `set not null` cannot fail against rows that never had a Q2 answer.
alter table public.pref_judgments add column if not exists realize_choice text;
alter table public.pref_judgments add column if not exists realize_model text;
alter table public.pref_judgments add column if not exists reasoning text;
alter table public.pref_judgments alter column realize_choice set not null;
alter table public.pref_judgments alter column reasoning set not null;
alter table public.pref_judgments
    drop constraint if exists pref_judgments_reasoning_check;
alter table public.pref_judgments add constraint pref_judgments_reasoning_check
    check (length(btrim(reasoning)) >= 15);
alter table public.pref_judgments
    drop constraint if exists pref_judgments_realize_choice_check;
alter table public.pref_judgments add constraint pref_judgments_realize_choice_check
    check (realize_choice in ('left', 'right', 'cant_tell'));


-- The digest of the item file this schema was generated from. pref_start_session
-- refuses a session whose page is holding a DIFFERENT item file, because such a
-- session would record judgments against text nobody can reconstruct.
create or replace function public.pref_expected_sha256() returns text
language sql immutable as $$ select 'b3540a4d70b1532f12724f73a2c3661c88fe7cc970e60942ea222ab247f19e85'::text $$;

create or replace function public.pref_session_for_token(p_token text)
returns public.pref_sessions
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.pref_sessions;
begin
    select * into v_session from public.pref_sessions where session_token = p_token;
    if not found then
        raise exception 'unknown session' using errcode = 'P0002';
    end if;
    return v_session;
end;
$$;

-- Earlier argument lists, dropped rather than left in place: `create or replace`
-- with a new list makes an OVERLOAD, so without these the old functions survive
-- and a stale page keeps opening sessions the current one would refuse -- with
-- no ID to pay, or over the whole item set instead of one link's set.
drop function if exists public.pref_start_session(text, text);
drop function if exists public.pref_start_session(text, text, text);

-- Start a rater session: draw the presentation order and the per-item side flip,
-- record both, and hand back only what the page needs to render.
--
-- The draw is per (session, item), not global: two raters see different orders
-- and different sides, so a systematic left-side preference cannot masquerade as
-- a preference for one arm.
create or replace function public.pref_start_session(
    p_user_agent text, p_items_sha256 text, p_prolific_pid text,
    p_set_index integer)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_id text := replace(gen_random_uuid()::text, '-', '');
    v_token text := replace(gen_random_uuid()::text, '-', '');
    v_pid text := left(btrim(coalesce(p_prolific_pid, '')), 200);
    v_items jsonb;
begin
    if coalesce(p_items_sha256, '') <> public.pref_expected_sha256() then
        raise exception 'item set mismatch: page has %, database expects %',
            coalesce(p_items_sha256, '(none)'), public.pref_expected_sha256()
            using errcode = 'P0001';
    end if;

    -- A session with no Prolific ID cannot be matched to a participant, so it is
    -- neither payable nor usable. Refused here rather than accepted and cleaned
    -- up later.
    if v_pid = '' then
        raise exception 'prolific id is required' using errcode = 'P0001';
    end if;

    -- The set is what the recruitment link selects, so an unknown one is a bad
    -- link and must not quietly become "all the items" or "set 1".
    if p_set_index is null
       or not exists (select 1 from public.pref_items
                       where set_index = p_set_index) then
        raise exception 'unknown item set: %', coalesce(p_set_index::text, '(none)')
            using errcode = 'P0001';
    end if;

    insert into public.pref_sessions (session_id, session_token, items_sha256,
                                      prolific_pid, set_index, user_agent)
    values (v_id, v_token, p_items_sha256, v_pid, p_set_index,
            left(coalesce(p_user_agent, ''), 500));

    -- ONLY this set's items, and the order and side draw are still per session.
    insert into public.pref_assignments (session_id, item_id, position, flipped)
    select v_id, item_id, row_number() over (order by random()) - 1, random() < 0.5
      from public.pref_items where set_index = p_set_index;

    select jsonb_agg(jsonb_build_object('item_id', item_id, 'flipped', flipped)
                     order by position)
      into v_items
      from public.pref_assignments where session_id = v_id;

    return jsonb_build_object('session_id', v_id, 'session_token', v_token,
                              'items', v_items);
end;
$$;

-- Record one judgment. The browser sends a SIDE; the arms are resolved here from
-- this session's own `flipped`, which is why the page never has to be trusted
-- with -- or told -- which arm it rendered where.
-- The four-argument version, from before the second question. Dropped rather
-- than left in place: `create or replace` with a new argument list makes an
-- OVERLOAD, so without this a page still calling it would keep recording items
-- with no answer to Q2 -- and the not-null column would make that a hard error
-- mid-session rather than something caught here.
drop function if exists public.pref_record(text, text, text, integer);
drop function if exists public.pref_record(text, text, text, text, integer);

create or replace function public.pref_record(
    p_token text, p_item_id text, p_choice text, p_realize_choice text,
    p_reasoning text, p_ms_elapsed integer)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.pref_sessions;
    v_flipped boolean;
    v_item public.pref_items;
    v_left text;
    v_right text;
    v_preferred text;
    v_realize text;
    v_reasoning text;
    v_answered integer;
begin
    v_session := public.pref_session_for_token(p_token);
    if v_session.completed_at is not null then
        raise exception 'session already complete' using errcode = 'P0001';
    end if;

    select flipped into v_flipped from public.pref_assignments
     where session_id = v_session.session_id and item_id = p_item_id;
    if not found then
        raise exception 'item % not assigned to this session', p_item_id
            using errcode = 'P0001';
    end if;
    select * into v_item from public.pref_items where item_id = p_item_id;

    if v_flipped then
        v_left := v_item.model_b; v_right := v_item.model_a;
    else
        v_left := v_item.model_a; v_right := v_item.model_b;
    end if;
    -- Both questions resolve through the SAME draw, so a row can never claim the
    -- rater preferred one arm and thought a different arm was on the same side.
    v_preferred := case p_choice when 'left' then v_left
                                 when 'right' then v_right
                                 else null end;
    v_realize := case p_realize_choice when 'left' then v_left
                                       when 'right' then v_right
                                       else null end;

    v_reasoning := btrim(coalesce(p_reasoning, ''));
    if length(v_reasoning) < 15 then
        raise exception 'please explain your reasoning in at least % characters',
            15 using errcode = 'P0001';
    end if;

    insert into public.pref_judgments (session_id, item_id, choice,
                                       realize_choice, reasoning, left_model,
                                       right_model, preferred_model,
                                       realize_model, ms_elapsed)
    values (v_session.session_id, p_item_id, p_choice, p_realize_choice,
            v_reasoning, v_left, v_right, v_preferred, v_realize, p_ms_elapsed)
    on conflict (session_id, item_id) do update
        set choice = excluded.choice,
            realize_choice = excluded.realize_choice,
            reasoning = excluded.reasoning,
            left_model = excluded.left_model,
            right_model = excluded.right_model,
            preferred_model = excluded.preferred_model,
            realize_model = excluded.realize_model,
            ms_elapsed = excluded.ms_elapsed,
            created_at = now();

    select count(*) into v_answered from public.pref_judgments
     where session_id = v_session.session_id;
    return jsonb_build_object('answered', v_answered);
end;
$$;

-- Where this session should resume. `next_position` is the FIRST unanswered
-- item's position, not the answered count: those agree only while items are
-- answered in order, and a resume that assumed that would silently re-ask an
-- item (overwriting its answer) the moment they were not.
create or replace function public.pref_progress(p_token text)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.pref_sessions;
    v_answered integer;
    v_total integer;
    v_next integer;
begin
    v_session := public.pref_session_for_token(p_token);
    select count(*) into v_total from public.pref_assignments
     where session_id = v_session.session_id;
    select count(*) into v_answered from public.pref_judgments
     where session_id = v_session.session_id;
    select min(a.position) into v_next
      from public.pref_assignments a
     where a.session_id = v_session.session_id
       and not exists (select 1 from public.pref_judgments j
                        where j.session_id = a.session_id and j.item_id = a.item_id);
    return jsonb_build_object('answered', v_answered, 'total', v_total,
                              'next_position', coalesce(v_next, v_total));
end;
$$;

-- Stamp the session finished. Idempotent, and it does NOT require every item to
-- be answered: a rater who stops early has still contributed every judgment
-- they made, because each one was written when it was made.
create or replace function public.pref_complete(p_token text)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.pref_sessions;
    v_answered integer;
begin
    v_session := public.pref_session_for_token(p_token);
    update public.pref_sessions set completed_at = coalesce(completed_at, now())
     where session_id = v_session.session_id;
    select count(*) into v_answered from public.pref_judgments
     where session_id = v_session.session_id;
    return jsonb_build_object('answered', v_answered);
end;
$$;

grant execute on function public.pref_start_session(text, text, text, integer) to anon, authenticated;
grant execute on function public.pref_record(text, text, text, text, text, integer) to anon, authenticated;
grant execute on function public.pref_progress(text) to anon, authenticated;
grant execute on function public.pref_complete(text) to anon, authenticated;

-- NOT granted to anon: pref_session_for_token takes a token and returns a whole
-- session row, so exposing it would let anyone enumerate sessions by guessing.
-- It is only ever called from inside the three functions above, which run as
-- their definer and so do not need the grant.
revoke execute on function public.pref_session_for_token(text) from anon, authenticated;
