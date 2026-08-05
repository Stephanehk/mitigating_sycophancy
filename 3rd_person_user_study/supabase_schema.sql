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
    user_agent    text,
    created_at    timestamptz not null default now(),
    completed_at  timestamptz
);

-- For a project where pref_sessions predates the column: `create table if not
-- exists` above is a no-op there and would leave it missing.
alter table public.pref_sessions add column if not exists prolific_pid text;

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

-- One row per answered item. `choice` is a SCREEN SIDE; the resolved arms are
-- stored beside it so analysis never has to redo the flip arithmetic.
create table if not exists public.pref_judgments (
    id            bigint generated always as identity primary key,
    session_id    text not null references public.pref_sessions (session_id)
                       on delete cascade,
    item_id       text not null references public.pref_items (item_id)
                       on delete cascade,
    choice        text not null check (choice in ('left', 'right', 'same', 'cant_tell')),
    -- Resolved inside pref_record from the session's own draw, never sent by the
    -- browser. `preferred_model` is null for 'same' and 'cant_tell'.
    left_model    text not null,
    right_model   text not null,
    preferred_model text,
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
insert into public.pref_items (item_id, comparison, env, model_a, model_b, cat_a, cat_b, target_cat_a, target_cat_b, origin, position) values
    ('neutral_mixed-OEQ-01-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 1, 0, 1, 'matched', 0),
    ('neutral_mixed-OEQ-01-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 1, 0, 1, 'matched', 1),
    ('neutral_mixed-OEQ-01-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 1, 0, 1, 'matched', 2),
    ('neutral_mixed-OEQ-02-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 2, 0, 2, 'matched', 3),
    ('neutral_mixed-OEQ-02-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 2, 0, 2, 'matched', 4),
    ('neutral_mixed-OEQ-02-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 2, 0, 2, 'matched', 5),
    ('neutral_mixed-OEQ-03-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 3, 0, 3, 'matched', 6),
    ('neutral_mixed-OEQ-03-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 3, 0, 3, 'matched', 7),
    ('neutral_mixed-OEQ-03-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 3, 0, 3, 'matched', 8),
    ('neutral_mixed-OEQ-10-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 0, 1, 0, 'matched', 9),
    ('neutral_mixed-OEQ-10-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 0, 1, 0, 'matched', 10),
    ('neutral_mixed-OEQ-10-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 0, 1, 0, 'matched', 11),
    ('neutral_mixed-OEQ-12-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 2, 1, 2, 'matched', 12),
    ('neutral_mixed-OEQ-12-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 2, 1, 2, 'matched', 13),
    ('neutral_mixed-OEQ-12-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 2, 1, 2, 'matched', 14),
    ('neutral_mixed-OEQ-13-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 3, 1, 3, 'matched', 15),
    ('neutral_mixed-OEQ-13-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 3, 1, 3, 'matched', 16),
    ('neutral_mixed-OEQ-13-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 3, 1, 3, 'matched', 17),
    ('neutral_mixed-OEQ-20-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 0, 2, 0, 'matched', 18),
    ('neutral_mixed-OEQ-20-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 0, 2, 0, 'matched', 19),
    ('neutral_mixed-OEQ-20-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 3, 2, 0, 'substitute', 20),
    ('neutral_mixed-OEQ-21-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 1, 2, 1, 'matched', 21),
    ('neutral_mixed-OEQ-21-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 1, 2, 1, 'matched', 22),
    ('neutral_mixed-OEQ-21-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 1, 2, 1, 'matched', 23),
    ('neutral_mixed-OEQ-23-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 3, 2, 3, 'matched', 24),
    ('neutral_mixed-OEQ-23-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 3, 2, 3, 'matched', 25),
    ('neutral_mixed-OEQ-23-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 3, 2, 3, 'matched', 26),
    ('neutral_mixed-OEQ-30-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 3, 0, 3, 0, 'matched', 27),
    ('neutral_mixed-OEQ-30-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 0, 1, 3, 0, 'substitute', 28),
    ('neutral_mixed-OEQ-30-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 3, 2, 3, 0, 'substitute', 29),
    ('neutral_mixed-OEQ-31-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 1, 2, 3, 1, 'substitute', 30),
    ('neutral_mixed-OEQ-31-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 3, 3, 1, 'substitute', 31),
    ('neutral_mixed-OEQ-31-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 2, 1, 3, 1, 'substitute', 32),
    ('neutral_mixed-OEQ-32-0', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 3, 2, 3, 2, 'matched', 33),
    ('neutral_mixed-OEQ-32-1', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 3, 2, 3, 2, 'matched', 34),
    ('neutral_mixed-OEQ-32-2', 'neutral_mixed', 'OEQ', 'ours', 'neutral_mixed', 3, 2, 3, 2, 'matched', 35),
    ('neutral_mixed-PAS-01-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 0, 1, 'substitute', 36),
    ('neutral_mixed-PAS-01-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 0, 1, 'substitute', 37),
    ('neutral_mixed-PAS-01-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 0, 1, 'substitute', 38),
    ('neutral_mixed-PAS-02-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 0, 2, 'matched', 39),
    ('neutral_mixed-PAS-02-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 0, 2, 'matched', 40),
    ('neutral_mixed-PAS-02-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 2, 0, 2, 'matched', 41),
    ('neutral_mixed-PAS-03-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 0, 3, 'matched', 42),
    ('neutral_mixed-PAS-03-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 0, 3, 'matched', 43),
    ('neutral_mixed-PAS-03-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 0, 3, 'matched', 44),
    ('neutral_mixed-PAS-10-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 1, 0, 'substitute', 45),
    ('neutral_mixed-PAS-10-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 1, 0, 'substitute', 46),
    ('neutral_mixed-PAS-10-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 0, 'substitute', 47),
    ('neutral_mixed-PAS-12-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 2, 1, 2, 'matched', 48),
    ('neutral_mixed-PAS-12-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 2, 'substitute', 49),
    ('neutral_mixed-PAS-12-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 2, 'substitute', 50),
    ('neutral_mixed-PAS-13-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 3, 'matched', 51),
    ('neutral_mixed-PAS-13-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 3, 'matched', 52),
    ('neutral_mixed-PAS-13-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 1, 3, 'matched', 53),
    ('neutral_mixed-PAS-20-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 0, 'substitute', 54),
    ('neutral_mixed-PAS-20-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 0, 'substitute', 55),
    ('neutral_mixed-PAS-20-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 0, 'substitute', 56),
    ('neutral_mixed-PAS-21-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 2, 1, 'substitute', 57),
    ('neutral_mixed-PAS-21-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 2, 1, 'substitute', 58),
    ('neutral_mixed-PAS-21-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 2, 1, 'substitute', 59),
    ('neutral_mixed-PAS-23-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 3, 'matched', 60),
    ('neutral_mixed-PAS-23-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 3, 'matched', 61),
    ('neutral_mixed-PAS-23-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 2, 3, 'matched', 62),
    ('neutral_mixed-PAS-30-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 3, 0, 'substitute', 63),
    ('neutral_mixed-PAS-30-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 3, 0, 'substitute', 64),
    ('neutral_mixed-PAS-30-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 3, 0, 'substitute', 65),
    ('neutral_mixed-PAS-31-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 3, 1, 'substitute', 66),
    ('neutral_mixed-PAS-31-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 0, 3, 3, 1, 'substitute', 67),
    ('neutral_mixed-PAS-31-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 2, 3, 3, 1, 'substitute', 68),
    ('neutral_mixed-PAS-32-0', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 3, 2, 3, 2, 'matched', 69),
    ('neutral_mixed-PAS-32-1', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 3, 2, 3, 2, 'matched', 70),
    ('neutral_mixed-PAS-32-2', 'neutral_mixed', 'PAS', 'ours', 'neutral_mixed', 1, 3, 3, 2, 'substitute', 71),
    ('untrained-OEQ-01-0', 'untrained', 'OEQ', 'ours', 'untrained', 0, 1, 0, 1, 'matched', 72),
    ('untrained-OEQ-01-1', 'untrained', 'OEQ', 'ours', 'untrained', 0, 1, 0, 1, 'matched', 73),
    ('untrained-OEQ-01-2', 'untrained', 'OEQ', 'ours', 'untrained', 0, 1, 0, 1, 'matched', 74),
    ('untrained-OEQ-02-0', 'untrained', 'OEQ', 'ours', 'untrained', 0, 2, 0, 2, 'matched', 75),
    ('untrained-OEQ-02-1', 'untrained', 'OEQ', 'ours', 'untrained', 0, 2, 0, 2, 'matched', 76),
    ('untrained-OEQ-02-2', 'untrained', 'OEQ', 'ours', 'untrained', 0, 2, 0, 2, 'matched', 77),
    ('untrained-OEQ-03-0', 'untrained', 'OEQ', 'ours', 'untrained', 0, 3, 0, 3, 'matched', 78),
    ('untrained-OEQ-03-1', 'untrained', 'OEQ', 'ours', 'untrained', 0, 3, 0, 3, 'matched', 79),
    ('untrained-OEQ-03-2', 'untrained', 'OEQ', 'ours', 'untrained', 0, 3, 0, 3, 'matched', 80),
    ('untrained-OEQ-10-0', 'untrained', 'OEQ', 'ours', 'untrained', 1, 0, 1, 0, 'matched', 81),
    ('untrained-OEQ-10-1', 'untrained', 'OEQ', 'ours', 'untrained', 0, 3, 1, 0, 'substitute', 82),
    ('untrained-OEQ-10-2', 'untrained', 'OEQ', 'ours', 'untrained', 1, 2, 1, 0, 'substitute', 83),
    ('untrained-OEQ-12-0', 'untrained', 'OEQ', 'ours', 'untrained', 1, 2, 1, 2, 'matched', 84),
    ('untrained-OEQ-12-1', 'untrained', 'OEQ', 'ours', 'untrained', 1, 2, 1, 2, 'matched', 85),
    ('untrained-OEQ-12-2', 'untrained', 'OEQ', 'ours', 'untrained', 1, 2, 1, 2, 'matched', 86),
    ('untrained-OEQ-13-0', 'untrained', 'OEQ', 'ours', 'untrained', 1, 3, 1, 3, 'matched', 87),
    ('untrained-OEQ-13-1', 'untrained', 'OEQ', 'ours', 'untrained', 1, 3, 1, 3, 'matched', 88),
    ('untrained-OEQ-13-2', 'untrained', 'OEQ', 'ours', 'untrained', 0, 3, 1, 3, 'substitute', 89),
    ('untrained-OEQ-20-0', 'untrained', 'OEQ', 'ours', 'untrained', 2, 0, 2, 0, 'matched', 90),
    ('untrained-OEQ-20-1', 'untrained', 'OEQ', 'ours', 'untrained', 1, 2, 2, 0, 'substitute', 91),
    ('untrained-OEQ-20-2', 'untrained', 'OEQ', 'ours', 'untrained', 2, 3, 2, 0, 'substitute', 92),
    ('untrained-OEQ-21-0', 'untrained', 'OEQ', 'ours', 'untrained', 2, 1, 2, 1, 'matched', 93),
    ('untrained-OEQ-21-1', 'untrained', 'OEQ', 'ours', 'untrained', 2, 1, 2, 1, 'matched', 94),
    ('untrained-OEQ-21-2', 'untrained', 'OEQ', 'ours', 'untrained', 2, 1, 2, 1, 'matched', 95),
    ('untrained-OEQ-23-0', 'untrained', 'OEQ', 'ours', 'untrained', 2, 3, 2, 3, 'matched', 96),
    ('untrained-OEQ-23-1', 'untrained', 'OEQ', 'ours', 'untrained', 2, 3, 2, 3, 'matched', 97),
    ('untrained-OEQ-23-2', 'untrained', 'OEQ', 'ours', 'untrained', 2, 3, 2, 3, 'matched', 98),
    ('untrained-OEQ-30-0', 'untrained', 'OEQ', 'ours', 'untrained', 3, 0, 3, 0, 'matched', 99),
    ('untrained-OEQ-30-1', 'untrained', 'OEQ', 'ours', 'untrained', 2, 1, 3, 0, 'substitute', 100),
    ('untrained-OEQ-30-2', 'untrained', 'OEQ', 'ours', 'untrained', 0, 1, 3, 0, 'substitute', 101),
    ('untrained-OEQ-31-0', 'untrained', 'OEQ', 'ours', 'untrained', 3, 2, 3, 1, 'substitute', 102),
    ('untrained-OEQ-31-1', 'untrained', 'OEQ', 'ours', 'untrained', 2, 3, 3, 1, 'substitute', 103),
    ('untrained-OEQ-31-2', 'untrained', 'OEQ', 'ours', 'untrained', 0, 2, 3, 1, 'substitute', 104),
    ('untrained-OEQ-32-0', 'untrained', 'OEQ', 'ours', 'untrained', 3, 2, 3, 2, 'matched', 105),
    ('untrained-OEQ-32-1', 'untrained', 'OEQ', 'ours', 'untrained', 3, 2, 3, 2, 'matched', 106),
    ('untrained-OEQ-32-2', 'untrained', 'OEQ', 'ours', 'untrained', 3, 2, 3, 2, 'matched', 107),
    ('untrained-PAS-01-0', 'untrained', 'PAS', 'ours', 'untrained', 0, 1, 0, 1, 'matched', 108),
    ('untrained-PAS-01-1', 'untrained', 'PAS', 'ours', 'untrained', 0, 1, 0, 1, 'matched', 109),
    ('untrained-PAS-01-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 1, 0, 1, 'matched', 110),
    ('untrained-PAS-02-0', 'untrained', 'PAS', 'ours', 'untrained', 0, 2, 0, 2, 'matched', 111),
    ('untrained-PAS-02-1', 'untrained', 'PAS', 'ours', 'untrained', 0, 2, 0, 2, 'matched', 112),
    ('untrained-PAS-02-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 2, 0, 2, 'matched', 113),
    ('untrained-PAS-03-0', 'untrained', 'PAS', 'ours', 'untrained', 0, 3, 0, 3, 'matched', 114),
    ('untrained-PAS-03-1', 'untrained', 'PAS', 'ours', 'untrained', 0, 3, 0, 3, 'matched', 115),
    ('untrained-PAS-03-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 3, 0, 3, 'matched', 116),
    ('untrained-PAS-10-0', 'untrained', 'PAS', 'ours', 'untrained', 1, 0, 1, 0, 'matched', 117),
    ('untrained-PAS-10-1', 'untrained', 'PAS', 'ours', 'untrained', 1, 0, 1, 0, 'matched', 118),
    ('untrained-PAS-10-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 3, 1, 0, 'substitute', 119),
    ('untrained-PAS-12-0', 'untrained', 'PAS', 'ours', 'untrained', 1, 2, 1, 2, 'matched', 120),
    ('untrained-PAS-12-1', 'untrained', 'PAS', 'ours', 'untrained', 1, 2, 1, 2, 'matched', 121),
    ('untrained-PAS-12-2', 'untrained', 'PAS', 'ours', 'untrained', 1, 2, 1, 2, 'matched', 122),
    ('untrained-PAS-13-0', 'untrained', 'PAS', 'ours', 'untrained', 1, 3, 1, 3, 'matched', 123),
    ('untrained-PAS-13-1', 'untrained', 'PAS', 'ours', 'untrained', 1, 3, 1, 3, 'matched', 124),
    ('untrained-PAS-13-2', 'untrained', 'PAS', 'ours', 'untrained', 1, 3, 1, 3, 'matched', 125),
    ('untrained-PAS-20-0', 'untrained', 'PAS', 'ours', 'untrained', 2, 0, 2, 0, 'matched', 126),
    ('untrained-PAS-20-1', 'untrained', 'PAS', 'ours', 'untrained', 2, 0, 2, 0, 'matched', 127),
    ('untrained-PAS-20-2', 'untrained', 'PAS', 'ours', 'untrained', 1, 2, 2, 0, 'substitute', 128),
    ('untrained-PAS-21-0', 'untrained', 'PAS', 'ours', 'untrained', 2, 1, 2, 1, 'matched', 129),
    ('untrained-PAS-21-1', 'untrained', 'PAS', 'ours', 'untrained', 2, 1, 2, 1, 'matched', 130),
    ('untrained-PAS-21-2', 'untrained', 'PAS', 'ours', 'untrained', 2, 1, 2, 1, 'matched', 131),
    ('untrained-PAS-23-0', 'untrained', 'PAS', 'ours', 'untrained', 2, 3, 2, 3, 'matched', 132),
    ('untrained-PAS-23-1', 'untrained', 'PAS', 'ours', 'untrained', 2, 3, 2, 3, 'matched', 133),
    ('untrained-PAS-23-2', 'untrained', 'PAS', 'ours', 'untrained', 2, 3, 2, 3, 'matched', 134),
    ('untrained-PAS-30-0', 'untrained', 'PAS', 'ours', 'untrained', 2, 3, 3, 0, 'substitute', 135),
    ('untrained-PAS-30-1', 'untrained', 'PAS', 'ours', 'untrained', 0, 1, 3, 0, 'substitute', 136),
    ('untrained-PAS-30-2', 'untrained', 'PAS', 'ours', 'untrained', 0, 1, 3, 0, 'substitute', 137),
    ('untrained-PAS-31-0', 'untrained', 'PAS', 'ours', 'untrained', 3, 1, 3, 1, 'matched', 138),
    ('untrained-PAS-31-1', 'untrained', 'PAS', 'ours', 'untrained', 0, 2, 3, 1, 'substitute', 139),
    ('untrained-PAS-31-2', 'untrained', 'PAS', 'ours', 'untrained', 2, 1, 3, 1, 'substitute', 140),
    ('untrained-PAS-32-0', 'untrained', 'PAS', 'ours', 'untrained', 3, 2, 3, 2, 'matched', 141),
    ('untrained-PAS-32-1', 'untrained', 'PAS', 'ours', 'untrained', 3, 2, 3, 2, 'matched', 142),
    ('untrained-PAS-32-2', 'untrained', 'PAS', 'ours', 'untrained', 3, 2, 3, 2, 'matched', 143);

-- The digest of the item file this schema was generated from. pref_start_session
-- refuses a session whose page is holding a DIFFERENT item file, because such a
-- session would record judgments against text nobody can reconstruct.
create or replace function public.pref_expected_sha256() returns text
language sql immutable as $$ select 'd0c7565b52bc5cf252e4c1ffcc614115e6793f4295a3f86f10447c98c92d80fd'::text $$;

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

-- The two-argument version, from before the Prolific ID was collected. Dropped
-- rather than left in place: `create or replace` with a new argument list makes an
-- OVERLOAD, so without this the old function would survive and a page still
-- calling it would keep opening sessions with no ID to pay.
drop function if exists public.pref_start_session(text, text);

-- Start a rater session: draw the presentation order and the per-item side flip,
-- record both, and hand back only what the page needs to render.
--
-- The draw is per (session, item), not global: two raters see different orders
-- and different sides, so a systematic left-side preference cannot masquerade as
-- a preference for one arm.
create or replace function public.pref_start_session(
    p_user_agent text, p_items_sha256 text, p_prolific_pid text)
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

    insert into public.pref_sessions (session_id, session_token, items_sha256,
                                      prolific_pid, user_agent)
    values (v_id, v_token, p_items_sha256, v_pid,
            left(coalesce(p_user_agent, ''), 500));

    insert into public.pref_assignments (session_id, item_id, position, flipped)
    select v_id, item_id, row_number() over (order by random()) - 1, random() < 0.5
      from public.pref_items;

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
create or replace function public.pref_record(
    p_token text, p_item_id text, p_choice text, p_ms_elapsed integer)
returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare
    v_session public.pref_sessions;
    v_flipped boolean;
    v_item public.pref_items;
    v_left text;
    v_right text;
    v_preferred text;
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
    v_preferred := case p_choice when 'left' then v_left
                                 when 'right' then v_right
                                 else null end;

    insert into public.pref_judgments (session_id, item_id, choice, left_model,
                                       right_model, preferred_model, ms_elapsed)
    values (v_session.session_id, p_item_id, p_choice, v_left, v_right,
            v_preferred, p_ms_elapsed)
    on conflict (session_id, item_id) do update
        set choice = excluded.choice,
            left_model = excluded.left_model,
            right_model = excluded.right_model,
            preferred_model = excluded.preferred_model,
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

grant execute on function public.pref_start_session(text, text, text) to anon, authenticated;
grant execute on function public.pref_record(text, text, text, integer) to anon, authenticated;
grant execute on function public.pref_progress(text) to anon, authenticated;
grant execute on function public.pref_complete(text) to anon, authenticated;

-- NOT granted to anon: pref_session_for_token takes a token and returns a whole
-- session row, so exposing it would let anyone enumerate sessions by guessing.
-- It is only ever called from inside the three functions above, which run as
-- their definer and so do not need the grant.
revoke execute on function public.pref_session_for_token(text) from anon, authenticated;
