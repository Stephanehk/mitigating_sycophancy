"""This study shares ONE Supabase project with study/, so its objects must not
collide with that study's.

`static/config.json` and `study/static/config.json` point at the same project ref.
Nothing enforces separation at the database level -- both schemas are applied by
hand into `public` -- so the only thing keeping the two studies' results apart is
that they name different tables. That is a naming convention, and a naming
convention nobody checks is one that holds until someone adds a table called
`sessions`.

What would go wrong is not a clash error. Both schemas use `create table if not
exists` and `create or replace function`, so a shared name would be silently
adopted by whichever study ran its DDL last -- and `3rd_person_user_study`'s
schema TRUNCATES the tables it seeds. A collision would delete live study/ data
without erroring.

So: every object this study creates is `pref_`-prefixed, study/'s are not, and
no destructive statement in either file names anything the other owns.
"""

import os
import re

import pytest

import config

STUDY_DIR = os.path.join(config.REPO, "study")
STUDY_SCHEMAS = ["supabase_schema.sql", "supabase_app_schema.sql",
                 "supabase_queue_schema.sql"]
OUR_SCHEMA = os.path.join(config.HERE, "supabase_schema.sql")

#: Every table, view and function this study is allowed to own.
OUR_PREFIX = "pref_"

_TABLE = re.compile(r"create table (?:if not exists )?public\.(\w+)", re.I)
_VIEW = re.compile(r"create (?:or replace )?view public\.(\w+)", re.I)
_FUNCTION = re.compile(r"create (?:or replace )?function public\.(\w+)", re.I)
#: Statements that destroy data or definitions, and the object they name.
_DESTRUCTIVE = re.compile(
    r"^\s*(?:truncate|delete\s+from|drop\s+(?:table|view|function)"
    r"(?:\s+if\s+exists)?)\s+(?:public\.)?(\w+)", re.I | re.M)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _objects(sql):
    """The tables, views and functions a schema file creates."""
    return (set(_TABLE.findall(sql)) | set(_VIEW.findall(sql)),
            set(_FUNCTION.findall(sql)))


@pytest.fixture(scope="module")
def study_sql():
    """study/'s three schema files concatenated. Skips if that study is absent,
    so this suite still runs in a checkout that only has this one."""
    paths = [os.path.join(STUDY_DIR, name) for name in STUDY_SCHEMAS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        pytest.skip(f"study/ schemas not present: {missing}")
    return "\n".join(_read(p) for p in paths)


@pytest.fixture(scope="module")
def our_sql():
    if not os.path.exists(OUR_SCHEMA):
        pytest.skip(f"{OUR_SCHEMA} not generated")
    return _read(OUR_SCHEMA)


def test_the_two_studies_share_no_table_or_function(our_sql, study_sql):
    ours_tables, ours_functions = _objects(our_sql)
    theirs_tables, theirs_functions = _objects(study_sql)
    assert ours_tables and theirs_tables, "parsed no tables -- the regex is wrong"

    assert not (ours_tables & theirs_tables), (
        "these tables are declared by BOTH studies, so one study's results "
        f"would land in the other's rows: {sorted(ours_tables & theirs_tables)}")
    assert not (ours_functions & theirs_functions), (
        "these functions are declared by BOTH studies, so applying one schema "
        "would redefine the other study's RPC: "
        f"{sorted(ours_functions & theirs_functions)}")


def test_every_object_this_study_creates_is_prefixed(our_sql):
    """The prefix is what makes the disjointness robust to study/ growing a new
    table, rather than true only of the two files as they stand today."""
    tables, functions = _objects(our_sql)
    for name in sorted(tables | functions):
        assert name.startswith(OUR_PREFIX), (
            f"{name} is not {OUR_PREFIX}-prefixed, so it is not reserved to "
            "this study in the shared project")


def test_no_destructive_statement_reaches_the_other_study(our_sql, study_sql):
    """The one that would lose data silently. Our schema truncates what it
    seeds; that truncate must never name a study/ table."""
    theirs_tables, theirs_functions = _objects(study_sql)
    for target in _DESTRUCTIVE.findall(our_sql):
        assert target not in theirs_tables | theirs_functions, (
            f"this schema destroys {target}, which belongs to study/")

    ours_tables, ours_functions = _objects(our_sql)
    for target in _DESTRUCTIVE.findall(study_sql):
        assert target not in ours_tables | ours_functions, (
            f"study/'s schema destroys {target}, which belongs to this study")


def test_the_truncate_cascade_cannot_reach_the_other_study(our_sql, study_sql):
    """`truncate pref_items cascade` follows foreign keys INTO pref_items. If a
    study/ table ever referenced one of ours, cascade would empty it too."""
    ours_tables, _ = _objects(our_sql)
    for referenced in re.findall(r"references\s+public\.(\w+)", study_sql, re.I):
        assert referenced not in ours_tables, (
            f"a study/ table has a foreign key to {referenced}; a truncate "
            "cascade here would empty it")
