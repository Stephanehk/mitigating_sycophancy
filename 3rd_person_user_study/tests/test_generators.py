"""Tests for the two generated artifacts: the Supabase schema and the client
bundle.

Nothing here executes SQL (see tests/conftest.py for why). The checks are the
ones that a syntax check would NOT catch anyway and that matter most:

  - the schema's pinned digest, seeded items and allowed choices agree with
    data/items.json and config.py, so the page, the database and the artifact
    cannot disagree about what the item set is;
  - the client bundle publishes no arm name and no judge label, which is the one
    property that keeps the rating page blind.
"""

import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

import build_items
import build_static
import config
import debug_arms
import generate_supabase_sql as gen


# --- the client bundle -----------------------------------------------------

def test_the_client_bundle_publishes_only_text(bundle):
    """The page gets the prompt and the two responses. It does not get which arm
    wrote either one, nor what the endorsement judge called them -- so the
    labels being studied cannot be read out of page source."""
    published = build_static.client_bundle()
    for item in published["items"]:
        assert set(item) == set(build_static.CLIENT_KEYS), item["item_id"]
    forbidden = {"model_a", "model_b", "cat_a", "cat_b", "comparison", "env"}
    assert not (set(build_static.CLIENT_KEYS) & forbidden)


def test_the_client_bundle_is_the_same_items_in_the_same_order(bundle):
    published = build_static.client_bundle()
    assert len(published["items"]) == len(bundle["items"])
    for shown, full in zip(published["items"], bundle["items"]):
        assert shown["item_id"] == full["item_id"]
        assert shown["prompt"] == full["prompt"]
        assert shown["response_a"] == full["response_a"]
        assert shown["response_b"] == full["response_b"]


def test_the_client_bundle_carries_the_full_item_digest(bundle):
    """The digest must be over the FULL item set, not the reduced view, because
    pref_start_session compares it against the schema's pinned value."""
    published = build_static.client_bundle()
    assert published["items_sha256"] == bundle["items_sha256"]
    assert published["items_sha256"] == build_items.items_digest(bundle["items"])


def test_write_config_refuses_placeholder_credentials(monkeypatch):
    """A config.json emitted with a blank key would give a page that loads and
    then silently records nothing."""
    monkeypatch.setattr(config, "SUPABASE_URL", "")
    with pytest.raises(AssertionError):
        build_static.write_config()


# --- the generated schema --------------------------------------------------

def test_schema_pins_the_committed_digest(schema_sql, bundle):
    assert f"select '{bundle['items_sha256']}'::text" in schema_sql


def test_schema_allows_exactly_the_configured_choices(schema_sql):
    match = re.search(r"check \(choice in \(([^)]*)\)\)", schema_sql)
    assert match, "no choice constraint in the schema"
    allowed = [value.strip().strip("'") for value in match.group(1).split(",")]
    assert allowed == config.CHOICES


def test_schema_seeds_every_item_and_only_those(schema_sql, bundle):
    seeded = re.findall(
        r"^    \('([^']+)', '([^']+)', '([^']+)', '([^']+)', '([^']+)',"
        r" (\d), (\d), (\d), (\d), '([^']+)', (\d+), (\d+)\)", schema_sql, re.M)
    assert len(seeded) == len(bundle["items"])
    by_id = {item["item_id"]: item for item in bundle["items"]}
    for position, row in enumerate(seeded):
        (item_id, comparison, env, model_a, model_b, cat_a, cat_b,
         target_a, target_b, origin, pos, set_index) = row
        item = by_id[item_id]
        assert (comparison, env, model_a, model_b) == (
            item["comparison"], item["env"], item["model_a"], item["model_b"])
        assert (int(cat_a), int(cat_b)) == (item["cat_a"], item["cat_b"])
        assert (int(target_a), int(target_b)) == (item["target_cat_a"],
                                                 item["target_cat_b"])
        assert origin == item["origin"]
        assert int(pos) == position
        assert int(set_index) == item["set_index"]


def test_schema_can_separate_substitutes_from_matched_items(schema_sql, bundle):
    """The whole reason target_cat_* and origin are seeded: an analysis that runs
    in SQL must be able to exclude substitutes without going back to
    data/items.json."""
    assert "origin       text not null check (origin in ('matched', 'substitute'))" \
        in schema_sql
    assert "target_cat_a integer not null" in schema_sql
    # And the check constraint refuses a row whose origin contradicts its labels.
    assert "check ((origin = 'matched')" in schema_sql
    seeded_substitutes = schema_sql.count(", 'substitute', ")
    assert seeded_substitutes == sum(1 for item in bundle["items"]
                                     if item["origin"] == "substitute")


def test_schema_does_not_leak_response_text(schema_sql, bundle):
    """Item text is deliberately NOT in the database; data/items.json is the
    record and the digest is the join key."""
    for item in bundle["items"][:10]:
        assert item["response_a"][:60] not in schema_sql
        assert item["prompt"][:60] not in schema_sql


def test_the_schema_stores_a_prolific_id_and_refuses_a_blank_one(schema_sql):
    """A session with no Prolific ID cannot be matched to a participant, so it is
    neither payable nor usable. The column is nullable on purpose -- this file is
    applied by hand to a project that may already hold sessions from before the
    column existed, and a `not null` alter would fail on them -- so the RPC is
    where the requirement has to be enforced."""
    assert "prolific_pid  text" in schema_sql
    assert "add column if not exists prolific_pid text" in schema_sql, \
        "a project whose pref_sessions predates the column would not gain it"
    assert "p_prolific_pid text" in schema_sql
    assert "raise exception 'prolific id is required'" in schema_sql
    assert "prolific_pid, set_index, user_agent" in schema_sql, "the id is not inserted"


def test_superseded_start_session_signatures_are_dropped(schema_sql):
    """`create or replace` with a new argument list makes an OVERLOAD, not a
    replacement: without the drop the old function survives and a page still
    calling it keeps opening sessions with no ID to pay."""
    assert "drop function if exists public.pref_start_session(text, text);" in schema_sql
    assert "grant execute on function public.pref_start_session(text, text)" \
        not in schema_sql
    assert "drop function if exists public.pref_start_session(text, text, text);" \
        in schema_sql, "the 3-arg form would survive as an overload"
    assert "grant execute on function public.pref_start_session(text, text, text, integer)" \
        in schema_sql

    # The drop has to come BEFORE the create, or it removes what was just made.
    assert (schema_sql.index("drop function if exists public.pref_start_session")
            < schema_sql.index("create or replace function public.pref_start_session"))


def test_every_granted_function_is_defined(schema_sql):
    defined = set(re.findall(r"create or replace function public\.(\w+)\(", schema_sql))
    granted = set(re.findall(r"grant execute on function public\.(\w+)\(", schema_sql))
    assert granted <= defined, f"granted but undefined: {sorted(granted - defined)}"
    assert {"pref_start_session", "pref_record", "pref_progress",
            "pref_complete"} <= granted


def test_the_token_lookup_is_not_reachable_from_the_browser(schema_sql):
    """pref_session_for_token takes a token and hands back a whole session row.
    Granting it to anon would let anyone probe for valid tokens."""
    assert "revoke execute on function public.pref_session_for_token(text)" in schema_sql
    assert "grant execute on function public.pref_session_for_token" not in schema_sql


def test_every_referenced_table_is_created(schema_sql):
    created = set(re.findall(r"create table if not exists public\.(\w+)", schema_sql))
    referenced = set(re.findall(r"references public\.(\w+)", schema_sql))
    assert referenced <= created, sorted(referenced - created)
    assert created == {"pref_sessions", "pref_items", "pref_assignments",
                       "pref_judgments"}


def test_row_level_security_is_on_for_every_table(schema_sql):
    created = set(re.findall(r"create table if not exists public\.(\w+)", schema_sql))
    for table in created:
        assert f"alter table public.{table}" in schema_sql
        assert re.search(rf"alter table public\.{table}\s+enable row level security",
                         schema_sql), table
    assert "create policy" not in schema_sql


def test_dollar_quoted_bodies_are_balanced(schema_sql):
    """An unbalanced $$ turns the rest of the file into one function body, which
    Postgres reports as a confusing error far from the real mistake."""
    assert schema_sql.count("$$") % 2 == 0, "odd number of $$ delimiters"
    for name, body in re.findall(
            r"create or replace function public\.(\w+)\([^)]*\)(.*?)\$\$;",
            schema_sql, re.S):
        assert body.count("$$") == 1, f"{name}: nested or missing $$ opener"


def test_every_function_pins_its_search_path(schema_sql):
    """A SECURITY DEFINER function without `set search_path` can be hijacked by a
    caller-controlled schema. The one exception is the immutable sql literal."""
    for name, header in re.findall(
            r"create or replace function public\.(\w+)\([^)]*\)\s*\nreturns [^\n]*\n(language[^\n]*(?:\n[^\n]*)?)",
            schema_sql):
        if "security definer" not in header:
            continue
        assert "set search_path = public, pg_temp" in header, name


def test_sql_string_literals_escape_quotes():
    assert gen.sql_str("it's") == "'it''s'"
    assert gen.sql_str("plain") == "'plain'"


def test_committed_schema_is_current(schema_sql):
    """The committed file is what gets pasted into the SQL editor, so it drifting
    from the generator is the failure that actually reaches production."""
    path = os.path.join(config.HERE, "supabase_schema.sql")
    if not os.path.exists(path):
        pytest.skip("supabase_schema.sql not generated yet")
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == schema_sql, (
            "regenerate: python 3rd_person_user_study/tools/"
            "generate_supabase_sql.py > 3rd_person_user_study/supabase_schema.sql")


# --- the page --------------------------------------------------------------

def test_the_page_never_names_an_arm_or_a_category():
    """The blindness property has to hold in app.js too: it renders `a`/`b` and
    sends a side. If it ever learned the arm, the published bundle would have to
    carry it."""
    path = os.path.join(config.STATIC_DIR, "app.js")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    for arm in config.ARMS:
        assert arm not in source, f"app.js mentions the arm {arm!r}"
    for token in ["cat_a", "cat_b", "model_a", "model_b", "comparison"]:
        assert token not in source, f"app.js mentions {token!r}"


def test_the_page_sends_only_configured_choices():
    path = os.path.join(config.STATIC_DIR, "app.js")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    for choice in config.CHOICES:
        assert f"'{choice}'" in source, f"app.js has no button for {choice!r}"


# --- the ?labels=1 arm overlay ---------------------------------------------

def test_arms_map_names_both_arms_of_every_item(bundle):
    mapping = debug_arms.arms_map(bundle["items"])
    assert len(mapping) == len(bundle["items"])
    for item in bundle["items"]:
        assert mapping[item["item_id"]] == {"a": item["model_a"],
                                            "b": item["model_b"]}
        assert set(mapping[item["item_id"]].values()) <= set(config.ARMS)


def test_arms_map_refuses_the_published_bundle():
    """The published items have no arm names, so mapping them would silently
    produce nothing to label. It fails instead."""
    published = build_static.client_bundle()
    with pytest.raises(AssertionError, match="no arm names"):
        debug_arms.arms_map(published["items"])


def test_the_published_item_set_still_carries_no_arm_names(bundle):
    """The mapping is published in its OWN file. That must not become an excuse
    for items.json to start carrying arm names: the plain rating page fetches
    items.json and nothing else, so this is what keeps it blind."""
    published = build_static.client_bundle()
    blob = json.dumps(published)
    for arm in config.ARMS:
        assert f'"{arm}"' not in blob, f"items.json names the arm {arm!r}"


def test_the_build_does_not_publish_the_arm_mapping():
    """The mapping is the answer key for a live study, so the deployed bundle
    must not contain it and build_static must not write it. `?labels=1` still
    works locally -- tools/local_server.py injects the overlay itself."""
    assert not hasattr(build_static, "write_debug_labels"), \
        "build_static still emits the overlay"
    for name in ("__debug_arms.json", "debug_labels.js"):
        assert not os.path.exists(os.path.join(config.STATIC_DIR, name)), \
            f"static/{name} would be deployed"


def test_the_generated_files_are_not_committed():
    """They are build output. Committing them would let the mapping fall behind
    data/items.json and label the wrong arms."""
    path = os.path.join(config.HERE, ".gitignore")
    with open(path, encoding="utf-8") as handle:
        ignored = handle.read()
    for name in ("static/__debug_arms.json", "static/debug_labels.js"):
        assert name in ignored, f"{name} is not gitignored"


def test_the_build_path_imports_without_pandas():
    """The deploy workflow installs nothing and only serializes an already-built
    data/items.json, so nothing it imports may need the sampling pipeline.

    This is not hypothetical: config.py imported run_quality_judges_pas at module
    scope, and every Pages run died on `No module named 'pandas'` before reaching
    the build. Run in a subprocess with the import blocked, because pandas IS
    importable in the dev env -- checking sys.modules here would prove nothing.
    """
    guard = textwrap.dedent("""
        import builtins, sys
        real_import = builtins.__import__
        def guard(name, *a, **k):
            if name.split(".")[0] == "pandas":
                raise ModuleNotFoundError("No module named 'pandas'")
            return real_import(name, *a, **k)
        builtins.__import__ = guard
        sys.path.insert(0, %r)
        sys.path.insert(0, %r)
        import config, debug_arms, build_static
        build_static.client_bundle()
    """) % (os.path.join(config.HERE, "tools"), config.HERE)
    done = subprocess.run([sys.executable, "-c", guard], capture_output=True,
                          text=True)
    assert done.returncode == 0, (
        "the static build needs pandas, so the Pages deploy will fail:\n"
        + done.stderr)
