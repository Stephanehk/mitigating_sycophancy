"""Tests for views.py: the content bundle.

Build-time only: tools/build_static.py turns this into static/content.json. It
is the client's entire source of truth for what the study says, so the thing
worth testing is that it is COMPLETE — every scenario, every questionnaire item,
every terminal message — and that it does not carry the one fact the client must
never learn.

The response-state tests that used to live here are gone with the API server;
that behaviour is now job_state() in plpgsql and is covered by test_rpcs.py.
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import content
import questionnaires
import scenarios
import views
from prompts import ARMS


# --- Content bundle --------------------------------------------------------

def test_the_bundle_carries_every_scenario_with_all_three_rendered_fields():
    bundle = views.content_bundle()
    assert set(bundle["screening"]["scenarios"]) == set(scenarios.scenario_keys())
    for key, scenario in bundle["screening"]["scenarios"].items():
        source = scenarios.get_scenario(key)
        assert scenario["title"] == source["title"]
        assert scenario["vignette"] == source["vignette"]
        assert scenario["question"] == source["question"]


def test_the_bundle_carries_every_questionnaire_page_and_item():
    bundle = views.content_bundle()
    pages = bundle["questionnaire_pages"]
    assert [p["page"] for p in pages] == questionnaires.QUESTIONNAIRE_PAGE_IDS

    bundled_keys = []
    for page in pages:
        for question in page["questions"]:
            bundled_keys.extend(questionnaires.response_keys(question))
    assert bundled_keys == questionnaires.all_response_keys(), \
        "the client cannot render an item the bundle does not carry"


def test_the_bundle_never_carries_the_arm_or_any_prompt():
    """The client must not be able to read, infer or claim its condition."""
    import prompts
    blob = json.dumps(views.content_bundle())
    for arm in ARMS:
        assert '"%s"' % arm not in blob
    assert prompts.BASE_SYSTEM_PROMPT not in blob
    assert prompts.SYCO_SYSTEM_PROMPT not in blob


def test_the_bundle_carries_a_completion_code_where_one_is_owed():
    """Nothing redirects any more; each terminal page shows a code instead.

    Screen-outs and decliners get the SCREEN-OUT code — that is the one Prolific
    expects for a returned slot, and paying them the completion code would be
    wrong.
    """
    terminal = views.content_bundle()["terminal"]
    assert set(terminal) == {"complete", "screened_out", "declined"}
    assert terminal["complete"]["code"] == config.PROLIFIC_COMPLETION_CODE
    assert terminal["screened_out"]["code"] == config.PROLIFIC_SCREENOUT_CODE
    for name, entry in terminal.items():
        assert "redirect_url" not in entry, "%s still carries a redirect URL" % name


def test_a_decliner_is_given_no_completion_code():
    """Someone who did not consent has not taken part, so there is nothing to
    pay for — they return the submission on Prolific instead.

    Screen-outs are different and DO get a code: they answered the screening,
    which is real data and a reportable result.
    """
    terminal = views.content_bundle()["terminal"]
    assert "code" not in terminal["declined"], (
        "a decliner is being shown a completion code")
    assert terminal["screened_out"].get("code"), (
        "a screen-out still needs a code — they completed the screening")
    assert views.content_bundle()["no_code_instruction"], (
        "with no code, a decliner must be told what to do instead")


def test_the_bundle_carries_the_validation_thresholds_the_client_enforces():
    bundle = views.content_bundle()
    assert bundle["screening"]["min_describe_chars"] == config.MIN_DESCRIBE_CHARS
    assert bundle["perspective"]["min_chars"] == config.MIN_PERSPECTIVE_CHARS
    assert bundle["max_text_chars"] == config.MAX_TEXT_CHARS
    # The review box holds the COMPOSED text, so its ceiling is the composed
    # one. Giving it the per-field cap would put a maxlength on the box that
    # the text already inside it can exceed.
    assert bundle["review"]["max_chars"] == config.MAX_MODEL_INPUT_CHARS
    assert bundle["review"]["min_chars"] == (config.MIN_DESCRIBE_CHARS
                                             + config.MIN_PERSPECTIVE_CHARS)


def test_the_composed_ceiling_clears_the_worst_case_the_client_can_build():
    """Two maxed-out fields joined by a blank line.

    On a no-review arm the participant never sees a page holding the composed
    text, so a ceiling below this is a dead end rather than something they can
    shorten.
    """
    assert config.MAX_MODEL_INPUT_CHARS >= 2 * config.MAX_TEXT_CHARS + len("\n\n")


def test_the_bundle_agrees_with_scenarios_about_which_answer_selects():
    bundle = views.content_bundle()
    expected = [c["value"] for c in scenarios.SIMILARITY_CHOICES if c["selects"]]
    assert bundle["screening"]["selects_scenario"] == expected
    assert bundle["screening"]["describe_required_for"] == list(scenarios.DESCRIBE_REQUIRED_FOR)


# --- The arm domain in SQL -------------------------------------------------

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sql(name):
    with open(os.path.join(_STUDY, name), encoding="utf-8") as handle:
        return handle.read()


def test_the_generated_schema_constrains_the_arm_column_to_exactly_the_arms():
    """The constraint must be REGENERATED from prompts.ARMS, not frozen.

    This is the bug of 2026-07-30, in test form: both hand-written schema files
    carried `arm in ('ours','untrained','syco','neutral')` inline, and both use
    `create table if not exists`, so re-running everything after two arms were
    added left the old four-name check in place — and the live database rejected
    every session drawn into a new arm.
    """
    sql = _sql("supabase_app_schema.sql")
    expected = "check (arm in (%s))" % ", ".join("'%s'" % arm for arm in ARMS)
    for table in ("study_sessions", "study_submissions"):
        assert "%s_arm_check" % table in sql, (
            "%s's arm constraint is not emitted, so it can never be updated"
            % table)
        assert "drop constraint if exists %s_arm_check" % table in sql, (
            "%s's arm constraint is added but never dropped, so re-running "
            "cannot widen it" % table)
    assert sql.count(expected) == 2, (
        "the generated arm domain does not match prompts.ARMS; expected %r"
        % (expected,))


def test_no_hand_written_schema_file_hardcodes_the_arm_list():
    """A second copy of the list is a second thing to forget."""
    for name in ("supabase_schema.sql", "supabase_queue_schema.sql"):
        sql = _sql(name)
        assert "arm in (" not in sql, (
            "%s hardcodes the arm domain again; it belongs in prompts.py, "
            "emitted by tools/generate_supabase_app_sql.py" % name)


# --- The manual-entry conditions in SQL ------------------------------------

def test_the_generated_schema_maps_every_manual_entry_condition_to_its_arm():
    """The key → arm mapping must be GENERATED, not written twice.

    The browser sends only the key from its URL, so this CASE is the whole
    definition of what /debug_c1 means. A hand-maintained copy would be the arm
    constraint's bug all over again: prompts.py changes, the database keeps
    sending people to the old condition, and nothing fails.
    """
    import re
    from prompts import DEBUG_CONDITIONS
    sql = _sql("supabase_app_schema.sql")
    # Matches only the CASE branches: the study's other `case` (job_state) is
    # the `case when <condition>` form, with no quoted literal after `when`.
    branches = re.findall(r"when '(\w+)' then '(\w+)'", sql)
    assert dict(branches) == DEBUG_CONDITIONS, (
        "the generated key -> arm mapping is not prompts.DEBUG_CONDITIONS: %r"
        % (branches,))


def test_start_session_drops_its_old_single_argument_signature():
    """Otherwise PostgREST sees an overload and a one-argument call — which is
    what every real participant makes — becomes ambiguous."""
    sql = _sql("supabase_app_schema.sql")
    assert "drop function if exists public.start_session(text);" in sql
    assert "grant execute on function public.start_session(text, text)" in sql, (
        "the two-argument start_session is not granted to anon, so the browser "
        "could not start a session at all")


def test_the_debug_flag_column_is_emitted_for_both_tables():
    """`create table if not exists` never revisits an existing table, so a
    column declared only in the hand-written schema would not exist in the live
    database — and submit_study inserts into it."""
    sql = _sql("supabase_app_schema.sql")
    for table in ("study_sessions", "study_submissions"):
        assert ("alter table public.%s add column if not exists is_debug" % table) in sql


def test_a_forced_arm_is_recorded_as_such():
    """is_debug must be written from the SESSION at submit time. If the flag were
    lost there, a hand-walked condition would be indistinguishable from a
    participant in the exported data."""
    sql = _sql("supabase_app_schema.sql")
    assert "v_session.is_debug" in sql, (
        "submit_study does not carry is_debug from the session")
    assert "'is_debug', v_arm is not null" in sql, (
        "start_session does not tell the client whether the link was honoured")


def test_the_generated_sql_is_up_to_date_with_its_generator():
    """The committed file is what gets pasted into the SQL editor, so a stale
    one is a deploy that silently does the wrong thing."""
    import subprocess
    generated = subprocess.run(
        [sys.executable, os.path.join(_STUDY, "tools", "generate_supabase_app_sql.py")],
        cwd=_STUDY, capture_output=True, text=True, check=True).stdout
    assert generated == _sql("supabase_app_schema.sql"), (
        "supabase_app_schema.sql is out of date — regenerate it with\n"
        "  python tools/generate_supabase_app_sql.py > supabase_app_schema.sql")


def test_the_bundle_is_json_serializable():
    """It is sent verbatim over the wire, so a stray tuple would 500 at runtime."""
    json.dumps(views.content_bundle())
