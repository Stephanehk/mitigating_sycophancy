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


def test_the_bundle_agrees_with_scenarios_about_which_answer_selects():
    bundle = views.content_bundle()
    expected = [c["value"] for c in scenarios.SIMILARITY_CHOICES if c["selects"]]
    assert bundle["screening"]["selects_scenario"] == expected
    assert bundle["screening"]["describe_required_for"] == list(scenarios.DESCRIBE_REQUIRED_FOR)


def test_the_bundle_is_json_serializable():
    """It is sent verbatim over the wire, so a stray tuple would 500 at runtime."""
    json.dumps(views.content_bundle())
