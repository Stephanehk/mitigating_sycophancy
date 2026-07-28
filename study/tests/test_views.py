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


def test_the_bundle_carries_all_three_terminal_destinations():
    terminal = views.content_bundle()["terminal"]
    assert set(terminal) == {"complete", "screened_out", "declined"}
    assert terminal["complete"]["redirect_url"] == config.PROLIFIC_COMPLETION_URL
    assert terminal["screened_out"]["redirect_url"] == config.PROLIFIC_SCREENOUT_URL
    assert terminal["declined"]["redirect_url"] == config.PROLIFIC_SCREENOUT_URL


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
