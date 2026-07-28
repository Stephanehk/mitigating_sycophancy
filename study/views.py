"""The content bundle: every string and question the SPA renders.

BUILD-TIME ONLY. Nothing imports this at run time — there is no application
server. tools/build_static.py calls content_bundle() to emit
static/content.json, which Amplify serves as a CDN-cached asset, and
tools/generate_supabase_app_sql.py reads MAX_MANUAL_RETRIES.

It stays a Python module rather than a hand-maintained JSON file so content.py,
questionnaires.py and scenarios.py remain the single source of truth for the
instrument.

WHAT THIS GAVE UP. The old server sent only the copy for the stage a
participant had reached, so the debrief could not be read early. A static bundle
loses that: the whole instrument is visible in devtools from the first page.
Judged acceptable, because the guarantees that actually protect the DVs — the
arm draw and one-job-per-session — are enforced inside Postgres and are not in
this bundle. The arm is never sent to the client at all.
"""

import config
import content
import questionnaires
import scenarios

#: How many times a participant may press "Try again" before the failure
#: becomes permanent. Each retry already gets a full JOB_MAX_ATTEMPTS budget
#: from the worker, so one manual retry is a second full round of attempts;
#: past that the GPU is genuinely unavailable and looping wastes their time.
#: Read by the SQL generator into study_config, where retry_response enforces it.
MAX_MANUAL_RETRIES = 1


def content_bundle():
    """Every string and question the client needs to render the whole study.

    Sent once, at session creation. Keeping it derived from content.py /
    scenarios.py / questionnaires.py rather than duplicated into JavaScript is
    the point: those modules stay the single source of truth for the instrument,
    and the client is a renderer.

    Deliberately ABSENT: the arm, and anything that would reveal it.
    """
    return {
        "consent": {
            "title": content.CONSENT_TITLE,
            "body": content.CONSENT_BODY,
            "choices": content.CONSENT_CHOICES,
        },
        "prolific": {
            "question": content.PROLIFIC_ID_QUESTION,
            "help": content.PROLIFIC_ID_HELP,
            "required_error": content.PROLIFIC_ID_REQUIRED_ERROR,
        },
        "prelude": content.PRELUDE,
        "screening": {
            "scenarios": {s["key"]: {
                "key": s["key"],
                "title": s["title"],
                "vignette": s["vignette"],
                "question": s["question"],
            } for s in scenarios.SCENARIOS},
            "choices": scenarios.SIMILARITY_CHOICES,
            "describe_question": content.DESCRIBE_SIMILAR_QUESTION,
            "describe_required_for": list(scenarios.DESCRIBE_REQUIRED_FOR),
            "selects_scenario": [c["value"] for c in scenarios.SIMILARITY_CHOICES
                                 if c["selects"]],
            "min_describe_chars": config.MIN_DESCRIBE_CHARS,
        },
        "perspective": {
            "intro": content.PERSPECTIVE_INTRO,
            "question": content.PERSPECTIVE_QUESTION,
            "reminder": content.ANONYMITY_REMINDER,
            "min_chars": config.MIN_PERSPECTIVE_CHARS,
        },
        "response_intro": {
            "text": content.RESPONSE_INTRO,
        },
        "review": {
            "headline": content.REVIEW_HEADLINE,
            "help": content.REVIEW_HELP,
            "confirm": content.REVIEW_CONFIRM,
            # The floor start_response enforces on the edited text: the same
            # total the two separate pages already required, so editing cannot
            # be used to get under the length requirement.
            "min_chars": config.MIN_DESCRIBE_CHARS + config.MIN_PERSPECTIVE_CHARS,
        },
        "response": {
            "reminder": content.RESPONSE_PAGE_REMINDER,
            "loading_headline": content.LOADING_HEADLINE,
            "loading_subtext": content.LOADING_SUBTEXT,
            "loading_slow": content.LOADING_SLOW_NOTICE,
            "failure_headline": content.FAILURE_HEADLINE,
            "failure_subtext": content.FAILURE_SUBTEXT,
            "failure_retry": content.FAILURE_RETRY_LABEL,
            "failure_exhausted": content.FAILURE_EXHAUSTED,
        },
        "questionnaire_pages": questionnaires.QUESTIONNAIRE_PAGES,
        "debrief": content.DEBRIEF,
        "additional_thoughts": {
            "headline": content.ADDITIONAL_THOUGHTS_HEADLINE,
            "question": content.ADDITIONAL_THOUGHTS_QUESTION,
        },
        # Each terminal page shows a completion CODE, except one. Nothing
        # redirects and nothing links out: an auto-redirect could strand a
        # participant who had finished, with no way to prove it.
        #
        # NO CODE FOR A DECLINER. Someone who did not consent has not taken
        # part, so there is nothing to pay for — they return the submission on
        # Prolific instead. Screen-outs DO get one: they answered the screening,
        # which is real data and a reportable result, and the screen-out code is
        # what Prolific expects for that slot.
        "terminal": {
            "complete": {"message": content.COMPLETE_MESSAGE,
                         "code": config.PROLIFIC_COMPLETION_CODE},
            "screened_out": {"message": content.SCREENOUT_MESSAGE,
                             "code": config.PROLIFIC_SCREENOUT_CODE},
            "declined": {"message": content.DECLINE_MESSAGE},
        },
        "code_label": content.CODE_LABEL,
        "code_instruction": content.CODE_INSTRUCTION,
        "no_code_instruction": content.DECLINE_INSTRUCTION,
        "max_text_chars": config.MAX_TEXT_CHARS,
    }
