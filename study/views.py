"""Server-to-client payloads for the sycophancy user study.

Two payloads, and that is the whole module:

  content_bundle()  — every piece of study copy, every scenario, every
                      questionnaire page, sent ONCE at session creation. The
                      client then renders the entire flow without talking to the
                      server again until it needs a generation.

  response_state()  — the state of a session's generation job, polled while the
                      loading screen is up.

WHAT THIS GAVE UP. The old stage-scoped payload only ever sent the copy for the
stage a participant had reached, so the debrief could not be read early. Sending
the bundle up front loses that: the whole instrument is visible in devtools from
the first page. That was judged an acceptable trade, because the guarantees that
actually protect the DVs — the arm draw and the one-job-per-session rule — are
enforced server-side and are NOT in this bundle. The arm is never sent to the
client at all.
"""

import config
import content
import questionnaires
import scenarios
import store

#: How many times a participant may press "Try again" before the failure
#: becomes permanent. Each retry already gets a full JOB_MAX_ATTEMPTS budget
#: from the worker, so one manual retry is a second full round of attempts;
#: past that the GPU is genuinely unavailable and looping wastes their time.
MAX_MANUAL_RETRIES = 1


def retry_exhausted(job):
    """Whether this job has used up its manual "Try again" presses.

    Reads `manual_retries`, which store.requeue_job increments and never resets,
    because the row is reused and its status therefore cannot distinguish one
    manual retry from two.
    """
    return job["manual_retries"] >= MAX_MANUAL_RETRIES


def response_state(session_id):
    """Client-facing view of a session's model response and its job.

    Reaps expired jobs first, so a participant polling the loading screen is the
    thing that drives lease recovery — no separate cron is needed for the
    single-worker deployment this study uses.

    Asserts a job exists: the client only polls after /api/response/start
    returned, and that endpoint creates one.
    """
    store.reap_expired_jobs()
    job = store.latest_job_for_session(session_id)
    assert job is not None, "session %r polled for a response with no job" % (session_id,)

    view = {
        "job_id": job["job_id"],
        "status": job["status"],
        "poll_interval_ms": config.CLIENT_POLL_INTERVAL_MS,
        "slow_notice_ms": config.CLIENT_SLOW_NOTICE_MS,
    }
    if job["status"] == "done":
        view["text"] = job["response_text"]
    if job["status"] == "failed":
        view["can_retry"] = not retry_exhausted(job)
    return view


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
            "banner": content.AI_DISCLAIMER_BANNER,
        },
        "response": {
            "banner": content.AI_DISCLAIMER_BANNER,
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
        "dv_banner": content.AI_DISCLAIMER_BANNER,
        "debrief": content.DEBRIEF,
        "additional_thoughts": {
            "headline": content.ADDITIONAL_THOUGHTS_HEADLINE,
            "question": content.ADDITIONAL_THOUGHTS_QUESTION,
        },
        "terminal": {
            "complete": {"message": content.COMPLETE_MESSAGE,
                         "redirect_url": config.PROLIFIC_COMPLETION_URL},
            "screened_out": {"message": content.SCREENOUT_MESSAGE,
                             "redirect_url": config.PROLIFIC_SCREENOUT_URL},
            "declined": {"message": content.DECLINE_MESSAGE,
                         "redirect_url": config.PROLIFIC_SCREENOUT_URL},
        },
        "max_text_chars": config.MAX_TEXT_CHARS,
        "redirect_delay_seconds": config.REDIRECT_DELAY_SECONDS,
    }
