"""Randomization and submission validation for the sycophancy user study.

Pure logic: no I/O, no HTTP, no database.

THE STAGE MACHINE THAT USED TO LIVE HERE IS GONE. Page-to-page navigation moved
into the browser (static/app.js), because every Continue press used to be a
server round trip plus two synchronous writes, and on this cluster's NFS mount
that cost ~213ms typical and stalled for 10-14 seconds under worker-poll
contention. The client now walks the flow locally and posts one payload at the
end.

What stayed on the server, and why:

  * `new_screening_order` / `assign_arm` — randomization. If the client drew its
    own arm, a participant could re-roll their condition from devtools, which is
    the one thing that would actually corrupt the DVs. server.py draws the arm
    once, at job creation, and never redraws it.
  * `clean_text` — the two free-text answers that become the model's input are
    the only participant strings the server accepts mid-study, so they are the
    only ones it can still validate at the point of entry.
  * `validate_submission` — a whole-payload check run once at /api/submit,
    replacing the old per-page `validate_page_submission`.

The stage sequence the CLIENT walks (static/app.js owns this now):

  consent -> declined                (terminal, redirect to Prolific)
  prolific_id
  prelude
  screening                          (4 scenarios, per-session random order,
                                      stops at the first "Very similar")
    -> screened_out                  (terminal, no scenario matched)
  perspective                        ("your side of the story")
  response_intro
  response                           (the ONE server round trip: loading -> reply)
  dv_felt_understood_trust ... big_five
  debrief
  additional_thoughts
  complete                           (the ONE submit, then redirect)
"""

import random

import config
import questionnaires
import scenarios
from prompts import ARMS


class ValidationError(Exception):
    """A participant submission that the study logic rejects.

    Raised for recoverable, participant-visible problems (text too short) and
    for structurally impossible input (an out-of-range answer code, which means
    a tampered or broken client). server.py turns this into a 400.
    """


# --- Randomization ---------------------------------------------------------

def new_screening_order(rng=None):
    """A random presentation order for the four screening scenarios.

    Reproduces the source survey's BlockRandomizer (SubSet 4, even
    presentation): all four scenarios, shuffled per participant. Drawn on the
    server at session creation and handed to the client once, so the order is
    recorded even though the client renders the pages.

    `rng` may be seeded in tests; production passes None.
    """
    keys = scenarios.scenario_keys()
    order = list(keys)
    (rng or random).shuffle(order)
    return order


def assign_arm(rng=None):
    """Pick a model arm by simple uniform random draw.

    Simple randomization was chosen over balanced assignment, so cell sizes will
    drift somewhat; the export records the realized counts.

    MUST stay server-side. This is drawn at job creation — the moment of
    eligibility, so screened-out participants consume no arm slot — and stored
    on the session, so a participant who reloads or retries keeps their arm.
    """
    return (rng or random).choice(list(ARMS))


# --- Free-text validation --------------------------------------------------

def clean_text(raw, field_name, min_chars, required=True):
    """Normalize and validate one free-text answer.

    Strips surrounding whitespace, enforces a minimum length (so a participant
    cannot advance past the two substantive prompts with "a"), and caps the
    length at config.MAX_TEXT_CHARS.

    Inputs: `raw` may be None. `field_name` appears in the error message.
    Returns the cleaned string, or None when `required` is False and the input
    was empty.
    Raises ValidationError on a rejected value.
    """
    text = (raw or "").strip()
    if not text:
        if required:
            raise ValidationError("Please answer the %s question before continuing." % field_name)
        return None
    if len(text) > config.MAX_TEXT_CHARS:
        raise ValidationError(
            "Your answer is too long (limit %d characters)." % config.MAX_TEXT_CHARS)
    if len(text) < min_chars:
        raise ValidationError(
            "Please write a bit more — at least %d characters." % min_chars)
    return text


def validate_model_input(scenario_key, description, perspective):
    """Check the two texts that become the model's user turn.

    These are the ONLY participant answers the server sees before the end of the
    study, because they are the only ones it needs: prompts.build_user_turn is
    composed from exactly these two. Everything else rides along in the final
    payload.

    Returns (scenario, cleaned_description, cleaned_perspective).
    Raises ValidationError on anything the client should have caught.
    """
    if scenario_key not in scenarios.SCENARIOS_BY_KEY:
        raise ValidationError("Unknown scenario.")
    cleaned_description = clean_text(
        description, "description", config.MIN_DESCRIBE_CHARS, required=True)
    cleaned_perspective = clean_text(
        perspective, "perspective", config.MIN_PERSPECTIVE_CHARS, required=True)
    return (scenarios.SCENARIOS_BY_KEY[scenario_key], cleaned_description, cleaned_perspective)


# --- Submission validation -------------------------------------------------

def _coerce_choice(question, raw_value):
    """Map a submitted value onto one of a question's declared option values.

    Accepts the int or its string form (JSON round-trips can yield either) and
    rejects anything not in the option list, so a tampered or stale client
    cannot write an out-of-range code into the dataset.
    """
    allowed = {option["value"] for option in question["options"]}
    if isinstance(raw_value, bool):
        raise ValidationError("Invalid answer for '%s'." % question["id"])
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped.lstrip("-").isdigit():
            raise ValidationError("Invalid answer for '%s'." % question["id"])
        raw_value = int(stripped)
    if not isinstance(raw_value, int) or raw_value not in allowed:
        raise ValidationError("Invalid answer for '%s'." % question["id"])
    return raw_value


def _question_for_response_key():
    """Map every questionnaire response key to the question that owns it."""
    owners = {}
    for question in questionnaires.all_questions():
        for key in questionnaires.response_keys(question):
            owners[key] = question
    return owners


def validate_questionnaire_answers(answers):
    """Normalize the questionnaire block of a submission.

    Inputs: `answers` — dict keyed by response key. Matrix items are keyed by
    item id; every other shape by question id.

    Returns (normalized_answers, missing_keys).

    TWO DIFFERENT FAILURE MODES, HANDLED DIFFERENTLY ON PURPOSE:

      * An out-of-domain value is a tampered or broken client, so it RAISES.
        Accepting it would put a code into the dataset that no option maps to.
      * A missing answer is REPORTED, not raised. Every question is required and
        static/widgets.js blocks Continue until it is answered, so a gap here
        means a client bug — but this runs at the very end of the study, and
        rejecting the payload would throw away a completed session over one
        blank cell. The gap is recorded as `missing_keys` on the stored row so
        analysis can see it instead of guessing.

    Unknown keys are dropped rather than stored, so a stale client cannot invent
    columns.
    """
    if not isinstance(answers, dict):
        raise ValidationError("Malformed questionnaire answers.")
    owners = _question_for_response_key()
    normalized = {}
    missing = []
    for key in questionnaires.all_response_keys():
        raw = answers.get(key)
        if raw is None or raw == "":
            missing.append(key)
            continue
        normalized[key] = _coerce_choice(owners[key], raw)
    return normalized, missing


def validate_screening_answers(raw_answers, screening_order):
    """Normalize the screening block of a submission.

    Inputs: `raw_answers` — list of {scenario_key, similarity, description}
    dicts as the client recorded them, and the session's `screening_order` so
    the presentation position can be recovered.

    Returns a list of dicts with `scenario_key`, `position`, `similarity` and
    `description` (None when the answer did not require one).

    Raises ValidationError on an unknown scenario or similarity value: those are
    closed vocabularies the client renders from the server's own bundle, so
    anything else is tampering rather than a participant mistake.
    """
    if not isinstance(raw_answers, list):
        raise ValidationError("Malformed screening answers.")
    valid_similarity = {choice["value"] for choice in scenarios.SIMILARITY_CHOICES}
    seen = set()
    out = []
    for entry in raw_answers:
        if not isinstance(entry, dict):
            raise ValidationError("Malformed screening answer.")
        key = entry.get("scenario_key")
        if key not in scenarios.SCENARIOS_BY_KEY:
            raise ValidationError("Unknown scenario in screening answers.")
        if key in seen:
            raise ValidationError("Duplicate screening answer for %r." % (key,))
        seen.add(key)
        similarity = entry.get("similarity")
        if similarity not in valid_similarity:
            raise ValidationError("Unknown similarity value in screening answers.")
        description = None
        if scenarios.describe_required(similarity):
            # Not required to be long here: the copy that mattered was already
            # length-checked at /api/response/start for the selected scenario.
            # A short one on a non-selected scenario is data, not an error.
            description = (entry.get("description") or "").strip()[:config.MAX_TEXT_CHARS] or None
        out.append({
            "scenario_key": key,
            "position": screening_order.index(key) if key in screening_order else None,
            "similarity": similarity,
            "description": description,
        })
    return out


def validate_submission(payload, screening_order):
    """Validate and normalize a whole end-of-study payload.

    Inputs: the client's `payload` dict and the session's `screening_order`.

    Returns a normalized dict with these keys:
      consent, prolific_pid, eligibility, selected_scenario, screening,
      perspective_text, additional_thoughts, answers, missing_keys, client_timings

    Raises ValidationError on structurally invalid input. Missing questionnaire
    answers are reported in `missing_keys` rather than raised — see
    validate_questionnaire_answers for why.
    """
    if not isinstance(payload, dict):
        raise ValidationError("Malformed submission.")

    consent = payload.get("consent")
    if consent not in ("agree", "decline"):
        raise ValidationError("Missing or invalid consent.")

    eligibility = payload.get("eligibility")
    if eligibility not in ("eligible", "screened_out", "declined"):
        raise ValidationError("Missing or invalid eligibility.")

    selected = payload.get("selected_scenario")
    if selected is not None and selected not in scenarios.SCENARIOS_BY_KEY:
        raise ValidationError("Unknown selected scenario.")
    if eligibility == "eligible" and selected is None:
        raise ValidationError("An eligible session must name a selected scenario.")

    answers, missing = validate_questionnaire_answers(payload.get("answers") or {})
    timings = payload.get("client_timings")
    if timings is not None and not isinstance(timings, dict):
        raise ValidationError("Malformed client timings.")

    return {
        "consent": consent,
        "prolific_pid": ((payload.get("prolific_pid") or "").strip()[:200]) or None,
        "eligibility": eligibility,
        "selected_scenario": selected,
        "screening": validate_screening_answers(payload.get("screening") or [], screening_order),
        "perspective_text": (payload.get("perspective_text") or "").strip()[
            :config.MAX_TEXT_CHARS] or None,
        "additional_thoughts": (payload.get("additional_thoughts") or "").strip()[
            :config.MAX_TEXT_CHARS] or None,
        "answers": answers,
        "missing_keys": missing,
        "client_timings": timings or {},
    }
