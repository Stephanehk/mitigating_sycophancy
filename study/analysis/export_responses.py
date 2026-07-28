"""Export the study's submissions to tidy CSVs for analysis.

Reads from Supabase (or, with --source local, the server's fallback JSONL) and
writes the same three files the SQLite-backed exporter used to:

  participants.csv  — one row per submission: arm, eligibility, the selected
                      scenario, the participant's own text, the exact response
                      they were shown, timings, and one column per
                      questionnaire response key.
  screening.csv     — one row per (session, scenario) screening answer, so the
                      screening funnel and the per-scenario match rate are
                      recoverable.
  generations.csv   — one row per generation, including per-stage outputs, so
                      the neutral arm's pre-edit Stage 1 text is available.

WHAT THIS CANNOT SHOW YOU. Only participants who reached a terminal page are in
Supabase at all — the browser holds the payload until then and abandoners post
nothing. So this export characterises completers, screen-outs and decliners, but
NOT mid-study dropout. `/api/health`'s `saw_response_never_submitted` counter is
the only signal for that, and it is a count, not data.

This script never fabricates or imputes a value. A question a participant did
not answer is written as an empty cell, and the column set is derived from
questionnaires.py rather than from whatever happens to be in the data, so a
missing item is visible as an empty column rather than a silently absent one.

Usage:
    python study/analysis/export_responses.py --out-dir study/data/export
    python study/analysis/export_responses.py --source local     # fallback JSONL
"""

import argparse
import csv
import datetime
import json
import os
import sys

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY_DIR)

import config
import questionnaires
import supabase_client

# Columns describing the session itself, before the questionnaire columns.
SESSION_COLUMNS = [
    "session_id", "prolific_pid", "arm", "eligibility", "consent",
    "selected_scenario", "screening_order", "n_screening_answered",
    "perspective_text", "perspective_chars", "shown_response",
    "shown_response_chars", "worker_attempts", "manual_retries",
    "session_created_at", "submitted_at", "duration_seconds",
    "attention_check_passed", "aias_attention_passed", "n_missing_answers",
]


def fetch_supabase(page_size=1000):
    """Read every submission row out of Supabase, paging until exhausted.

    Assumption: the service-role key is configured (config.SUPABASE_*). Fails
    loudly rather than returning a partial export.
    """
    assert supabase_client.is_configured(), (
        "Supabase is not configured; set STUDY_SUPABASE_URL and "
        "STUDY_SUPABASE_SERVICE_KEY, or pass --source local.")
    endpoint = "%s/rest/v1/%s" % (config.SUPABASE_URL.rstrip("/"), config.SUPABASE_TABLE)
    headers = {
        "apikey": config.SUPABASE_SERVICE_KEY,
        "Authorization": "Bearer %s" % config.SUPABASE_SERVICE_KEY,
    }
    rows = []
    offset = 0
    while True:
        response = requests.get(
            endpoint, headers=headers,
            params={"select": "*", "order": "id.asc",
                    "offset": offset, "limit": page_size},
            timeout=config.SUPABASE_TIMEOUT_SECONDS)
        assert response.status_code == 200, (
            "Supabase read failed: HTTP %d %s" % (response.status_code, response.text[:500]))
        batch = response.json()
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        offset += page_size


def read_local(path):
    """Read submissions from the server's fallback JSONL, newest wins per session.

    The file is append-only and a replayed submission can appear twice, so later
    lines override earlier ones for the same session_id.
    """
    assert os.path.exists(path), "no fallback file at %r" % (path,)
    by_session = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert "session_id" in row, "line %d has no session_id" % (number,)
            by_session[row["session_id"]] = row
    return list(by_session.values())


def _parse_iso(value):
    """Parse a stored timestamp into a datetime, or None.

    Accepts both the server's "...%fZ" form and the timestamptz form Supabase
    returns, because rows can come from either source.
    """
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo is None else parsed


def _duration_seconds(start, end):
    """Whole seconds between two stored timestamps, or None if either is absent."""
    first, last = _parse_iso(start), _parse_iso(end)
    if first is None or last is None:
        return None
    if (first.tzinfo is None) != (last.tzinfo is None):
        return None
    return round((last - first).total_seconds(), 1)


def _scale_point_lookup():
    """Map (response_key, stored value) -> 1-based scale position.

    Needed because several QSF questions skip choice ids (response_objectivity
    stores 7 and 8 for scale points 6 and 7). Analyses want the position, but
    the raw code is what preserves comparability with the source study's data,
    so both are exported.

    Keyed by response key rather than question id so matrix items resolve
    directly, without a second lookup back to their owning question.
    """
    lookup = {}
    for question in questionnaires.all_questions():
        for key in questionnaires.response_keys(question):
            for option in question["options"]:
                if "scale_point" in option:
                    lookup[(key, option["value"])] = option["scale_point"]
    return lookup


def _attention_passed(answers, item_prefix, expected):
    """Whether every scored row of an attention check has the expected value.

    Returns None when the participant skipped the check entirely, so a skip is
    distinguishable from a failure.
    """
    keys = [key for key in answers if key.startswith(item_prefix)]
    if not keys:
        return None
    return all(answers[key] == expected for key in keys)


def build_participant_rows(submissions):
    """Assemble the participants.csv rows and their column order."""
    scale_points = _scale_point_lookup()
    response_keys = questionnaires.all_response_keys()

    # A parallel *_point column is emitted for every key on a numeric scale,
    # because the raw stored code is a QSF choice id and is not always the
    # scale position (see questionnaires.py).
    scaled_keys = []
    for question in questionnaires.all_questions():
        if not any("scale_point" in option for option in question["options"]):
            continue
        scaled_keys.extend(questionnaires.response_keys(question))

    columns = (SESSION_COLUMNS + response_keys + ["additional_thoughts"]
               + ["%s_point" % key for key in scaled_keys])

    rows = []
    for submission in submissions:
        answers = submission.get("answers") or {}
        order = submission.get("screening_order") or []
        row = {
            "session_id": submission["session_id"],
            "prolific_pid": submission.get("prolific_pid") or "",
            "arm": submission.get("arm") or "",
            "eligibility": submission.get("eligibility") or "",
            "consent": submission.get("consent") or "",
            "selected_scenario": submission.get("selected_scenario") or "",
            "screening_order": "|".join(order),
            "n_screening_answered": len(submission.get("screening") or []),
            "perspective_text": submission.get("perspective_text") or "",
            "perspective_chars": len(submission.get("perspective_text") or ""),
            "shown_response": submission.get("shown_response") or "",
            "shown_response_chars": len(submission.get("shown_response") or ""),
            "worker_attempts": submission.get("worker_attempts"),
            "manual_retries": submission.get("manual_retries"),
            "session_created_at": submission.get("session_created_at") or "",
            "submitted_at": submission.get("submitted_at") or "",
            "duration_seconds": _duration_seconds(
                submission.get("session_created_at"), submission.get("submitted_at")),
            "attention_check_passed": _attention_passed(answers, "attention_check_", expected=1),
            "aias_attention_passed": (
                None if "AIAS_5" not in answers else answers["AIAS_5"] == 10),
            "n_missing_answers": len(submission.get("missing_keys") or []),
            "additional_thoughts": submission.get("additional_thoughts") or "",
        }

        for key in response_keys:
            row[key] = answers.get(key, "")

        for key in scaled_keys:
            value = answers.get(key)
            row["%s_point" % key] = ("" if value is None
                                     else scale_points.get((key, value), ""))

        rows.append(row)
    return columns, rows


def build_screening_rows(submissions):
    """One row per screening answer across all submissions."""
    columns = ["session_id", "arm", "eligibility", "position", "scenario_key",
               "similarity", "selected", "description", "description_chars"]
    rows = []
    for submission in submissions:
        for answer in submission.get("screening") or []:
            rows.append({
                "session_id": submission["session_id"],
                "arm": submission.get("arm") or "",
                "eligibility": submission.get("eligibility") or "",
                "position": answer.get("position"),
                "scenario_key": answer["scenario_key"],
                "similarity": answer["similarity"],
                "selected": answer["scenario_key"] == submission.get("selected_scenario"),
                "description": answer.get("description") or "",
                "description_chars": len(answer.get("description") or ""),
            })
    return columns, rows


def build_generation_rows(submissions):
    """One row per generation, with per-stage outputs broken out."""
    columns = ["session_id", "job_id", "arm", "topic_title", "user_text",
               "response_text", "response_chars", "stage1_output", "stage2_output",
               "n_stages", "worker_attempts", "manual_retries"]
    rows = []
    for submission in submissions:
        if not submission.get("job_id"):
            continue
        stages = submission.get("stages") or []
        rows.append({
            "session_id": submission["session_id"],
            "job_id": submission["job_id"],
            "arm": submission.get("arm") or "",
            "topic_title": submission.get("topic_title") or "",
            "user_text": submission.get("user_text") or "",
            "response_text": submission.get("shown_response") or "",
            "response_chars": len(submission.get("shown_response") or ""),
            "stage1_output": stages[0].get("output", "") if len(stages) > 0 else "",
            "stage2_output": stages[1].get("output", "") if len(stages) > 1 else "",
            "n_stages": len(stages),
            "worker_attempts": submission.get("worker_attempts"),
            "manual_retries": submission.get("manual_retries"),
        })
    return columns, rows


def write_csv(path, columns, rows):
    """Write rows to `path`. Fails if a row carries a key not in `columns`."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join(_STUDY_DIR, "data", "export"))
    parser.add_argument("--source", choices=("supabase", "local"), default="supabase",
                        help="Read from Supabase (default) or the server's fallback JSONL")
    parser.add_argument("--local-path", default=config.SUBMISSION_FALLBACK_PATH)
    parser.add_argument("--eligible-only", action="store_true",
                        help="Drop screen-outs and decliners")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.source == "supabase":
        submissions = fetch_supabase()
        source_label = "%s/%s" % (config.SUPABASE_URL, config.SUPABASE_TABLE)
    else:
        submissions = read_local(args.local_path)
        source_label = args.local_path

    total = len(submissions)
    if args.eligible_only:
        submissions = [s for s in submissions if s.get("eligibility") == "eligible"]

    columns, rows = build_participant_rows(submissions)
    n_participants = write_csv(os.path.join(args.out_dir, "participants.csv"), columns, rows)

    columns, screening = build_screening_rows(submissions)
    n_screening = write_csv(os.path.join(args.out_dir, "screening.csv"), columns, screening)

    columns, generations = build_generation_rows(submissions)
    n_generations = write_csv(os.path.join(args.out_dir, "generations.csv"), columns, generations)

    print("source:       %s" % source_label)
    print("submissions:  %d read, %d exported" % (total, n_participants))
    print("screening:    %d answers" % n_screening)
    print("generations:  %d responses" % n_generations)
    print("out-dir:      %s" % args.out_dir)


if __name__ == "__main__":
    main()
