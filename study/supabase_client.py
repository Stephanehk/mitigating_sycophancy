"""Ships a completed session's payload to Supabase.

One participant, one row, one insert, at the end of the study. The service-role
key is read from the environment and used only here, server-side, so it never
reaches a browser.

DURABILITY ORDER — this is the important part of the module. Every submission is
appended to a local JSONL file BEFORE the network call. If Supabase is down, the
participant still finishes, the row is on disk, and
`analysis/backfill_supabase.py` replays it later. The alternative orderings both
lose data: posting first means a crash between the post and the local write
leaves no local trace, and posting *instead* of writing locally means an outage
silently drops completed sessions, which under the completers-only design is the
only copy that will ever exist.

ERROR-HANDLING NOTE. The repo's default is to let code fail fast and this module
mostly does — a malformed row, a missing config, a bad path all raise. The one
deliberate exception is the HTTP call itself, wrapped so that a Supabase outage
cannot strand a participant on the last page of a study they have already
completed. The failure is recorded on the returned dict and logged loudly; it is
never swallowed silently.
"""

import json
import os
import sys

import requests

import config


def _fallback_path():
    """The local JSONL path, with its directory created."""
    path = config.SUBMISSION_FALLBACK_PATH
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def append_fallback(row):
    """Append one submission row to the local JSONL file.

    Called before every Supabase attempt. One JSON object per line, flushed and
    fsynced, so a hard kill cannot leave a half-written record.

    Inputs: `row` — the dict destined for Supabase. Must be JSON-serializable.
    Returns the path written to.
    """
    path = _fallback_path()
    line = json.dumps(row, ensure_ascii=False, sort_keys=True)
    assert "\n" not in line, "serialized row must be one line"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def is_configured():
    """Whether a Supabase destination is set up. False means fallback-only."""
    return bool(config.SUPABASE_URL and config.SUPABASE_SERVICE_KEY)


def _endpoint():
    return "%s/rest/v1/%s" % (config.SUPABASE_URL.rstrip("/"), config.SUPABASE_TABLE)


def post_row(row):
    """POST one row to the Supabase REST endpoint.

    Returns None on success; a short error string on failure. Does not raise on
    a network or HTTP error — see the module note.

    Assumption: the table has a UNIQUE constraint on session_id (see
    supabase_schema.sql), so a duplicate insert is rejected by Postgres rather
    than silently doubling a participant.
    """
    if not is_configured():
        return "supabase not configured"

    headers = {
        "apikey": config.SUPABASE_SERVICE_KEY,
        "Authorization": "Bearer %s" % config.SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
        # Keeps the response small; we only care whether it was accepted.
        "Prefer": "return=minimal",
    }
    try:
        response = requests.post(_endpoint(), headers=headers, json=row,
                                 timeout=config.SUPABASE_TIMEOUT_SECONDS)
    except requests.RequestException as error:
        return "request failed: %s" % (error,)
    if response.status_code not in (200, 201, 204):
        return "http %d: %s" % (response.status_code, response.text[:500])
    return None


def submit(row):
    """Persist one completed session: local file first, then Supabase.

    Returns {"stored_locally": path, "supabase_error": str|None}. The caller
    should treat a non-None `supabase_error` as "recoverable, replay later",
    not as a participant-facing failure.
    """
    path = append_fallback(row)
    error = post_row(row)
    if error is not None:
        print("SUPABASE SUBMIT FAILED for session %s: %s (row is in %s)"
              % (row.get("session_id"), error, path), file=sys.stderr, flush=True)
    return {"stored_locally": path, "supabase_error": error}
