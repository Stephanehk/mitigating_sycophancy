"""Replay locally-stored submissions into Supabase.

The server appends every completed submission to a JSONL file BEFORE posting it
(see supabase_client.submit), so a Supabase outage leaves the row on disk and
the participant unaffected. This script pushes those rows up afterwards.

Safe to run repeatedly. The Supabase table has a UNIQUE constraint on
session_id, so a row that already made it is rejected by Postgres with a 409 and
reported as "already present" rather than duplicating a participant.

Usage:
    python study/analysis/backfill_supabase.py --dry-run
    python study/analysis/backfill_supabase.py
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY_DIR)

import config
import supabase_client


def load_rows(path):
    """Read the fallback JSONL, keeping the LAST row per session_id.

    The file is append-only and a replayed submission can appear more than once;
    the last write is the authoritative one.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=config.SUBMISSION_FALLBACK_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be pushed without writing")
    args = parser.parse_args()

    assert supabase_client.is_configured(), (
        "Supabase is not configured; set STUDY_SUPABASE_URL and "
        "STUDY_SUPABASE_SERVICE_KEY.")

    rows = load_rows(args.path)
    print("%d unique sessions in %s" % (len(rows), args.path))
    if args.dry_run:
        for row in rows:
            print("  would push %s (arm=%s)" % (row["session_id"], row.get("arm")))
        return

    pushed = 0
    already = 0
    failed = []
    for row in rows:
        error = supabase_client.post_row(row)
        if error is None:
            pushed += 1
        elif "http 409" in error:
            # UNIQUE(session_id) rejected it: this row is already up there.
            already += 1
        else:
            failed.append((row["session_id"], error))

    print("pushed:  %d" % pushed)
    print("already: %d" % already)
    print("failed:  %d" % len(failed))
    for session_id, error in failed:
        print("  %s: %s" % (session_id, error))
    # A failure here is worth a non-zero exit so a wrapper script notices.
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
