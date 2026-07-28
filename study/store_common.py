"""Helpers shared by both store backends.

Lives in its own module so store.py, store_sqlite.py and store_supabase.py can
all use them without importing each other (store.py imports the backends, so a
backend importing store.py would be circular).

Timestamps are ISO-8601 UTC strings with a 'Z' suffix. They sort
lexicographically in the same order they sort chronologically, which is what
lets the SQLite backend's lease and deadline comparisons happen in SQL. The
Supabase backend stores real timestamptz columns and Postgres does the
comparing, so there the format matters only at the API boundary.
"""

import datetime
import uuid


def utcnow():
    """Current UTC time as an ISO-8601 string with a 'Z' suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def in_seconds(seconds):
    """An ISO-8601 UTC timestamp `seconds` in the future."""
    moment = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_id():
    """A fresh opaque identifier (session ids, job ids, tokens)."""
    return uuid.uuid4().hex
