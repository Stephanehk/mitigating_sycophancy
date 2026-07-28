"""Datastore facade: picks a backend from STUDY_DATABASE_URL at call time.

Two backends implement the same contract:

  store_sqlite.py    — a local file. Local development and the test suite. One
                       always-on process only; it cannot back a Lambda fleet
                       because invocations share no filesystem.
  store_supabase.py  — Postgres via PostgREST. Production.

Callers import `store` and never know which is live. `server.py`, the worker
and the export are all written against this one surface.

WHY A LAZY `__getattr__` RATHER THAN AN IMPORT-TIME CHOICE. Tests point
`config.DATABASE_URL` at a fresh temporary file AFTER this module is imported,
so resolving the backend at import time would pin every test to whatever the
environment happened to say. Resolving per call costs a dict lookup and keeps
the tests honest.

NEITHER BACKEND STORES PARTICIPANT ANSWERS. Those live in the browser for the
duration of the session and go to Supabase in one payload at the end; see
supabase_client.py and README §1b.
"""

import config
# Pure helpers, backend-independent, re-exported so callers can use
# store.utcnow() without caring which backend is live.
from store_common import in_seconds, new_id, utcnow  # noqa: F401

#: URL schemes that select the Supabase backend. Anything else is treated as a
#: local file path for SQLite, which keeps the existing "sqlite:///..." default
#: working untouched.
_SUPABASE_SCHEMES = ("supabase://", "postgresql://", "postgres://")


def backend():
    """The store module matching the configured DATABASE_URL.

    Exposed (not underscore-private) because tests and ops scripts legitimately
    want to assert which backend is live before touching real data.
    """
    if config.DATABASE_URL.startswith(_SUPABASE_SCHEMES):
        import store_supabase
        return store_supabase
    import store_sqlite
    return store_sqlite


def backend_name():
    """'supabase' or 'sqlite'. For /api/health and startup logging."""
    return "supabase" if config.DATABASE_URL.startswith(_SUPABASE_SCHEMES) else "sqlite"


def __getattr__(name):
    """Delegate every datastore call to the live backend (PEP 562).

    Raises AttributeError for anything neither backend defines, so a typo'd
    call site still fails loudly rather than silently returning None.
    """
    return getattr(backend(), name)
