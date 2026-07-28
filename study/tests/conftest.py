"""Shared pytest fixtures for the study's HTTP-level tests.

Two fixtures:

  client    — a FastAPI TestClient bound to a throwaway SQLite file.
  submitted — the rows /api/submit would have sent to Supabase, captured in a
              list instead of posted. Every test that walks to the end of the
              study asserts against this, because it is now the ONLY place
              participant answers land.

`submitted` also redirects the durability fallback to a temp file, so the tests
exercise the real supabase_client.submit path (local append, then post) without
touching the repo's data/ directory or the network.

The walk-the-flow helpers live in flow_helpers.py rather than here, because they
are ordinary functions and importing them explicitly makes it obvious which
module a test's scaffolding came from.
"""

import os
import sys
import tempfile

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS_DIR))
# So test modules can `from flow_helpers import ...` regardless of the cwd
# pytest was invoked from.
sys.path.insert(0, _TESTS_DIR)

import config
import store
import supabase_client


@pytest.fixture()
def client(monkeypatch):
    """A TestClient on a fresh temporary database, torn down afterwards."""
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    os.unlink(path)
    monkeypatch.setattr(config, "DATABASE_URL", "sqlite:///" + path)
    monkeypatch.setattr(config, "WORKER_LONGPOLL_SECONDS", 1)
    store.init_db()

    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as test_client:
        yield test_client
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.unlink(path + suffix)


@pytest.fixture()
def submitted(monkeypatch, tmp_path):
    """Capture rows destined for Supabase; returns the list they land in.

    Patches only the HTTP call, so append_fallback still runs for real against a
    temp file — that ordering (disk first, network second) is the durability
    guarantee, and a test that stubbed out the whole of submit() would not
    exercise it.
    """
    rows = []
    monkeypatch.setattr(config, "SUBMISSION_FALLBACK_PATH",
                        str(tmp_path / "submissions.jsonl"))
    monkeypatch.setattr(config, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(config, "SUPABASE_SERVICE_KEY", "test-service-key")

    def fake_post(row):
        rows.append(row)
        return None

    monkeypatch.setattr(supabase_client, "post_row", fake_post)
    return rows
