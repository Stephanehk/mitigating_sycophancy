"""Shared fixtures. Everything that touches data now talks to the real database.

There is no application server and no local datastore to fake, so the tests that
matter drive the actual Supabase RPCs with the actual keys. That is a deliberate
change of testing strategy, not just a mechanical port.

WHY. The suite this replaces mocked the network and passed while production was
broken — twice. `test_a_screened_out_participant_still_submits` asserted
`arm is None` and passed, while a `not null` constraint would have rejected
every screen-out. And the whole security model was only ever exercised with the
service_role key, which bypasses RLS and grants, so the anon key could read the
worker secret and nobody noticed. Mocks cannot catch either class of bug.

Two clients, and which one a test uses is itself meaningful:

  anon     the PUBLIC key, exactly what ships in the page and runs on the GPU
           worker. Every participant-behaviour test uses this, so the tests
           exercise the same permissions a real browser has.
  service  privileged, bypasses RLS. Used ONLY to inspect what landed and to
           clean up — never to perform an action under test, because doing so
           would prove nothing about what a participant can actually do.

Tests are skipped when credentials are absent, so the suite still runs offline;
the parts that run offline are the ones that read source files (prompts, the
client contract, the export).
"""

import os
import sys
import uuid

import pytest
import requests

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS_DIR))
sys.path.insert(0, _TESTS_DIR)

import config

#: PostgREST paths, read from config so this fixture always cleans the tables
#: the app actually writes to (see config.TABLE_SUFFIX). A stale literal here
#: would leave every test row behind in the live study.
SESSIONS = config.TABLE_SESSIONS
SUBMISSIONS = config.TABLE_SUBMISSIONS


def supabase_configured():
    return bool(config.SUPABASE_URL and config.SUPABASE_ANON_KEY
                and config.SUPABASE_SERVICE_KEY)


class Client:
    """Minimal PostgREST client. `rpc` raises on failure; `try_rpc` does not."""

    def __init__(self, key):
        assert key, "no key"
        self.base = config.SUPABASE_URL.rstrip("/") + "/rest/v1"
        self.headers = {"apikey": key, "Authorization": "Bearer " + key,
                        "Content-Type": "application/json"}

    def try_rpc(self, name, args=None):
        """The raw response, so a test can assert on a REFUSAL."""
        return requests.post("%s/rpc/%s" % (self.base, name), headers=self.headers,
                             json=args or {}, timeout=60)

    def rpc(self, name, args=None):
        response = self.try_rpc(name, args)
        assert response.status_code in (200, 204), (
            "%s -> %d %s" % (name, response.status_code, response.text[:400]))
        return response.json() if response.text.strip() else None

    def get(self, path):
        response = requests.get(self.base + path, headers=self.headers, timeout=60)
        assert response.status_code == 200, (
            "GET %s -> %d %s" % (path, response.status_code, response.text[:400]))
        return response.json()

    def try_get(self, path):
        return requests.get(self.base + path, headers=self.headers, timeout=60)

    def delete(self, path):
        requests.delete(self.base + path, headers=self.headers, timeout=60)


@pytest.fixture(scope="session")
def anon():
    """The public key — what a participant's browser and the worker actually hold."""
    if not supabase_configured():
        pytest.skip("Supabase not configured")
    return Client(config.SUPABASE_ANON_KEY)


@pytest.fixture(scope="session")
def service():
    """Privileged. Inspection and cleanup only, never the action under test."""
    if not supabase_configured():
        pytest.skip("Supabase not configured")
    return Client(config.SUPABASE_SERVICE_KEY)


@pytest.fixture(scope="session")
def worker_token():
    if not os.path.exists(os.path.join(os.path.dirname(_TESTS_DIR), ".trial_token")):
        pytest.skip("no .trial_token to authenticate a worker with")
    with open(os.path.join(os.path.dirname(_TESTS_DIR), ".trial_token")) as handle:
        return handle.read().strip()


@pytest.fixture()
def clean(service):
    """Delete every row a test creates, identified by diffing session ids.

    Ids are generated inside the code under test and cannot be predicted here,
    so the fixture snapshots before and removes the difference afterwards. Jobs
    and submissions cascade off the session's foreign key.

    Runs even when a test fails, so a red suite cannot leave rows in the same
    tables the real study writes to.
    """
    before = {row["session_id"] for row in service.get("/" + SESSIONS + "?select=session_id")}
    subs_before = {row["session_id"] for row in
                   service.get("/" + SUBMISSIONS + "?select=session_id")}
    yield
    for session_id in ({row["session_id"] for row in
                        service.get("/" + SUBMISSIONS + "?select=session_id")} - subs_before):
        service.delete("/" + SUBMISSIONS + "?session_id=eq.%s" % session_id)
    for session_id in ({row["session_id"] for row in
                        service.get("/" + SESSIONS + "?select=session_id")} - before):
        service.delete("/" + SESSIONS + "?session_id=eq.%s" % session_id)


@pytest.fixture()
def tag():
    """A unique marker for rows a test creates directly."""
    return "test-" + uuid.uuid4().hex[:10]
