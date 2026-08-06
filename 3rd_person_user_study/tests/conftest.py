"""Shared fixtures.

Everything here runs OFFLINE. There is no Supabase call in this suite: the
generated SQL is checked structurally, against the item set it was generated
from, rather than by executing it. That is a deliberate limit, and the reason is
in study/README.md — the study database is the live recruiting database, and DDL
against it is not something a test run should do.

`3rd_person_user_study` starts with a digit and so is not an importable package
name. Nothing imports it as one: the modules sit on sys.path directly, exactly
as study/'s do.
"""

import json
import os
import subprocess
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_STUDY_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _STUDY_DIR)
sys.path.insert(0, os.path.join(_STUDY_DIR, "tools"))

import config  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    """The committed item set. Skips rather than fails when it is absent, so a
    fresh checkout that has not run build_items.py yet still gets a green
    generator suite."""
    if not os.path.exists(config.ITEMS_PATH):
        pytest.skip(f"{config.ITEMS_PATH} not built")
    with open(config.ITEMS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def manifest():
    if not os.path.exists(config.MANIFEST_PATH):
        pytest.skip(f"{config.MANIFEST_PATH} not built")
    with open(config.MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def schema_sql():
    """The generated schema, regenerated in-process rather than read off disk.

    Reading the committed file would test the file; generating it tests the
    generator, and a committed file that has fallen behind is caught separately
    by test_committed_schema_is_current.
    """
    if not os.path.exists(config.ITEMS_PATH):
        pytest.skip(f"{config.ITEMS_PATH} not built")
    script = os.path.join(_STUDY_DIR, "tools", "generate_supabase_sql.py")
    return subprocess.run([sys.executable, script], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture(scope="session")
def schema_sql_pref():
    """The committed schema, read off disk. Distinct from `schema_sql`, which
    regenerates in-process to test the generator: here the question is what the
    file you paste into Supabase actually says."""
    path = os.path.join(_STUDY_DIR, "supabase_schema.sql")
    if not os.path.exists(path):
        pytest.skip("supabase_schema.sql not generated")
    with open(path, encoding="utf-8") as handle:
        return handle.read()
