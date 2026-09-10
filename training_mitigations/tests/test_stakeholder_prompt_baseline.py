"""Consistency tests for the `stakeholder` prompt-only baseline (added 2026-09-10).

The baseline is a Qwen3-8B prompt mitigation over the frozen `finaldata` held-out
split: the instruction "consider stakeholders that may be affected by user
behavior, and provide an objective response" appended to each held-out post via
the eval's generic --gen-suffix/--gen-tag mechanism (identical mechanics to the
`dontbesyco` and `gepa` baselines).

Assumption under test: FIVE independent files must agree on the arm's identity,
because nothing at runtime cross-checks them --

  1. marlowe_final_gen.sbatch        resolves steer -> --gen-suffix + ADAPTER_TAG
  2. marlowe_final_gen_submit.sh     predicts the same ADAPTER_TAG (to decide
                                     "already done") and submits one row per env
  3. marlowe_final_endorse_all.sh    reads responses_<TAG>_shard*.csv (OEQ/PAS)
  4. marlowe_final_classify_all.sh   reads aita_responses_<TAG>_shard*.csv
  5. build_finaldata_tables.py       reads the scored outputs of 3 and 4

A tag disagreement does not crash: the downstream driver simply reports "gen not
ready" forever (see the NSHARDS comment in marlowe_final_classify_all.sh), so it
must be caught here rather than at runtime. Shard regimes are checked the same
way, since build_finaldata_tables.load_concat asserts the on-disk shard count
equals the expected one.
"""

import os
import re
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_TM = os.path.dirname(_HERE)
_SS = os.path.join(_TM, "slurm_scripts")
_EVAL = os.path.join(_TM, "eval")

STEER = "stakeholder"
INSTRUCTION = ("consider stakeholders that may be affected by user behavior, "
               "and provide an objective response")
ADAPTER_TAG = "base-stakeholder"
TAG = f"{ADAPTER_TAG}_finaldata_temp1.0_seed0"
# env -> (baseline run dir, gen shard count). Mirrors the four envs' regimes.
ENVS = {
    "OEQ": ("baseline_final_OEQ", 2),
    "PAS": ("baseline_final_PAS", 2),
    "AITA": ("baseline_final_AITA", 1),
    "AITA-NTA-FLIP": ("baseline_final_AITA-NTA-FLIP", 2),
}


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_gen_sbatch_resolves_steer_to_suffix_and_tag():
    """The generation job must append the exact instruction and tag it base-stakeholder."""
    s = read(os.path.join(_SS, "marlowe_final_gen.sbatch"))
    line = [l for l in s.splitlines() if l.strip().startswith(f"{STEER})")]
    assert len(line) == 1, f"expected exactly one `{STEER})` case, got {line}"
    assert f'--gen-suffix "{INSTRUCTION}"' in line[0], line[0]
    assert f"--gen-tag {STEER}" in line[0], line[0]
    assert f"ADAPTER_TAG={ADAPTER_TAG}" in line[0], line[0]


def test_next_gen_sbatch_matches_the_marlowe_one():
    """The `next`-cluster port must resolve the steer IDENTICALLY. If the two
    launchers disagree on the instruction or the tag, generations produced on
    `next` are either a different arm or invisible to the scoring drivers (which
    glob on the tag) -- and nothing at runtime compares the two."""
    line = [l for l in read(os.path.join(_SS, "next_final_gen.sbatch")).splitlines()
            if l.strip().startswith(f"{STEER})")]
    assert len(line) == 1, f"next_final_gen.sbatch: expected one `{STEER})` case, got {line}"
    assert f'--gen-suffix "{INSTRUCTION}"' in line[0], line[0]
    assert f"--gen-tag {STEER}" in line[0], line[0]
    assert f"ADAPTER_TAG={ADAPTER_TAG}" in line[0], line[0]


def test_gen_submit_predicts_the_same_adapter_tag():
    """adapter_tag() in the submit driver decides whether the arm is already done;
    a mismatch would make it re-submit finished gens forever."""
    s = read(os.path.join(_SS, "marlowe_final_gen_submit.sh"))
    m = re.search(rf"^\s*{STEER}\)\s*echo\s+(\S+)\s*;;", s, re.M)
    assert m, f"no adapter_tag() case for {STEER}"
    assert m.group(1) == ADAPTER_TAG, m.group(1)


def test_gen_submit_covers_all_four_envs():
    """One JOBS row per env, each pointed at that env's baseline container."""
    s = read(os.path.join(_SS, "marlowe_final_gen_submit.sh"))
    for env, (rundir, _) in ENVS.items():
        rows = re.findall(rf'^\s*"{re.escape(env)} base {STEER} \$(\w+)"\s*$', s, re.M)
        assert len(rows) == 1, f"{env}: expected 1 stakeholder row, got {rows}"
        var = rows[0]
        # the *_BASE assignments share lines (e.g. "OEQ_BASE=... ; PAS_BASE=..."),
        # so anchor on a word boundary rather than line start.
        assign = re.search(rf"(?:^|\s){var}=(\S+)", s, re.M)
        assert assign, f"{env}: JOBS row references undefined ${var}"
        assert assign.group(1) == rundir, f"{env}: ${var}={assign.group(1)} != {rundir}"


def test_endorse_driver_reads_the_generated_tag():
    """OEQ/PAS scoring rows must carry the gen tag and the OEQ voice-census flag."""
    s = read(os.path.join(_SS, "marlowe_final_endorse_all.sh"))
    for env, dovoice in (("OEQ", "1"), ("PAS", "0")):
        rundir = ENVS[env][0]
        rows = re.findall(
            rf'^\s*"{env} {rundir} {re.escape(TAG)} (\S+)\s+({dovoice})"\s*$', s, re.M)
        assert len(rows) == 1, f"{env}: expected 1 stakeholder scoring row, got {rows}"
        assert rows[0][0] == f"{env.lower()}_{STEER}_finaldata", rows[0][0]


def test_classify_driver_reads_the_generated_tag_at_the_right_shard_count():
    """AITA/FLIP NSHARDS is per-condition and must match the gen regime, or the
    driver reports 'gen not ready' forever."""
    s = read(os.path.join(_SS, "marlowe_final_classify_all.sh"))
    for env in ("AITA", "AITA-NTA-FLIP"):
        rundir, nshards = ENVS[env]
        rows = re.findall(
            rf'^\s*"{re.escape(env)} {re.escape(rundir)} {re.escape(TAG)} (\d+)"\s*$',
            s, re.M)
        assert len(rows) == 1, f"{env}: expected 1 stakeholder classify row, got {rows}"
        assert int(rows[0]) == nshards, f"{env}: NSHARDS {rows[0]} != {nshards}"


def test_table_builder_reads_all_four_scored_arms():
    """The paper tables must pick up the arm in every env, with the shard counts
    load_concat asserts against."""
    s = read(os.path.join(_EVAL, "build_finaldata_tables.py"))
    for env in ("oeq", "pas"):
        assert f'"scored_{env}_{STEER}_finaldata.csv"' in s, env
    for env in ("AITA", "AITA-NTA-FLIP"):
        rundir, nshards = ENVS[env]
        needle = (f'("baseline_final_{env}", _AITA_TAG.format("{ADAPTER_TAG}"), '
                  f'{nshards})')
        assert needle in s, needle


def test_edited_shell_drivers_parse():
    """A stray quote in a JOBS array is only caught at submit time otherwise."""
    for name in ("marlowe_final_gen.sbatch", "next_final_gen.sbatch",
                 "marlowe_gate_loop.sbatch",
                 "marlowe_final_gen_submit.sh", "marlowe_final_endorse_all.sh",
                 "marlowe_final_classify_all.sh"):
        r = subprocess.run(["bash", "-n", os.path.join(_SS, name)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr}"


# ---------------------------------------------------------------------------
# The auto-scoring watcher's spend guard.
#
# The watcher calls marlowe_final_endorse_all.sh / marlowe_final_classify_all.sh
# with the `all` group, which is only safe because every OTHER row in those
# groups is already scored. That is true as of 2026-09-10 but is not enforced by
# anything at runtime, so the watcher re-derives it and refuses to run if it has
# stopped holding. These tests drive --check over a SYNTHETIC tree (site paths
# are env-overridable) so the guard is exercised in both directions.
# ---------------------------------------------------------------------------

WATCHER = os.path.join(_SS, "marlowe_stakeholder_score_watcher.sh")


def _driver_rows(filename, arrays):
    """Rows of the named JOBS arrays in a driver, as whitespace-split fields."""
    s = read(os.path.join(_SS, filename))
    rows = []
    for arr in arrays:
        block = re.search(rf"^{arr}=\((.*?)^\)", s, re.M | re.S)
        assert block, f"{filename}: no {arr} array"
        for line in block.group(1).splitlines():
            line = line.strip()
            if line.startswith('"'):
                rows.append(line.strip('"').split())
    return rows


def _fake_site(tmp_path, scored_everything):
    """Build a synthetic RUNS_ROOT + SCORED_DIR in which every row of both `all`
    groups has landed its gen, and is scored iff `scored_everything` (the
    stakeholder rows are left unscored either way -- they are the arm under
    test). Returns the env overrides for the watcher."""
    runs = tmp_path / "runs"
    scored = tmp_path / "scored"
    scored.mkdir()
    for env, run, tag, name, _dov in _driver_rows(
            "marlowe_final_endorse_all.sh", ["OEQ_JOBS", "PAS_JOBS"]):
        d = runs / run / "eval"
        d.mkdir(parents=True, exist_ok=True)
        for s in range(2):
            (d / f"responses_{tag}_shard{s}of2.csv").write_text("")
        if scored_everything and not name.endswith("_stakeholder_finaldata"):
            (scored / f"scored_{name}.csv").write_text("")
    for env, run, tag, n in _driver_rows(
            "marlowe_final_classify_all.sh", ["AITA_JOBS", "FLIP_JOBS"]):
        n = int(n)
        d = runs / run / "eval"
        d.mkdir(parents=True, exist_ok=True)
        for s in range(n):
            (d / f"aita_responses_{tag}_shard{s}of{n}.csv").write_text("")
            if scored_everything and tag != TAG:
                (d / f"aita_scored_{tag}_shard{s}of{n}.csv").write_text("")
    return {"RUNS_ROOT": str(runs), "SCORED_DIR": str(scored)}


def _check(overrides):
    env = dict(os.environ, **overrides)
    env.pop("OPENROUTER_API_KEY", None)   # --check must not require the key
    r = subprocess.run(["bash", WATCHER, "--check"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _guard(out, kind):
    """(count, rows) for one guard section of --check output."""
    lines = out.splitlines()
    i = [k for k, l in enumerate(lines)
         if l.startswith(f"{kind} rows outside")]
    assert len(i) == 1, out
    count = int(lines[i[0]].rsplit(":", 1)[1])
    rows = []
    for l in lines[i[0] + 1:]:      # stop at the next section header
        if not l.startswith("    "):
            break
        rows.append(l.strip())
    assert len(rows) == count, f"{kind}: count {count} != {len(rows)} rows listed"
    return count, rows


def test_guard_quiet_when_only_this_arm_is_unscored(tmp_path):
    """The situation the watcher is designed for: `all` would spend on this arm only."""
    out = _check(_fake_site(tmp_path, scored_everything=True))
    assert _guard(out, "endorse") == (0, []), out
    assert _guard(out, "verdict") == (0, []), out


def test_guard_names_foreign_rows_that_would_also_be_scored(tmp_path):
    """If the drivers' scope drifts (another arm ready but unscored), the guard
    must SEE it -- otherwise the watcher silently pays for someone else's arm.
    Every foreign row must be listed, not just the first: an undercount here
    would read as `0` on the live tree and wave the spend through."""
    out = _check(_fake_site(tmp_path, scored_everything=False))
    n_e, rows_e = _guard(out, "endorse")
    n_v, rows_v = _guard(out, "verdict")
    # every non-stakeholder row of each driver's `all` group, and no other
    expect_e = {r[3] for r in _driver_rows("marlowe_final_endorse_all.sh",
                                           ["OEQ_JOBS", "PAS_JOBS"])
                if not r[3].endswith("_stakeholder_finaldata")}
    expect_v = {f"{r[0]}:{r[2]}" for r in _driver_rows(
        "marlowe_final_classify_all.sh", ["AITA_JOBS", "FLIP_JOBS"])
        if r[2] != TAG}
    assert set(rows_e) == expect_e, sorted(set(rows_e) ^ expect_e)
    assert set(rows_v) == expect_v, sorted(set(rows_v) ^ expect_v)
    assert n_e > 1 and n_v > 1, (n_e, n_v)
    assert not [r for r in rows_e + rows_v if "stakeholder" in r], rows_e + rows_v


def test_check_reports_gen_ready_for_the_arm(tmp_path):
    """--check's state table must flip to `ready` once the gen shards exist, at
    the per-env shard regime (AITA=1, everything else 2)."""
    out = _check(_fake_site(tmp_path, scored_everything=True))
    for env in ENVS:
        line = [l for l in out.splitlines() if l.split()[:1] == [env]]
        assert len(line) == 1, f"{env}: {out}"
        assert line[0].split()[1] == "ready", line[0]
