"""Tunable configuration for the sycophancy user study.

Everything an operator might need to change between a local dry run and a live
Prolific launch lives here, read from the environment with dev-safe defaults.

Fail-fast policy: settings whose default would be UNSAFE in production (the
worker shared secret, the Prolific completion codes) are validated by
`check_production_config()`, which server.py calls at startup whenever
STUDY_ENV=production. A local run needs no environment at all.
"""

import os


def _env(name, default):
    """Read an environment variable, falling back to a development default."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name, default):
    return int(_env(name, str(default)))


def _env_float(name, default):
    return float(_env(name, str(default)))


# --- Environment -----------------------------------------------------------

# "development" (default) or "production". Production enables the config
# assertions below and disables the debug endpoints.
STUDY_ENV = _env("STUDY_ENV", "development")
IS_PRODUCTION = STUDY_ENV == "production"


# --- Datastore -------------------------------------------------------------
# The local SQLite database now holds ONLY the generation queue and the thin
# session rows the queue needs (see schema.sql). Participant answers never touch
# it — the browser accumulates them and posts one payload to Supabase at the end
# (see supabase_client.py).
#
# KEEP THIS OFF NFS. Measured on this cluster, SQLite on the /next NFS mount
# runs ~213ms per request against ~40ms on local disk, and lock contention with
# the worker's long poll produced sporadic 10-14 SECOND stalls. next1 has a
# local NVMe at /scr. check_production_config() refuses to boot on a network
# filesystem for exactly this reason.

DATABASE_URL = _env("STUDY_DATABASE_URL", "sqlite:///" + os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "study.db"))

# Filesystem types that make SQLite slow and lock-prone. Checked in production.
NETWORK_FILESYSTEMS = ("nfs", "nfs4", "cifs", "smb3", "fuse.sshfs", "lustre", "gpfs")


# --- Supabase --------------------------------------------------------------
# The real datastore for participant responses. The browser holds one JSON
# payload for the whole session and posts it once, at the end, to /api/submit;
# the server forwards it to Supabase with the service-role key, which therefore
# never reaches the browser.
#
# Every completed submission is ALSO appended to SUBMISSION_FALLBACK_PATH before
# the network call, so a Supabase outage degrades to "replay the file later"
# rather than "the participant's data is gone". See analysis/backfill_supabase.py.

def _normalize_supabase_url(raw):
    """Reduce a Supabase URL to the bare project origin.

    The dashboard shows the project origin as "Project URL" but ALSO shows
    "API URL" as `https://<ref>.supabase.co/rest/v1/`, and the second is the one
    sitting next to the API keys people copy. supabase_client appends
    `/rest/v1/<table>` itself, so pasting the longer form would build
    `.../rest/v1/rest/v1/study_submissions` and 404 at submit time — at the end
    of a participant's session.

    The two forms are unambiguous, so this trims rather than rejects. Anything
    else (a placeholder, a non-https value) is left alone for
    check_production_config to complain about.
    """
    trimmed = raw.rstrip("/")
    if trimmed.endswith("/rest/v1"):
        trimmed = trimmed[: -len("/rest/v1")]
    return trimmed


#: THE TABLE GENERATION (added 2026-08-07). Every table this study touches
#: carries this suffix.
#:
#: WHY. The 2026-08-07 revision is a different instrument, not a tweak: four new
#: screening scenarios with new keys, and a draw over `ours`/`untrained` (the
#: review-page arms) instead of their no-review twins. Appending those rows to
#: the tables the previous run filled would put two instruments in one table,
#: distinguishable only by reading `selected_scenario` and `arm` and knowing the
#: changeover date. Worse, `study_scenarios` is TRUNCATED and repopulated on every
#: deploy of the generated schema — so re-running it against the old tables would
#: delete the old key→topic_title mapping outright.
#:
#: A NEW SET OF TABLES IN THE SAME PROJECT, not a new project: same URL, same
#: anon key, same worker token, so nothing deployed needs new credentials. The
#: RPC names are deliberately NOT suffixed — `create or replace function`
#: repoints them at the new tables and the browser and worker call them
#: unchanged. The old tables keep their rows and stop being written to.
#:
#: NOT env-driven, on purpose. The three .sql files are static artifacts a human
#: pastes into the Supabase SQL editor, so a suffix that could differ at runtime
#: from the one baked into them would be a way to point the app at tables that do
#: not exist. tests/test_views.py pins the two together.
TABLE_SUFFIX = "_v2"

#: The five tables the app owns. Everything that names a table in Python reads
#: it from here; the .sql files spell the same names out and the test above keeps
#: them in step.
TABLE_SUBMISSIONS = "study_submissions" + TABLE_SUFFIX
TABLE_SESSIONS = "study_sessions" + TABLE_SUFFIX
TABLE_JOBS = "study_jobs" + TABLE_SUFFIX
TABLE_SCENARIOS = "study_scenarios" + TABLE_SUFFIX
TABLE_QUESTIONNAIRE_OPTIONS = "questionnaire_options" + TABLE_SUFFIX
TABLE_CONFIG = "study_config" + TABLE_SUFFIX

ALL_TABLES = (TABLE_SUBMISSIONS, TABLE_SESSIONS, TABLE_JOBS, TABLE_SCENARIOS,
              TABLE_QUESTIONNAIRE_OPTIONS, TABLE_CONFIG)

#: The unsuffixed names, i.e. the tables the pre-2026-08-07 study wrote to. Kept
#: as data so a half-finished rename is a test failure rather than a table that
#: silently keeps collecting.
LEGACY_TABLES = tuple(name[: -len(TABLE_SUFFIX)] for name in ALL_TABLES)

SUPABASE_URL = _normalize_supabase_url(_env("STUDY_SUPABASE_URL", ""))

# PUBLIC. Ships in the page and runs on the GPU worker. Powerless on its own:
# RLS blocks every table and each RPC authenticates its own caller. This is the
# only Supabase credential that gets deployed anywhere.
SUPABASE_ANON_KEY = _env("STUDY_SUPABASE_ANON_KEY", "")

# PRIVILEGED — bypasses RLS. Used only from a developer machine, for exports and
# ops. Must never be built into the page or exported to the cluster.
SUPABASE_SERVICE_KEY = _env("STUDY_SUPABASE_SERVICE_KEY", "")
#: Which submissions table the EXPORT reads. Defaults to the current generation;
#: point it at "study_submissions" to export the pre-2026-08-07 run.
SUPABASE_TABLE = _env("STUDY_SUPABASE_TABLE", TABLE_SUBMISSIONS)
SUPABASE_TIMEOUT_SECONDS = _env_int("STUDY_SUPABASE_TIMEOUT_SECONDS", 20)

SUBMISSION_FALLBACK_PATH = _env("STUDY_SUBMISSION_FALLBACK_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "submissions.jsonl"))


# --- Worker authentication -------------------------------------------------
# The GPU worker on next6/next7 authenticates to the job endpoints with this
# shared secret in an `X-Worker-Token` header. The dev default is deliberately
# obvious and is rejected in production.

WORKER_TOKEN = _env("STUDY_WORKER_TOKEN", "dev-insecure-worker-token")
INSECURE_WORKER_TOKEN = "dev-insecure-worker-token"


# --- Prolific --------------------------------------------------------------
# TWO CODES, ONE PER PATH (changed 2026-08-01):
#   cc=C1Q8MWKX  completed the study — they saw a response and answered the battery
#   cc=C1IXI2WK  screened out — they never got as far as sending anything to the model
#
# This replaces the single shared code set on 2026-07-31. Prolific can now tell
# the two groups apart and pay them differently, and its dashboard shows a real
# screen-out rate. Your own data could always tell them apart — `eligibility` is
# 'eligible' vs 'screened_out' on every submission row — so the export's funnel
# is a cross-check on Prolific's, not the only copy of it.
#
# A DECLINER STILL GETS NO CODE (see views.content_bundle): they did not consent,
# so there is nothing to pay for and they return the submission instead.
#
# An environment variable still wins over either default, so a stale
# STUDY_PROLIFIC_COMPLETION_URL/_CODE or _SCREENOUT_URL/_CODE in the Amplify
# console would silently override the values below.

PROLIFIC_COMPLETION_URL = _env(
    "STUDY_PROLIFIC_COMPLETION_URL",
    "https://app.prolific.com/submissions/complete?cc=C1Q8MWKX")
PROLIFIC_SCREENOUT_URL = _env(
    "STUDY_PROLIFIC_SCREENOUT_URL",
    "https://app.prolific.com/submissions/complete?cc=C1IXI2WK")

# Codes inherited from the source survey; used only to warn at startup.
_SOURCE_SURVEY_CODES = ("C1CRO5LN", "C1L7XBXC")

# THE COMPLETION CODES THEMSELVES, shown on the last page.
#
# Participants are no longer redirected to Prolific — the final screen displays
# the code and they paste it into Prolific themselves. An auto-redirect meant a
# slow or blocked navigation could strand someone who had finished, with no way
# to prove it; a code on screen cannot fail that way.
#
# Derived from the completion URLs by default so a URL-only configuration still
# works, but set them explicitly in production.
def _code_from_url(url, fallback=""):
    """The `cc` parameter of a Prolific completion URL, or `fallback`."""
    import urllib.parse
    query = urllib.parse.urlparse(url).query
    return urllib.parse.parse_qs(query).get("cc", [fallback])[0]


PROLIFIC_COMPLETION_CODE = _env("STUDY_PROLIFIC_COMPLETION_CODE",
                                _code_from_url(PROLIFIC_COMPLETION_URL))
PROLIFIC_SCREENOUT_CODE = _env("STUDY_PROLIFIC_SCREENOUT_CODE",
                               _code_from_url(PROLIFIC_SCREENOUT_URL))

# Kept only so the codes above can be derived from them; nothing redirects now.
REDIRECT_DELAY_SECONDS = _env_int("STUDY_REDIRECT_DELAY_SECONDS", 5)


# --- Generation ------------------------------------------------------------

BASE_MODEL_ID = _env("STUDY_BASE_MODEL_ID", "Qwen/Qwen3-8B")

# Our OEQ-trained adapter. Served as a named LoRA on the same vLLM engine.
LORA_NAME = _env("STUDY_LORA_NAME", "oeq_voice2")
LORA_PATH = _env("STUDY_LORA_PATH", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training_mitigations", "runs", "dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5"))

# Sampling. temperature 1.0 matches eval/eval_endorsement_heldout.py's default,
# which is the condition the adapter's reported numbers were produced under.
# max_tokens is generous so nothing truncates mid-sentence; the syco/neutral
# prompts self-limit to ~100 words, the bare arms are unconstrained.
GEN_TEMPERATURE = _env_float("STUDY_GEN_TEMPERATURE", 1.0)
GEN_TOP_P = _env_float("STUDY_GEN_TOP_P", 1.0)
GEN_MAX_TOKENS = _env_int("STUDY_GEN_MAX_TOKENS", 512)

# The cap for a stage that REWRITES a previous stage's output rather than
# generating fresh (added 2026-08-07 for `ours_sandwich`). Such a stage has to
# re-emit the whole of Stage 1 plus its additions, so GEN_MAX_TOKENS — which
# sized Stage 1 in the first place — cannot fit it by construction.
#
# 2048 against a 512-token Stage 1 is deliberate headroom, not a guess at the
# needed size: truncating here would cut exactly the long responses the sandwich
# prompt works hardest to preserve, and a participant would see a reply that
# stops mid-sentence. Cheap insurance — vLLM bills the tokens it emits, not the
# cap.
GEN_REWRITE_MAX_TOKENS = _env_int("STUDY_GEN_REWRITE_MAX_TOKENS", 2048)

# Wall-clock cap on a SINGLE generation (HF backend). Without it, one
# pathologically slow generation blocks the worker forever: the worker is
# single-threaded, so it never returns to the queue and every other
# participant's job sits pending behind it. Observed on a contended next6:
# a generation that normally takes ~40s ran past 25 minutes.
# Hitting this cap is treated as a failed attempt, not as a short response,
# so a participant is never shown a sentence that stops mid-word.
GEN_MAX_SECONDS = _env_int("STUDY_GEN_MAX_SECONDS", 240)

# Qwen3's chat template honours this; it keeps the model out of thinking mode,
# matching build_chat_input() in training_mitigations/core/llama_local.py.
ENABLE_THINKING = False


# --- Job queue -------------------------------------------------------------

# A worker claims a job by taking a lease. If the worker dies mid-job the lease
# expires and the job returns to the queue, so a crashed worker cannot strand a
# participant. Must comfortably exceed the slowest two-stage generation.
JOB_LEASE_SECONDS = _env_int("STUDY_JOB_LEASE_SECONDS", 180)

# How many times a job may be attempted before it is marked permanently failed
# and the participant is shown FAILURE_EXHAUSTED.
JOB_MAX_ATTEMPTS = _env_int("STUDY_JOB_MAX_ATTEMPTS", 4)

# Wall-clock budget from job creation. Past this, the job fails even if
# attempts remain, so a participant is never left on the loading screen.
#
# RAISED FROM 420 TO 660 ON 2026-08-07, when `ours_sandwich` put a two-stage arm
# back into the draw. GEN_MAX_SECONDS caps a SINGLE generation at 240, so a
# two-stage job's worst case is 480 — already past the old deadline before any
# retry. A job that cannot finish inside its own deadline even once would fail
# every time it hit the slow path, and it would look like a GPU problem rather
# than an arithmetic one. 660 leaves room for the worst case plus a claim delay.
# Typical is nowhere near it: measured stages run 4-20s each.
JOB_DEADLINE_SECONDS = _env_int("STUDY_JOB_DEADLINE_SECONDS", 660)

# How long the worker waits before asking for a job again when the queue is
# empty. THIS USED TO BE A 20-SECOND LONG POLL, and the change is a deployment
# constraint, not a preference: behind Lambda you are billed for the whole time
# a request is held open. With four workers, a 20s long poll costs roughly
# 5.2M GB-s/month (~$85) against the 400k GB-s free tier, purely to wait. A 2s
# short poll is ~260k GB-s/month — free — because each request returns in
# milliseconds.
#
# The cost is up to WORKER_POLL_SECONDS of extra pickup latency, which is
# invisible next to a 4-18s generation. Keeping the queue behind the worker
# token also means the Supabase service-role key stays on the server and never
# reaches the GPU cluster.
WORKER_POLL_SECONDS = _env_float("STUDY_WORKER_POLL_SECONDS", 2.0)
CLIENT_POLL_INTERVAL_MS = _env_int("STUDY_CLIENT_POLL_INTERVAL_MS", 1200)

# After this long on the loading screen the client shows LOADING_SLOW_NOTICE.
CLIENT_SLOW_NOTICE_MS = _env_int("STUDY_CLIENT_SLOW_NOTICE_MS", 12000)


# --- Validation ------------------------------------------------------------

# Minimum characters for the two free-text answers that matter substantively.
MIN_DESCRIBE_CHARS = _env_int("STUDY_MIN_DESCRIBE_CHARS", 20)
MIN_PERSPECTIVE_CHARS = _env_int("STUDY_MIN_PERSPECTIVE_CHARS", 50)

# Hard cap on any single free-text field, mirroring the source survey's
# maxlength="1024" on the chat input.
MAX_TEXT_CHARS = _env_int("STUDY_MAX_TEXT_CHARS", 4000)

# Cap on the COMPOSED model input, which is two of those fields joined by a
# blank line — hence twice the field cap plus the separator, not the field cap
# itself.
#
# This is not cosmetic. On the two no-review arms the participant never sees a
# page holding the composed text, so a rejection at start_response is a dead
# end: there is nothing they can shorten and no way forward. Anything below the
# worst case two maxed-out fields can produce would leave exactly that gap.
MAX_MODEL_INPUT_CHARS = 2 * MAX_TEXT_CHARS + len("\n\n")


def filesystem_type(path):
    """The filesystem type backing `path`, from /proc/mounts, or None.

    Walks up to the nearest existing ancestor (the database file itself may not
    exist yet) and picks the longest mount point that prefixes it, which is how
    overlapping mounts are disambiguated.

    Returns None on a platform without /proc/mounts, which makes the caller's
    check advisory there rather than wrong.
    """
    if not os.path.exists("/proc/mounts"):
        return None
    target = os.path.abspath(path)
    while not os.path.exists(target) and target != "/":
        target = os.path.dirname(target)
    best = None
    with open("/proc/mounts") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            mount_point, fs_type = fields[1], fields[2]
            if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
                if best is None or len(mount_point) > len(best[0]):
                    best = (mount_point, fs_type)
    return None if best is None else best[1]


def sqlite_path(database_url):
    """The filesystem path inside a sqlite:/// URL, or None for other schemes."""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return None
    return database_url[len(prefix):]


def check_production_config():
    """Fail loudly if a production launch is missing required configuration.

    Called by server.py at startup when STUDY_ENV=production. Each check covers
    a setting whose development default would silently corrupt a live run: an
    unauthenticated job queue, Prolific codes that pay the wrong study, an
    unconfigured Supabase (which would send every completed session to the
    fallback file and nowhere else), or a database on a network filesystem
    (measured 10-14s lock stalls on this cluster's NFS).

    Raises AssertionError listing every problem at once, rather than one per
    restart.
    """
    problems = []
    if WORKER_TOKEN == INSECURE_WORKER_TOKEN:
        problems.append(
            "STUDY_WORKER_TOKEN is still the insecure development default; "
            "set it to a random secret shared with the GPU worker.")
    for label, url in (("STUDY_PROLIFIC_COMPLETION_URL", PROLIFIC_COMPLETION_URL),
                       ("STUDY_PROLIFIC_SCREENOUT_URL", PROLIFIC_SCREENOUT_URL)):
        for code in _SOURCE_SURVEY_CODES:
            if code in url:
                problems.append(
                    "%s still points at the SOURCE survey's Prolific completion "
                    "code %s; set it to this study's own code." % (label, code))
    if not DATABASE_URL.startswith(("sqlite:///", "postgresql://", "postgres://")):
        problems.append("STUDY_DATABASE_URL has an unsupported scheme: %r" % (DATABASE_URL,))

    path = sqlite_path(DATABASE_URL)
    if path is not None:
        fs_type = filesystem_type(path)
        if fs_type in NETWORK_FILESYSTEMS:
            problems.append(
                "STUDY_DATABASE_URL points at %r, which is on a %s mount. SQLite on a "
                "network filesystem stalls for seconds under worker-poll contention; "
                "put the database on local disk (next1 has /scr)." % (path, fs_type))

    if not SUPABASE_URL:
        problems.append("STUDY_SUPABASE_URL is unset; completed sessions would only "
                        "reach the local fallback file.")
    if not SUPABASE_SERVICE_KEY:
        problems.append("STUDY_SUPABASE_SERVICE_KEY is unset.")

    # An unedited copy of .supabase_env.example passes the "is it set?" checks
    # above, reports supabase_configured=true, and then 401s on every submit —
    # at the end of a participant's session, which is the worst place to find
    # out. The example uses angle brackets, which are not legal in either value.
    for label, value in (("STUDY_SUPABASE_URL", SUPABASE_URL),
                         ("STUDY_SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY)):
        if "<" in value or ">" in value:
            problems.append(
                "%s still holds a placeholder (%r); paste the real value from "
                "the Supabase dashboard." % (label, value[:40]))
    if SUPABASE_URL and not SUPABASE_URL.startswith("https://"):
        problems.append("STUDY_SUPABASE_URL should be the https:// Project URL, got %r"
                        % (SUPABASE_URL[:60],))

    assert not problems, (
        "production configuration is incomplete:\n  - " + "\n  - ".join(problems))
