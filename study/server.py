"""HTTP server for the sycophancy user study.

Serves the participant SPA from static/ and two JSON APIs:

  /api/*          — the participant flow. Authenticated by an opaque
                    `session_token` the browser holds, issued at session
                    creation.

  /api/worker/*   — the generation queue, consumed by the GPU worker on
                    next6/next7. Authenticated by the shared secret in
                    `X-Worker-Token`. The worker connects OUTBOUND to this
                    server; nothing ever connects into the cluster.

THE PARTICIPANT API IS DELIBERATELY TINY. A participant now touches the server
exactly three times: once to start (get a token and the content bundle), once to
request their generation (plus status polls while it runs), and once to submit
everything at the end. Every page turn in between is local to the browser.

That is a reversal of the previous design, in which the server owned the stage
machine and every Continue press was a round trip plus two synchronous writes.
Measured on this cluster, that cost ~213ms per press on the NFS-backed database
and stalled 10-14 seconds when the worker's long poll contended for the write
lock. Moving navigation into the client removes the round trip entirely rather
than making it faster.

WHAT THE SERVER STILL REFUSES TO DELEGATE:
  * the arm draw — one per session, at job creation, never redrawn, never sent
    to the browser. A client-side draw could be re-rolled from devtools.
  * one job per session — /api/response/start is idempotent, so the response
    cannot be regenerated until the participant likes it.
Both are what keep the DVs interpretable.

Error-handling note: the repo's default is to let code fail fast, and that is
what these endpoints do — a programming error raises and returns 500. The
deliberate exceptions are the three places graceful degradation is required:
`ValidationError` becomes an inline 400, worker-reported generation failures are
recorded as job attempts rather than crashing the server, and a Supabase outage
at submit time is recorded rather than shown to a participant who has already
finished (see supabase_client).
"""

import contextlib
import json
import os
import sys
import time

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import flow
import prompts
import store
import supabase_client
import views
from prompts import ARMS

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_HERE, "static")


@contextlib.asynccontextmanager
async def lifespan(_app):
    """Create the schema and refuse to start a misconfigured production run.

    The production check runs before the first request is served, so a launch
    with the development worker token, the source survey's Prolific codes, an
    unconfigured Supabase, or a database on NFS fails at boot rather than
    silently collecting unusable data.
    """
    if config.IS_PRODUCTION:
        config.check_production_config()
    store.init_db()
    if not supabase_client.is_configured():
        print("WARNING: Supabase is not configured; submissions will only be "
              "written to %s" % (config.SUBMISSION_FALLBACK_PATH,),
              file=sys.stderr, flush=True)
    yield


app = FastAPI(title="Sycophancy user study", docs_url=None, redoc_url=None,
              lifespan=lifespan)


@app.exception_handler(flow.ValidationError)
async def _validation_handler(request, exc):
    """Turn a participant-facing validation failure into an inline 400."""
    return JSONResponse(status_code=400, content={"error": str(exc)})


# --- Session helpers -------------------------------------------------------

def _require_session(token):
    """Look up a session by token, or 401.

    Returns the session dict. A missing or unknown token means the browser lost
    its session, which the client handles by starting a new one.
    """
    if not token:
        raise HTTPException(status_code=401, detail="missing session token")
    session = store.get_session_by_token(token)
    if session is None:
        raise HTTPException(status_code=401, detail="unknown session")
    return session


# --- Participant API -------------------------------------------------------

@app.post("/api/session")
def create_session(request: Request, payload: dict = Body(default={})):
    """Start a session: issue a token, draw the screening order, send the copy.

    Returns everything the client needs to run the entire study offline from
    here: the session token, this participant's randomized scenario order, and
    the full content bundle.

    Resuming: a client that presents a known token gets the same token and a
    fresh bundle back, which is what makes a refresh non-destructive. The
    ANSWERS are not resumed from here — the browser keeps those in localStorage,
    because the server never sees them until the end.
    """
    token = (payload or {}).get("session_token")
    if token:
        existing = store.get_session_by_token(token)
        if existing is not None:
            return {
                "session_token": token,
                "session_id": existing["session_id"],
                "screening_order": json.loads(existing["screening_order"]),
                "has_job": store.latest_job_for_session(existing["session_id"]) is not None,
                "submitted": existing["submitted_at"] is not None,
                "content": views.content_bundle(),
            }

    session_id, new_token = store.create_session(
        flow.new_screening_order(), request.headers.get("user-agent", "")[:500])
    return {
        "session_token": new_token,
        "session_id": session_id,
        "screening_order": json.loads(store.get_session(session_id)["screening_order"]),
        "has_job": False,
        "submitted": False,
        "content": views.content_bundle(),
    }


@app.post("/api/response/start")
def start_response(payload: dict = Body(...), x_session_token: str = Header(default=None)):
    """Enqueue the generation job from the participant's two free-text answers.

    Body: {scenario_key, description, perspective}. These are the ONLY answers
    the server sees before the end of the study, because they are the only ones
    it needs — `prompts.build_user_turn` is composed from exactly these two, and
    NO text from scenarios.py reaches the model.

    IDEMPOTENT, AND THAT IS A DATA-INTEGRITY RULE, NOT A CONVENIENCE. A session
    gets one job, forever. A participant who double-clicks, refreshes the
    loading screen, or edits their story and re-posts is reattached to the
    existing job and the existing arm. Without this they could regenerate until
    they liked the answer, which would corrupt every DV.
    """
    session = _require_session(x_session_token)
    existing = store.latest_job_for_session(session["session_id"])
    if existing is not None:
        return {"response": views.response_state(session["session_id"])}

    scenario, description, perspective = flow.validate_model_input(
        (payload or {}).get("scenario_key"),
        (payload or {}).get("description"),
        (payload or {}).get("perspective"))

    # Drawn here — the moment of eligibility — so screened-out participants
    # never consume an arm slot. Sticky thereafter: the branch above returns
    # before this line on every subsequent call.
    arm = session["arm"] or flow.assign_arm()
    assert arm in ARMS, "invalid arm: %r" % (arm,)
    if session["arm"] != arm:
        store.update_session(session["session_id"], arm=arm)

    user_text = prompts.build_user_turn(description, perspective)
    store.create_job(session["session_id"], arm, scenario["topic_title"], user_text)
    return {"response": views.response_state(session["session_id"])}


@app.get("/api/response/status")
def response_status(x_session_token: str = Header(default=None)):
    """Poll the generation job. Drives the loading screen."""
    session = _require_session(x_session_token)
    return {"response": views.response_state(session["session_id"])}


@app.post("/api/response/retry")
def retry_response(x_session_token: str = Header(default=None)):
    """Requeue a failed generation, backing the participant's 'Try again'."""
    session = _require_session(x_session_token)
    job = store.latest_job_for_session(session["session_id"])
    assert job is not None, "retry with no job"
    if job["status"] != "failed":
        raise HTTPException(status_code=409, detail={"error": "job is not in a failed state"})
    if views.retry_exhausted(job):
        raise HTTPException(status_code=409, detail={"error": "retries exhausted"})
    store.requeue_job(job["job_id"])
    return {"response": views.response_state(session["session_id"])}


@app.post("/api/submit")
def submit(payload: dict = Body(...), x_session_token: str = Header(default=None)):
    """Accept the whole study in one payload and ship it to Supabase.

    This is the only place participant answers reach the server. The client's
    payload is validated, then MERGED WITH SERVER-SIDE TRUTH — arm, job id, the
    exact `user_text` the model received, the response text, per-stage outputs,
    attempt counts — none of which is taken from the client, because the client
    was never told the arm and must not be able to claim one.

    Idempotent by `submitted_at`: a duplicate submit returns ok without writing
    Supabase a second time, so a client retry after a slow response cannot
    double-count a participant.

    A Supabase outage does NOT fail this request. The row is already durable in
    the local JSONL by then, and the participant has finished the study; the
    error is reported in the response body for the logs and replayed later by
    analysis/backfill_supabase.py.
    """
    session = _require_session(x_session_token)
    screening_order = json.loads(session["screening_order"])
    raw = (payload or {}).get("payload") or {}
    clean = flow.validate_submission(raw, screening_order)

    if session["submitted_at"] is not None:
        return {"ok": True, "duplicate": True}

    job = store.latest_job_for_session(session["session_id"])
    row = {
        "session_id": session["session_id"],
        # Server-owned. Never round-tripped through the browser.
        "arm": session["arm"],
        "job_id": None if job is None else job["job_id"],
        "topic_title": None if job is None else job["topic_title"],
        "user_text": None if job is None else job["user_text"],
        "shown_response": None if job is None else job["response_text"],
        "shown_response_chars": None if job is None or job["response_text"] is None
                                else len(job["response_text"]),
        "stages": None if job is None or not job["stages_json"]
                  else json.loads(job["stages_json"]),
        "worker_attempts": None if job is None else job["attempts"],
        "manual_retries": None if job is None else job["manual_retries"],
        "screening_order": screening_order,
        "session_created_at": session["created_at"],
        # Stamped here rather than left to Supabase's `default now()`, so the
        # fallback JSONL carries it too and `--source local` exports the same
        # durations as a Supabase-sourced export.
        "submitted_at": store.utcnow(),
        # Participant-reported, validated above.
        "prolific_pid": clean["prolific_pid"],
        "consent": clean["consent"],
        "eligibility": clean["eligibility"],
        "selected_scenario": clean["selected_scenario"],
        "screening": clean["screening"],
        "perspective_text": clean["perspective_text"],
        "additional_thoughts": clean["additional_thoughts"],
        "answers": clean["answers"],
        "missing_keys": clean["missing_keys"],
        "client_timings": clean["client_timings"],
        # The client's payload EXACTLY as received, before validation. A
        # backstop against a validator bug quietly dropping something: the
        # promoted columns above are the normalized view, this is the original.
        "payload": raw,
    }

    # An eligible session must have produced a job; anything else means the
    # client claimed eligibility it never earned.
    if clean["eligibility"] == "eligible" and job is None:
        raise flow.ValidationError("Eligible submission with no generation job.")

    if not store.mark_submitted(session["session_id"]):
        return {"ok": True, "duplicate": True}

    result = supabase_client.submit(row)
    return {"ok": True, "duplicate": False, "supabase_error": result["supabase_error"]}


# --- Worker API ------------------------------------------------------------

def _require_worker(token):
    """Authenticate the GPU worker, or 401."""
    if not token or token != config.WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="bad worker token")


@app.post("/api/worker/claim")
def worker_claim(payload: dict = Body(...), x_worker_token: str = Header(default=None)):
    """Hand out the next pending job, or {"job": None} immediately.

    RETURNS AT ONCE — it does not wait for work. This was a 20-second long poll
    that held the request open until a job appeared, which is the right design
    on a always-on server and the wrong one behind Lambda, where you are billed
    for the whole time the request is held. With four workers that idle wait
    costs ~5.2M GB-s/month against a 400k free tier; returning immediately and
    letting the worker sleep WORKER_POLL_SECONDS costs ~260k. The worker owns
    the waiting now (see run_worker.py).

    Concurrency is safe by construction: store.claim_job is atomic (SQLite
    BEGIN IMMEDIATE, or Postgres FOR UPDATE SKIP LOCKED), so N workers polling
    simultaneously take N different jobs rather than racing for one.
    """
    _require_worker(x_worker_token)
    worker_id = ((payload or {}).get("worker_id") or "unknown")[:120]

    store.reap_expired_jobs()
    job = store.claim_job(worker_id)
    if job is None:
        return {"job": None}
    return {"job": {
        "job_id": job["job_id"],
        "arm": job["arm"],
        "topic_title": job["topic_title"],
        "user_text": job["user_text"],
        "attempts": job["attempts"],
    }}


@app.post("/api/worker/heartbeat")
def worker_heartbeat(payload: dict = Body(...), x_worker_token: str = Header(default=None)):
    """Extend the lease on a job that is still generating."""
    _require_worker(x_worker_token)
    job_id = (payload or {}).get("job_id")
    worker_id = (payload or {}).get("worker_id")
    return {"extended": store.extend_lease(job_id, worker_id)}


@app.post("/api/worker/complete")
def worker_complete(payload: dict = Body(...), x_worker_token: str = Header(default=None)):
    """Record a finished generation.

    `stages` carries every stage's raw output, so the neutral arm's pre-edit
    Stage 1 text survives for analysis. `accepted` is False when this worker
    had already lost its lease, which tells it to drop the result.
    """
    _require_worker(x_worker_token)
    job_id = (payload or {}).get("job_id")
    worker_id = (payload or {}).get("worker_id")
    text = ((payload or {}).get("response_text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="response_text must be non-empty")
    accepted = store.complete_job(job_id, worker_id, text, (payload or {}).get("stages") or [])
    return {"accepted": accepted}


@app.post("/api/worker/fail")
def worker_fail(payload: dict = Body(...), x_worker_token: str = Header(default=None)):
    """Record a failed attempt; the store decides requeue vs terminal failure."""
    _require_worker(x_worker_token)
    status = store.fail_job_attempt((payload or {}).get("job_id"),
                                    (payload or {}).get("worker_id"),
                                    (payload or {}).get("error") or "unspecified")
    return {"status": status}


# --- Ops -------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Liveness plus the numbers worth watching during a live run.

    `saw_response_never_submitted` is the completers-only design's blind spot
    made visible: participants who reached the response and then abandoned,
    whose answers exist nowhere. A rise means something is broken after the
    response page.
    """
    store.reap_expired_jobs()
    return {
        "ok": True,
        "env": config.STUDY_ENV,
        "queue": store.queue_depth(),
        "submitted_by_arm": store.arm_counts(),
        "saw_response_never_submitted": store.unsubmitted_sessions_with_done_jobs(),
        "supabase_configured": supabase_client.is_configured(),
    }


# --- Static SPA ------------------------------------------------------------

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index():
    """Serve the SPA shell."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))
