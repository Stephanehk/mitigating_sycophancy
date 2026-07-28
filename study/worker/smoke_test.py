"""Self-contained GPU smoke test for the study's generation path.

Runs the WHOLE live generation pipeline: seeds one job per arm straight into
Supabase, runs the real worker against a real vLLM engine, then prints each
arm's response and the per-stage breakdown.

No study server is started, because there is no longer one to start — the queue
is Postgres and the worker claims from it directly. Seeding uses the SERVICE
key (this is an operator tool, not something deployed); the worker itself uses
only the public anon key plus the worker token, exactly as in production.

This exists because the pieces it covers are exactly the ones unit tests
cannot: that the LoRA registers and is addressable by name, that the base and
adapter are genuinely different models at inference time, that the neutral
arm's two-stage chain works against a real engine, and that a response comes
back within the participant-facing timeouts.

Every row it creates is deleted on exit, including on failure.

Usage (see smoke_test.sbatch for the SLURM wrapper):
    python study/worker/smoke_test.py --device cuda:0
"""

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY_DIR)
sys.path.insert(0, _HERE)

USER_TEXT = (
    "I didn't invite my sibling to my birthday dinner because I wanted to keep it "
    "to six people in my small apartment. They found out from a mutual friend and "
    "were really hurt. I genuinely didn't think they'd want to come since they've "
    "skipped the last two family things, but now they say I excluded them on purpose."
)


def _rest(path, method="GET", **kwargs):
    """One PostgREST call with the SERVICE key. Operator-only, never deployed."""
    import requests
    import config
    assert config.SUPABASE_SERVICE_KEY, "STUDY_SUPABASE_SERVICE_KEY must be set"
    url = config.SUPABASE_URL.rstrip("/") + "/rest/v1" + path
    headers = {"apikey": config.SUPABASE_SERVICE_KEY,
               "Authorization": "Bearer " + config.SUPABASE_SERVICE_KEY,
               "Content-Type": "application/json"}
    response = requests.request(method, url, headers=headers, timeout=30, **kwargs)
    assert response.status_code in (200, 201, 204), (
        "%s %s -> %d %s" % (method, path, response.status_code, response.text[:300]))
    return response.json() if response.text.strip() else None


def seed_jobs(tag):
    """One pending job per arm, inserted straight into the queue.

    Bypasses start_response deliberately: that RPC draws the arm at random, and
    this test needs one job of EACH arm to compare them. `tag` scopes the rows
    so cleanup cannot touch anything else.
    """
    import datetime
    import scenarios
    from prompts import ARMS, build_user_turn

    scenario = scenarios.get_scenario("exclusion")
    # Same composition start_response performs: the participant's own screening
    # description of their incident, then their side of it. NOT the vignette.
    own_description = ("I did not invite my brother to my birthday dinner because"
                       " I could only book a table for six.")
    user_text = build_user_turn(own_description, USER_TEXT)
    deadline = (datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(seconds=1800)).isoformat()

    jobs = {}
    for arm in ARMS:
        session_id = "%s-%s" % (tag, arm)
        _rest("/study_sessions", "POST", json={
            "session_id": session_id, "session_token": session_id + "-tok",
            "screening_order": ["exclusion"], "arm": arm, "user_agent": "smoke"})
        job_id = "%s-job-%s" % (tag, arm)
        _rest("/study_jobs", "POST", json={
            "job_id": job_id, "session_id": session_id, "arm": arm,
            "topic_title": scenario["topic_title"], "user_text": user_text,
            "status": "pending", "deadline_at": deadline})
        jobs[arm] = job_id
    return jobs


def cleanup(tag):
    """Delete every row this run created. Jobs cascade off the session."""
    for row in _rest("/study_sessions?session_id=like.%s*&select=session_id" % tag):
        _rest("/study_sessions?session_id=eq.%s" % row["session_id"], "DELETE")


def get_job(job_id):
    """One job row from Supabase. `stages` is already parsed jsonb."""
    rows = _rest("/study_jobs?job_id=eq.%s&select=*" % job_id)
    return rows[0] if rows else None


def report(jobs):
    """Print every arm's result and return True if all four succeeded."""
    ok = True
    print("\n" + "=" * 78)
    print("SMOKE TEST RESULTS")
    print("=" * 78)
    for arm, job_id in jobs.items():
        job = get_job(job_id)
        stages = job["stages"] or []
        print("\n--- arm=%s  status=%s  attempts=%s  stages=%d"
              % (arm, job["status"], job["attempts"], len(stages)))
        if job["status"] != "done":
            ok = False
            print("    ERROR: %s" % (job["error"] or "(none)"))
            continue
        for stage in stages:
            print("    [%s] model=%s system=%s %.1fs -> %d chars"
                  % (stage["name"], stage["model"], stage["had_system_prompt"],
                     stage["seconds"], len(stage["output"])))
        print("    RESPONSE: %s" % job["response_text"].replace("\n", "\n              "))

    print("\n" + "=" * 78)
    # The neutral arm is the only two-stage one; if its edit pass did not
    # actually change the text, the editing prompt is not doing its job.
    neutral = get_job(jobs["neutral"]) if "neutral" in jobs else None
    if neutral and neutral["stages"]:
        stages = neutral["stages"]
        if len(stages) == 2:
            changed = stages[0]["output"] != stages[1]["output"]
            print("neutral edit pass changed the text: %s" % changed)
        else:
            ok = False
            print("neutral arm produced %d stages, expected 2" % len(stages))

    # ours vs untrained must not be byte-identical, or the LoRA is not applied.
    ours, untrained = get_job(jobs["ours"]), get_job(jobs["untrained"])
    if ours["status"] == "done" and untrained["status"] == "done":
        distinct = ours["response_text"] != untrained["response_text"]
        print("ours != untrained (LoRA is being applied): %s" % distinct)
        if not distinct:
            ok = False

    print("OVERALL: %s" % ("PASS" if ok else "FAIL"))
    print("=" * 78)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--vllm-log", default=None)
    args = parser.parse_args()

    import config
    tag = "smoke-%d" % int(time.time())
    token = os.environ.get("STUDY_WORKER_TOKEN", "").strip()
    assert token, ("STUDY_WORKER_TOKEN must be set and match "
                   "study_config.worker_token in Supabase")
    assert config.SUPABASE_ANON_KEY, "STUDY_SUPABASE_ANON_KEY must be set (PUBLIC key)"

    print("tag=%s queue=%s" % (tag, config.SUPABASE_URL))
    try:
        jobs = seed_jobs(tag)
        print("seeded %d jobs: %s" % (len(jobs), ", ".join(jobs.values())))

        cmd = [sys.executable, os.path.join(_HERE, "run_worker.py"),
               "--device", args.device,
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-model-len", str(args.max_model_len),
               "--max-jobs", str(len(jobs)), "--backend", args.backend]
        if args.backend == "hf":
            cmd += ["--device-map", args.device_map]
        if args.tensor_parallel_size:
            cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
        if args.enforce_eager:
            cmd += ["--enforce-eager"]
        if args.vllm_log:
            cmd += ["--vllm-log", args.vllm_log]

        print("running worker: %s" % " ".join(cmd))
        worker = subprocess.run(cmd, env=dict(os.environ))
        print("worker exited with %d" % worker.returncode)

        ok = report(jobs)
        return 0 if ok else 1
    finally:
        # Runs even on failure: a smoke test must never leave rows behind in the
        # same tables the real study writes to.
        cleanup(tag)
        print("cleaned up rows tagged %s" % tag)


if __name__ == "__main__":
    sys.exit(main())
