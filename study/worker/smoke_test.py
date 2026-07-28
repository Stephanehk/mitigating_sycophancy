"""Self-contained GPU smoke test for the study's generation path.

Runs the WHOLE live pipeline on one node with no external dependencies: starts
the study server on localhost against a throwaway database, seeds one job per
arm, runs the real worker against a real vLLM engine, then prints each arm's
response and the per-stage breakdown.

This exists because the pieces it covers are exactly the ones unit tests
cannot: that the LoRA registers and is addressable by name, that the base and
adapter are genuinely different models at inference time, that the neutral
arm's two-stage chain works against a real engine, and that a response comes
back within the participant-facing timeouts.

Everything is torn down on exit; the temporary database is deleted.

Usage (see smoke_test.sbatch for the SLURM wrapper):
    python study/worker/smoke_test.py --device cuda:0
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
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


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_for_server(base_url, timeout=90):
    """Block until the study server answers /api/health."""
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(base_url + "/api/health", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    raise AssertionError("study server did not come up at %s" % base_url)


def seed_jobs(db_url):
    """Insert one session + job per arm. Returns {arm: job_id}."""
    import config
    config.DATABASE_URL = db_url
    import store
    import scenarios
    from prompts import ARMS, build_user_turn

    scenario = scenarios.get_scenario("exclusion")
    # Same composition the server performs: the participant's own screening
    # description of their incident, then their side of it. NOT the vignette —
    # see prompts.build_user_turn.
    own_description = ("I did not invite my brother to my birthday dinner because"
                       " I could only book a table for six.")
    user_text = build_user_turn(own_description, USER_TEXT)

    store.init_db()
    jobs = {}
    for arm in ARMS:
        session_id, _token = store.create_session(["exclusion"], "smoke")
        store.update_session(session_id, stage="response", arm=arm, consent="agree",
                             eligibility="eligible", selected_scenario="exclusion",
                             perspective_text=USER_TEXT)
        jobs[arm] = store.create_job(session_id, arm, scenario["topic_title"], user_text)
    return jobs


def report(db_url, jobs):
    """Print every arm's result and return True if all four succeeded."""
    import config
    config.DATABASE_URL = db_url
    import store

    ok = True
    print("\n" + "=" * 78)
    print("SMOKE TEST RESULTS")
    print("=" * 78)
    for arm, job_id in jobs.items():
        job = store.get_job(job_id)
        stages = json.loads(job["stages_json"]) if job["stages_json"] else []
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
    neutral = store.get_job(jobs.get("neutral")) if "neutral" in jobs else None
    if neutral and neutral["stages_json"]:
        stages = json.loads(neutral["stages_json"])
        if len(stages) == 2:
            changed = stages[0]["output"] != stages[1]["output"]
            print("neutral edit pass changed the text: %s" % changed)
        else:
            ok = False
            print("neutral arm produced %d stages, expected 2" % len(stages))

    # ours vs untrained must not be byte-identical, or the LoRA is not applied.
    ours, untrained = store.get_job(jobs["ours"]), store.get_job(jobs["untrained"])
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

    workdir = tempfile.mkdtemp(prefix="syco_smoke_")
    db_url = "sqlite:///" + os.path.join(workdir, "smoke.db")
    port = free_port()
    base_url = "http://127.0.0.1:%d" % port
    token = "smoke-token-local"

    env = dict(os.environ)
    env.update({"STUDY_DATABASE_URL": db_url, "STUDY_WORKER_TOKEN": token,
                "STUDY_ENV": "development"})

    print("workdir=%s" % workdir)
    print("starting study server on %s" % base_url)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=_STUDY_DIR, env=env)
    try:
        wait_for_server(base_url)
        os.environ["STUDY_DATABASE_URL"] = db_url
        jobs = seed_jobs(db_url)
        print("seeded %d jobs: %s" % (len(jobs), ", ".join(jobs)))

        cmd = [sys.executable, os.path.join(_HERE, "run_worker.py"),
               "--server-url", base_url, "--device", args.device,
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
        worker = subprocess.run(cmd, env=env)
        print("worker exited with %d" % worker.returncode)

        ok = report(db_url, jobs)
        return 0 if ok else 1
    finally:
        server.terminate()
        server.wait(timeout=30)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
