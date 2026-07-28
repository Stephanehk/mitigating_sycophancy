"""Start one vLLM engine and N worker processes against it.

WHY N PROCESSES AND NOT ONE. run_worker.py is strictly serial: claim a job,
generate, repeat. Pointing it at vLLM instead of transformers therefore buys
almost nothing on its own, because vLLM's continuous batching needs several
requests in flight at once and a single serial worker never provides them. The
throughput win comes from N workers sharing ONE engine, which is what this
script arranges.

Measured before this existed: one HF worker drains a 30-job burst in ~4.3
minutes at mean arm timings, and jobs past roughly #23 blow through
JOB_DEADLINE_SECONDS entirely when the arm mix skews to `ours`.

One engine, not N. Each vLLM engine reserves its share of the card up front
(gpu_memory_utilization), so four engines would each get a quarter of the KV
cache and batch nothing. One engine holding the whole card, fed by four
clients, is the configuration that actually batches.

Shutdown: SLURM tears the whole process group down with the job cgroup, and
this script also terminates its children on SIGTERM/SIGINT so an interactive
Ctrl-C does not leave orphaned workers polling the study server.

    python run_worker_pool.py --workers 4
"""

import argparse
import os
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import config
import vllm_server


def log(message):
    print("[pool %s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--supabase-url", default=config.SUPABASE_URL,
                        help="Supabase project the workers claim jobs from")
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker processes sharing the engine (default %(default)s)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=config.WORKER_POLL_SECONDS)
    parser.add_argument("--vllm-log", default=None)
    args = parser.parse_args()

    assert args.workers >= 1, "--workers must be at least 1"
    assert os.environ.get("STUDY_WORKER_TOKEN"), (
        "STUDY_WORKER_TOKEN must be exported and match study_config.worker_token")
    assert config.SUPABASE_ANON_KEY, "STUDY_SUPABASE_ANON_KEY must be exported (PUBLIC key)"
    assert args.supabase_url, "STUDY_SUPABASE_URL must be set"

    log("launching vLLM engine (%s on %s, tp=%s)"
        % (config.BASE_MODEL_ID, args.device, args.tensor_parallel_size or 1))
    _proc, base_url = vllm_server.start_engine(
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        log_path=args.vllm_log,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )
    # Registered once here rather than by each worker: repeated registration of
    # the same adapter name is redundant, and a race between four workers doing
    # it simultaneously is a needless failure mode.
    vllm_server.register_lora(base_url, config.LORA_NAME, config.LORA_PATH)
    log("engine ready at %s; starting %d workers" % (base_url, args.workers))

    children = []
    for index in range(args.workers):
        child = subprocess.Popen(
            [sys.executable, os.path.join(_HERE, "run_worker.py"),
             "--supabase-url", args.supabase_url,
             "--backend", "vllm",
             "--vllm-url", base_url,
             "--poll-seconds", str(args.poll_seconds)],
            env=dict(os.environ))
        children.append(child)
        log("worker %d pid=%d" % (index, child.pid))

    def shutdown(signum, _frame):
        log("signal %d; stopping %d workers" % (signum, len(children)))
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # A worker that dies is a real fault — it means the engine rejected
    # something or the study server is unreachable — so surface it rather than
    # silently running degraded with three of four workers.
    while True:
        for index, child in enumerate(children):
            code = child.poll()
            if code is not None:
                log("worker %d (pid %d) EXITED with code %s; shutting the pool down"
                    % (index, child.pid, code))
                shutdown(signal.SIGTERM, None)
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
