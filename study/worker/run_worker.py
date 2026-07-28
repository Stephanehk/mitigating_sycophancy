"""GPU-side worker for the sycophancy user study.

Runs on next6/next7, next to a vLLM engine serving Qwen3-8B plus our OEQ LoRA.
It reaches the study server by connecting OUTBOUND over HTTPS and long-polling
for work; nothing ever connects into the cluster. That is what lets the study
site live on the public internet while generation happens on a private-IP node.

Loop, per iteration:
  1. POST /api/worker/claim   (long poll, returns within ~20s either way)
  2. Build the arm's generation plan (prompts.build_generation_plan)
  3. Run each stage against the local vLLM engine, extending the job lease
     between stages so a slow two-stage neutral generation is not reaped
  4. POST /api/worker/complete, or /api/worker/fail on any error

Failure policy. This file is the one place in the study that deliberately
catches broad exceptions, because the brief requires the site to degrade
gracefully: a transient CUDA hiccup, an HTTP blip, or a malformed generation
must become a recorded failed attempt (which the server requeues) rather than
a dead worker and a participant stuck on a loading screen. Programming errors
in prompt construction still surface, because they are raised before the
try block and reported with their traceback.

Run it with:
    python study/worker/run_worker.py --server-url https://<host> --device cuda:0
The shared secret comes from STUDY_WORKER_TOKEN in the environment.
"""

import argparse
import os
import socket
import sys
import time
import traceback

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY_DIR)
sys.path.insert(0, _HERE)

import config
from prompts import build_generation_plan
import vllm_server


def log(message):
    """Timestamped stderr line. Worker logs go to the SLURM job's stderr."""
    sys.stderr.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), message))
    sys.stderr.flush()


def make_worker_id():
    """A stable, human-readable id: hostname plus pid."""
    return "%s:%d" % (socket.gethostname(), os.getpid())


class StudyClient:
    """Thin HTTP client for the study server's worker API."""

    def __init__(self, server_url, token, worker_id, timeout=60):
        self.base = server_url.rstrip("/")
        self.worker_id = worker_id
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-Worker-Token": token})

    def _post(self, path, payload, timeout=None):
        response = self.session.post(
            self.base + path, json=payload, timeout=timeout or self.timeout)
        assert response.status_code != 401, (
            "study server rejected the worker token; check STUDY_WORKER_TOKEN "
            "matches the server's")
        response.raise_for_status()
        return response.json()

    def claim(self):
        """Ask for a job. Returns the job dict, or None if the queue is empty.

        Returns immediately either way — the server no longer holds the request
        open. The caller is responsible for sleeping between empty claims; see
        the loop in main() and config.WORKER_POLL_SECONDS for why the long poll
        was removed.
        """
        body = self._post("/api/worker/claim", {"worker_id": self.worker_id}, timeout=30)
        return body.get("job")

    def heartbeat(self, job_id):
        """Extend the lease on an in-flight job."""
        return self._post("/api/worker/heartbeat",
                          {"job_id": job_id, "worker_id": self.worker_id}).get("extended")

    def complete(self, job_id, response_text, stages):
        """Report a finished generation. Returns whether the server took it."""
        return self._post("/api/worker/complete", {
            "job_id": job_id,
            "worker_id": self.worker_id,
            "response_text": response_text,
            "stages": stages,
        }).get("accepted")

    def fail(self, job_id, error):
        """Report a failed attempt."""
        return self._post("/api/worker/fail", {
            "job_id": job_id, "worker_id": self.worker_id, "error": error,
        }).get("status")


class VLLMBackend:
    """Generation against a local vLLM OpenAI-compatible engine.

    Selects the arm's model by name: our LoRA is registered as a named adapter
    on the same engine, so `adapter="lora"` and `adapter=None` address one
    process. Requires a GPU that can hold the model, or working NCCL for
    tensor parallelism — see HFBackend for the a4000 case.
    """

    def __init__(self, base_url):
        from openai import OpenAI
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url + "/v1", api_key="EMPTY", timeout=300)

    def describe(self):
        return "vLLM engine at %s (%s + LoRA %s)" % (
            self.base_url, config.BASE_MODEL_ID, config.LORA_NAME)

    def model_label(self, adapter):
        return vllm_server.served_model_name(adapter)

    def generate(self, adapter, system, user_text):
        """Run one generation stage.

        `enable_thinking=False` is passed through `chat_template_kwargs` so
        Qwen3 behaves exactly as it does in training_mitigations' HF path
        (core/llama_local.build_chat_input), which is the condition our
        adapter's reported numbers came from.

        Returns the generated text, stripped. Raises on empty content, because
        shipping an empty bubble to a participant is worse than retrying.
        """
        model_name = self.model_label(adapter)
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=config.GEN_TEMPERATURE,
            top_p=config.GEN_TOP_P,
            max_tokens=config.GEN_MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": config.ENABLE_THINKING}},
        )
        text = (completion.choices[0].message.content or "").strip()
        assert text, "vLLM returned empty content for model %r" % (model_name,)
        return text


def run_job(backend, study, job):
    """Execute one job end to end and report the result.

    `backend` is any object exposing generate(adapter, system, user_text) and
    model_label(adapter) — VLLMBackend or HFBackend. Keeping the arm logic
    backend-agnostic is what lets the same worker run on a single a6000 (vLLM)
    or on two a4000s (HF + device_map).

    Returns True if the job was completed and accepted by the server.
    """
    # `user_text` is already the composed user turn (vignette + the
    # participant's perspective); the server builds it at job creation so the
    # stored value is exactly what the model receives.
    plan = build_generation_plan(job["arm"], job["user_text"])

    stages = []
    current_user = None
    for index, stage in enumerate(plan["stages"]):
        # A stage with user=None consumes the previous stage's output; that is
        # how the neutral arm's editing pass receives its input.
        user_text = stage["user"] if stage["user"] is not None else current_user
        assert user_text, "stage %r has no user text" % (stage["name"],)
        if index > 0:
            study.heartbeat(job["job_id"])
        started = time.time()
        output = backend.generate(plan["adapter"], stage["system"], user_text)
        stages.append({
            "name": stage["name"],
            "model": backend.model_label(plan["adapter"]),
            "had_system_prompt": stage["system"] is not None,
            "input_chars": len(user_text),
            "output": output,
            "seconds": round(time.time() - started, 3),
        })
        current_user = output

    return study.complete(job["job_id"], current_user, stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True,
                        help="Base URL of the study server, e.g. https://study.example.org")
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm",
                        help="vllm: one engine, LoRA as a named adapter. Needs a GPU "
                             "that holds the model (a6000) or working NCCL for TP. "
                             "hf: transformers + accelerate device_map, LoRA toggled "
                             "with PEFT disable_adapter(). Splits by layer with no NCCL, "
                             "so this is the one that works on 2x a4000 (next5/next6).")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU for the vLLM engine (default cuda:0)")
    parser.add_argument("--device-map", default="auto",
                        help="HF backend only: accelerate device_map (default auto)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--poll-seconds", type=float, default=config.WORKER_POLL_SECONDS,
                        help="Wait between claims when the queue is empty. The server "
                             "returns immediately rather than long-polling, so this is "
                             "where the waiting happens (default %(default)s).")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="vLLM backend only. NOTE: TP across next5/next6 a4000s "
                             "hangs in NCCL; use --backend hf on those nodes.")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Skip torch.compile and CUDA-graph capture. Cuts engine "
                             "startup by several minutes at some per-token cost; a good "
                             "trade for this study's low request rate.")
    parser.add_argument("--vllm-log", default=None, help="Path for the vLLM server log")
    parser.add_argument("--vllm-url", default=None,
                        help="Address an ALREADY-RUNNING vLLM engine instead of launching one")
    parser.add_argument("--max-jobs", type=int, default=0,
                        help="Exit after this many completed jobs (0 = run forever). For smoke tests.")
    args = parser.parse_args()

    token = os.environ.get("STUDY_WORKER_TOKEN", "").strip()
    assert token, "STUDY_WORKER_TOKEN must be set (must match the study server's)"
    if config.IS_PRODUCTION:
        assert token != config.INSECURE_WORKER_TOKEN, (
            "refusing to run a production worker with the development token")

    worker_id = make_worker_id()
    log("worker %s starting; server=%s" % (worker_id, args.server_url))

    if args.backend == "hf":
        log("loading HF backend (%s, device_map=%s)" % (config.BASE_MODEL_ID, args.device_map))
        from hf_backend import HFBackend
        backend = HFBackend(device_map=args.device_map, device=args.device)
    elif args.vllm_url:
        backend = VLLMBackend(args.vllm_url.rstrip("/"))
        vllm_server.register_lora(backend.base_url, config.LORA_NAME, config.LORA_PATH)
    else:
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
        vllm_server.register_lora(base_url, config.LORA_NAME, config.LORA_PATH)
        backend = VLLMBackend(base_url)

    log("backend ready: %s" % backend.describe())

    study = StudyClient(args.server_url, token, worker_id)
    completed = 0
    attempted = 0
    idle_since = time.time()

    while True:
        # The claim call is the only place a transient network error should
        # not consume a job attempt, so it is retried with a short backoff.
        try:
            job = study.claim()
        except Exception as exc:
            log("claim failed (%s); retrying in 5s" % exc)
            time.sleep(5)
            continue

        if job is None:
            # A bounded run (--max-jobs, i.e. the smoke test) is DONE when the
            # queue drains, whether its jobs succeeded or failed. Without this
            # the worker spins on an empty queue until the SLURM time limit,
            # holding its GPUs — which is exactly what happened when the first
            # HF smoke run's jobs all failed.
            if args.max_jobs:
                log("queue empty after %d completed / %d attempted; exiting"
                    % (completed, attempted))
                return 0 if completed >= args.max_jobs else 1
            if time.time() - idle_since > 300:
                log("idle; queue empty")
                idle_since = time.time()
            # The wait the server used to do for us. Without this the worker
            # would busy-loop the claim endpoint, which behind Lambda is a
            # bill rather than merely rude.
            time.sleep(args.poll_seconds)
            continue

        idle_since = time.time()
        attempted += 1
        log("claimed job %s arm=%s attempt=%s" % (job["job_id"], job["arm"], job["attempts"]))
        started = time.time()
        try:
            accepted = run_job(backend, study, job)
        except Exception as exc:
            detail = "%s: %s" % (type(exc).__name__, exc)
            log("job %s FAILED after %.1fs — %s" % (job["job_id"], time.time() - started, detail))
            traceback.print_exc()
            # Reporting the failure is itself best-effort: if the server is
            # unreachable the job's lease simply expires and it is requeued.
            try:
                status = study.fail(job["job_id"], detail)
                log("job %s reported failed; server set status=%s" % (job["job_id"], status))
            except Exception as report_exc:
                log("could not report failure for %s (%s); lease will expire"
                    % (job["job_id"], report_exc))
            continue

        log("job %s done in %.1fs (accepted=%s)"
            % (job["job_id"], time.time() - started, accepted))
        completed += 1
        if args.max_jobs and completed >= args.max_jobs:
            log("reached --max-jobs=%d; exiting" % args.max_jobs)
            return


if __name__ == "__main__":
    main()
