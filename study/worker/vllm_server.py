"""Launch and address the study's vLLM engine.

ONE engine serves all four arms. Qwen3-8B is loaded once as the base model,
and our OEQ-trained DPO LoRA is registered as a named adapter via
`--enable-lora`, so a request selects its arm purely by which model name it
addresses:

  model=<base_model_id>  -> untrained / syco / neutral (they differ only in
                            their system prompt)
  model=<lora_name>      -> ours

That is what makes a four-arm live study fit on a single GPU with no model
reloading between participants.

This module reuses `training_mitigations/core/vllm_sim.py` for the launch and
health-poll rather than duplicating the subprocess plumbing: that code already
handles CUDA_VISIBLE_DEVICES mapping, the bf16 dtype pin needed for parity
with the HF path, and atexit teardown.

Assumptions, all validated below:
  - Qwen3-8B is present in the HF cache (the cluster's DNS is flaky; see the
    repo's vLLM notes about pre-caching to $HF_HOME/hub with HF_HUB_OFFLINE=1).
  - config.LORA_PATH contains an adapter_config.json whose base model matches
    config.BASE_MODEL_ID, so we cannot silently serve a 32B adapter on the 8B
    base.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_STUDY_DIR)

sys.path.insert(0, _STUDY_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "training_mitigations", "core"))

import config
from vllm_sim import launch_vllm_server, wait_for_vllm_server


def validate_adapter(lora_path, base_model_id):
    """Check that the LoRA at `lora_path` belongs to `base_model_id`.

    Serving an adapter trained on a different base silently produces garbage
    rather than erroring, and with two adapters in runs/ (8B and 32B) that is a
    live risk. Returns the adapter's declared rank, which the caller passes to
    vLLM as --max-lora-rank.
    """
    config_path = os.path.join(lora_path, "adapter_config.json")
    assert os.path.isfile(config_path), (
        "no adapter_config.json at %r — is STUDY_LORA_PATH correct?" % (lora_path,))
    with open(config_path) as handle:
        adapter = json.load(handle)

    declared_base = adapter.get("base_model_name_or_path")
    assert declared_base == base_model_id, (
        "adapter at %r was trained on %r but the study is configured to serve %r"
        % (lora_path, declared_base, base_model_id))

    weights = os.path.join(lora_path, "adapter_model.safetensors")
    assert os.path.isfile(weights), "no adapter_model.safetensors at %r" % (lora_path,)

    rank = adapter.get("r")
    assert isinstance(rank, int) and rank > 0, "adapter has no usable rank: %r" % (rank,)
    return rank


def start_engine(device="cuda:0", gpu_memory_utilization=0.90, max_model_len=8192,
                 log_path=None, tensor_parallel_size=None, enforce_eager=False):
    """Start the study's vLLM engine and block until it answers /health.

    `tensor_parallel_size` shards the model across that many GPUs. Qwen3-8B in
    bf16 is ~16.4GB of weights, so it does NOT fit on a single 16GB a4000
    (next5/next6) — pass 2 there. A single 48GB a6000 (next7) needs no sharding.

    Returns (proc, base_url). The caller is responsible for keeping the process
    alive; SLURM tears it down with the job cgroup, and vllm_sim registers an
    atexit terminate for interactive runs.
    """
    rank = validate_adapter(config.LORA_PATH, config.BASE_MODEL_ID)
    proc, base_url = launch_vllm_server(
        config.BASE_MODEL_ID,
        device=device,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        log_path=log_path,
        enable_lora=True,
        max_lora_rank=rank,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
    )
    wait_for_vllm_server(proc, base_url)
    return proc, base_url


def register_lora(base_url, lora_name, lora_path, timeout=120):
    """Load our adapter into a running engine under `lora_name`.

    vLLM's runtime LoRA endpoint is enabled by vllm_sim's
    VLLM_ALLOW_RUNTIME_LORA_UPDATING. Registering at runtime rather than via a
    launch flag means a re-registration after an adapter swap does not require
    restarting the engine.

    Idempotent: an already-registered name reports as an error containing
    "already", which is treated as success.
    """
    import requests

    response = requests.post(
        base_url + "/v1/load_lora_adapter",
        json={"lora_name": lora_name, "lora_path": os.path.abspath(lora_path)},
        timeout=timeout,
    )
    if response.status_code == 200:
        return True
    body = response.text or ""
    assert "already" in body.lower(), (
        "failed to register LoRA %r from %r: HTTP %d %s"
        % (lora_name, lora_path, response.status_code, body[:500]))
    return True


def served_model_name(adapter):
    """Map a generation plan's `adapter` field to the vLLM model name.

    `adapter="lora"` addresses our trained adapter; None addresses the base.
    """
    if adapter is None:
        return config.BASE_MODEL_ID
    assert adapter == "lora", "unknown adapter selector: %r" % (adapter,)
    return config.LORA_NAME
