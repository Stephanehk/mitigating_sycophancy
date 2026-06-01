"""Cheap process-memory probes for diagnosing the host-RAM OOM in the DPO
training loop. Both instruments are designed to add < 0.1% to wall time.

  - start_mem_sampler(path): a daemon thread that appends a CSV row (host
    RSS + per-GPU allocated/reserved) every `interval` seconds. Runs off
    the training path, so it costs nothing on the generate hot loop, and
    it catches transient spikes that fall between SLURM's ~30s sacct
    samples. The file is line-buffered so rows survive the SIGKILL the
    OOM killer sends — this is what we lost to stdout buffering last run.

  - mem_line(tag): a one-line flushed in-band marker for the loop, so the
    continuous CSV can be aligned to what the code was doing (which
    prompt, sampling vs veto).

Host RSS comes from /proc/self/status (VmRSS = current, VmHWM = peak).
CUDA figures are torch caching-allocator counters — plain bookkeeping
reads, no device sync, so they never stall generate.

Assumptions:
  - Linux: /proc/self/status exists and reports VmRSS / VmHWM in kB.
  - torch.cuda is importable; with no visible GPUs the device list is
    empty and only host RSS is logged (the probes still work).
"""

import threading
import time

import torch


def _devices(devices):
    """Resolve the device-index list to log. None -> all visible CUDA
    devices, in index order. An explicit list is returned unchanged.
    """
    if devices is None:
        return list(range(torch.cuda.device_count()))
    return devices


def _read_proc_mem():
    """Return (vmrss_kb, vmhwm_kb) for this process from
    /proc/self/status. One small file read. VmRSS is the current
    resident set; VmHWM is the high-water mark since process start.
    """
    rss = hwm = 0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1])
    return rss, hwm


def _cuda_cols(devices):
    """Flat list [alloc_mb, resv_mb, alloc_mb, resv_mb, ...] across
    `devices`. Reads allocator counters only — no synchronize(), so this
    does not stall in-flight generate calls.
    """
    cols = []
    for d in devices:
        cols.append(torch.cuda.memory_allocated(d) / 1e6)
        cols.append(torch.cuda.memory_reserved(d) / 1e6)
    return cols


def start_mem_sampler(path, devices=None, interval=2.0):
    """Spawn a daemon thread logging host + GPU memory to `path` every
    `interval` seconds.

    The file is opened line-buffered so partial output is preserved if
    the process is SIGKILLed (the exact failure mode we're chasing).

    Inputs:
      path:     CSV output path.
      devices:  device indices to log, or None for all visible GPUs.
      interval: seconds between samples (off the training path).

    Returns the threading.Event used to stop the thread; set it to halt
    sampling (the thread is a daemon, so leaving it running is harmless).
    """
    devices = _devices(devices)
    f = open(path, "w", buffering=1)
    header = ["t_s", "rss_mb", "hwm_mb"]
    for d in devices:
        header += [f"cuda{d}_alloc_mb", f"cuda{d}_resv_mb"]
    f.write(",".join(header) + "\n")

    t0 = time.time()
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            rss, hwm = _read_proc_mem()
            cols = [f"{time.time() - t0:.1f}", f"{rss / 1024:.0f}", f"{hwm / 1024:.0f}"]
            cols += [f"{v:.0f}" for v in _cuda_cols(devices)]
            f.write(",".join(cols) + "\n")
            stop.wait(interval)

    threading.Thread(target=_loop, name="mem-sampler", daemon=True).start()
    return stop


def mem_line(tag, devices=None):
    """Print one flushed marker line tagging the current host + GPU
    memory with `tag`. Use at phase boundaries in the training loop so
    the continuous CSV timeline can be aligned to code phases.

    Cost: one /proc read plus a few allocator-counter reads (~tens of
    microseconds), negligible against a generate call.
    """
    devices = _devices(devices)
    rss, hwm = _read_proc_mem()
    cuda = _cuda_cols(devices)
    parts = [f"rss={rss / 1024:.0f}MB", f"hwm={hwm / 1024:.0f}MB"]
    for i, d in enumerate(devices):
        parts.append(f"cuda{d}={cuda[2 * i]:.0f}/{cuda[2 * i + 1]:.0f}MB")
    print(f"[mem] {tag}  " + "  ".join(parts), flush=True)
