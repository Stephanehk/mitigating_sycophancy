"""Resumable per-row checkpointing: derive the .checkpoint.jsonl path,
load prior rows keyed by (id, condition), and thread-safely append one
result record per completed task."""

import json
import os


def checkpoint_path_for(out_path):
    """Derive the per-row checkpoint path from --out.

    Replaces the .csv extension with .checkpoint.jsonl. The checkpoint
    captures one JSON object per completed (id, condition) task as soon
    as the future returns, so a mid-run crash leaves all prior progress
    on disk and a re-run can resume from there.
    """
    base, _ = os.path.splitext(out_path)
    return base + ".checkpoint.jsonl"


def load_checkpoint(path):
    """Return dict[(id, condition_name)] -> result-record from `path`.

    Returns {} if the file doesn't exist. Asserts on malformed lines so a
    corrupted checkpoint can't silently shrink a resumed run. Each record
    is the dict returned by `run_one`, JSON round-tripped (None becomes
    null and back to None; ints / strs / nested trace dicts round-trip
    cleanly).
    """
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            assert "id" in rec and "condition" in rec, (
                f"checkpoint row {ln} missing id or condition: {rec!r}"
            )
            rows[(rec["id"], rec["condition"])] = rec
    return rows


def append_checkpoint(path, record, lock):
    """Thread-safely append one result record to the checkpoint file.

    `lock` is a `threading.Lock` shared across the executor's worker
    threads. We open/close per call so a SIGINT or process kill leaves
    the file flushed up through the last successful write.
    """
    with lock:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
