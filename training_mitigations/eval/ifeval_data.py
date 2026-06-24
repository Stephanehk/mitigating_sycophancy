"""Load IFEval records from a local jsonl (preferred, network-free) or the HF
hub. Single source of truth shared by eval_ifeval.py (gen) and score_ifeval.py.

A record is a dict with keys: key, prompt, instruction_id_list, kwargs. The
local jsonl (sycoharm_runs/ifeval/ifeval_data.jsonl) is a verbatim dump of
google/IFEval, so loading it avoids any hub access (important when HF_HUB_OFFLINE
is set for the model cache) and pins the eval set for reproducibility.

`source` is treated as a local jsonl path if it is an existing file, else as an
HF dataset name to load via `datasets.load_dataset(source, split)`.
"""

import json
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # training_mitigations/
DEFAULT_DATA = os.path.join(_HERE, "sycoharm_runs", "ifeval", "ifeval_data.jsonl")


def load_records(source=DEFAULT_DATA, split="train"):
    """Return a list of IFEval record dicts (key, prompt, instruction_id_list,
    kwargs). Reads a local jsonl when `source` is a file, else the HF hub."""
    if os.path.isfile(source):
        with open(source) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        assert recs, f"no records in {source}"
        return recs
    from datasets import load_dataset
    ds = load_dataset(source, split=split)
    return [dict(key=ex["key"], prompt=ex["prompt"],
                 instruction_id_list=ex["instruction_id_list"],
                 kwargs=ex["kwargs"]) for ex in ds]
