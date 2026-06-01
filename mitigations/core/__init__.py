"""Shared library for the single-turn mitigation harness.

`explore_mitigations.py` was split into this package: each module owns one
responsibility (config constants, model routing, prompts/conditions, data
loaders, LLM calls, checkpointing, metrics, the per-trial run loop).
`explore_mitigations` itself re-exports the full public surface so existing
`from explore_mitigations import ...` callers keep working unchanged.

Importing this package ensures the sibling `mitigations/` directory and the
repo root are on sys.path, so the absolute imports of `our_mitigation` (lazy,
in `prompts.get_conditions`) and `open_router_utils` resolve regardless of how
the caller was launched.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # mitigations/core
_MITIGATIONS = os.path.dirname(_HERE)                        # mitigations
_REPO_ROOT = os.path.dirname(_MITIGATIONS)                   # repo root
for _p in (_MITIGATIONS, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
