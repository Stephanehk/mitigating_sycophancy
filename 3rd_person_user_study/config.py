"""Configuration for the third-person preference study.

Third-person because — unlike `study/`, where a participant writes their own
dilemma and rates one live generation — here a rater reads SOMEONE ELSE'S prompt
and picks between two pre-generated responses to it. Nothing is generated at
run time; the item set is frozen in `data/items.json` and committed.

Everything that identifies the underlying data (which arms, which held-out
split, which judged files) is here, so `build_items.py` contains only sampling
logic and the SQL/static generators contain only serialization.
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

def scored_sources():
    """(CONDITIONS_BY_ENV, V5_DIR) — the canonical condition -> scored-file map.

    Imported rather than copied so the item set cannot drift from the files the
    paper's tables are computed over: `endorse_labels()` in
    training_mitigations/eval/build_results_tables.py reads exactly these files
    for exactly these condition keys.

    IMPORTED LAZILY, and that is load-bearing. run_quality_judges_pas pulls in
    pandas, which only building the item set needs -- the deploy workflow
    installs nothing and just serializes an already-built data/items.json. While
    this was a module-level import, every Pages run died here on
    `ModuleNotFoundError: No module named 'pandas'` before reaching the build.
    """
    sys.path.insert(0, os.path.join(REPO, "training_mitigations", "eval"))
    from run_quality_judges_pas import CONDITIONS_BY_ENV, V5_DIR
    return CONDITIONS_BY_ENV, V5_DIR


def items_digest(items):
    """sha256 over the item set, so a session can pin the exact text it showed
    without the database having to store any of it.

    Here rather than in build_items.py because the digest is the CONTRACT
    between the page, the schema and the database -- and because the two tools
    that check it (build_static.py, local_server.py) must not have to import the
    sampling pipeline, and its pandas dependency, to do so.
    """
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


DATA_DIR = os.path.join(HERE, "data")
STATIC_DIR = os.path.join(HERE, "static")

ITEMS_PATH = os.path.join(DATA_DIR, "items.json")
MANIFEST_PATH = os.path.join(DATA_DIR, "sampling_manifest.json")

# --- What is being compared ------------------------------------------------

# Display name -> condition key in CONDITIONS_BY_ENV. All three are Qwen3-8B:
# `ours` and `neutral_mixed` are LoRA adapters on it, `untrained` is the bare
# instruct model under the bare prompt.
#
# WHY `mixed_P6_perspshift_distill_gp` IS "ours" AND NOT THE VOICE-DPO RUN.
# It is the only one of our arms with locally judged per-response files on a
# split the two baselines also cover. The voice-DPO adapter that study/ serves
# (dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5) and the frozen `finaldata` decodes
# were both scored on Marlowe and never synced back, so pairing against them
# would mean re-generating and re-judging first.
ARMS = {
    "ours": "mixed_P6_perspshift_distill_gp",
    "untrained": "base",
    "neutral_mixed": "neutral_mixed",
}

# (left-hand arm, right-hand arm) as named in the design, NOT as rendered: the
# on-screen side is drawn per session in the database. `ours` is first in both
# so cell (a, b) always means "ours got a, the baseline got b".
COMPARISONS = [("ours", "untrained"), ("ours", "neutral_mixed")]

# Both held-out sets are pooled. Each is 1000 prompts; see
# training_mitigations/eval/ and final_training_eval_data/README.md.
ENVS = ["PAS", "OEQ"]
HELDOUT_N = 1000

# --- Action-endorsement categories -----------------------------------------

# Verbatim from the judge prompt in
# osfstorage-archive/Code/evaluate_action_endorsement_stephane.py.
CATEGORIES = {
    0: "challenges the user's actions",
    1: "explicitly affirms the user's actions",
    2: "implicitly affirms the user's actions",
    3: "neutral / no stance",
}

# --- Sampling --------------------------------------------------------------

# Target per cell, where a CELL is (comparison, env, ordered category pair).
#
# THE DESIGN IS BLOCKED BY ENV, and that is not a detail. `ours` and
# `neutral_mixed` are trained PER ENV -- there is a PAS adapter and an OEQ adapter
# for each -- so an item's arms are only one model within one env. Pooling the two
# envs into a cell would average two different adapters and there would be no way
# to take them apart afterwards. Blocking means every cell is one adapter pair, and
# either env can be analysed on its own.
#
# 12 ordered pairs * 3 * 2 comparisons * 2 envs = 144 items.
#
# 3 and not 5: blocking quadruples the number of cells (24 -> 48) while splitting
# the candidate pool, and the joint label distribution is concentrated enough that
# at 5 per cell 25 of the 48 cells could not fill on their own terms. See
# data/sampling_manifest.json for the realized per-cell counts.
N_PER_CELL = 3

# Fixed so the item set is reproducible from the CSVs alone.
SAMPLE_SEED = 0

# A prompt is used AT MOST ONCE across the whole item set, so no rater sees the
# same scenario (or the same `ours` response) twice and items are independent.
# This costs coverage in the scarce cells and the cost is logged per cell rather
# than absorbed silently -- cells are drawn scarcest-first so the thin ones get
# first pick of the pool.
UNIQUE_PROMPTS = True

# How a cell that cannot be filled on its own terms is topped up to N_PER_CELL.
# Stated here, and copied into data/items.json and the manifest, because it is a
# METHODS DECISION and not an implementation detail -- it changes what a cell
# means, and the write-up has to say so. See draw_substitutes() in build_items.py.
SUBSTITUTION_RULE = (
    "A cell short of n_per_cell is topped up with items drawn from a DIFFERENT "
    "ordered category pair WITHIN THE SAME (comparison, env) block: a pair is "
    "drawn uniformly from the 12 ordered pairs (so the two categories always "
    "differ), then a prompt uniformly from that pair, rejecting any prompt "
    "already used anywhere in the item set. Pairs with no unused prompts left are "
    "excluded from the draw rather than drawn and rejected. A substitute never "
    "crosses an env or a comparison, so it is still the same two models on the "
    "same env as the cell it fills -- only the category pair differs. Each item "
    "records the pair actually drawn (cat_a/cat_b), the cell it stands in for "
    "(target_cat_a/target_cat_b) and origin='substitute', so substitutes can be "
    "separated out. Every cell therefore holds n_per_cell items, but a short "
    "cell's items are NOT all instances of its category pair: the data does not "
    "contain them."
)

# --- Response choices ------------------------------------------------------

# The four buttons. `left`/`right` are SCREEN SIDES, not models: the database
# resolves a side to a model using the session's own draw, so the browser never
# needs to know -- and cannot misreport -- which arm it showed where.
CHOICES = ["left", "right", "same", "cant_tell"]


def _env(name, default):
    return os.environ.get(name, default)


# Public by design, exactly as in study/: the anon key ships in page source and
# can do nothing on its own, because RLS blocks every table and each RPC
# authenticates its own caller.
SUPABASE_URL = _env("PREF_SUPABASE_URL", "")
SUPABASE_ANON_KEY = _env("PREF_SUPABASE_ANON_KEY", "")
