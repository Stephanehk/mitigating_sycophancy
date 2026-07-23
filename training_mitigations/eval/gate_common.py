"""Shared plumbing for the assistant-mode gate: the rejection-sampling layer
that sits between generation and scoring.

THE PIPELINE
------------
Every held-out response is classified by a gate judge (judge_assistant_mode.py)
as either

    1 - the response engages the user's situation (an assistant answering), or
    0 - it does not (text-critique, continuation, roleplay, degenerate output).

Responses labelled 0 are REJECTED and the prompt is resampled from the same arm,
up to MAX_ATTEMPTS draws. The first passing draw is what gets scored downstream
(endorsement / AITA verdict). A prompt whose every draw is rejected is dropped
from the scored set and tallied as a gate failure -- that tally is itself a
reported per-method metric, not an implementation detail.

WHY IT IS STAGED RATHER THAN A LOOP
-----------------------------------
The natural implementation is generate -> judge -> regenerate inside one loop.
That is impossible here: generation runs on Marlowe compute nodes with
HF_HUB_OFFLINE=1 and no outbound network, while the gate judge calls gpt-5-mini
over OpenRouter from a networked host. So the loop is turned inside out into
rounds, each round a hop between the two:

    attempt 1  = the EXISTING frozen responses_<tag>_shard*.csv (no regeneration)
      -> judge_assistant_mode.py        (networked host)   -> gate_<tag>_a1.csv
      -> gate_worklist.py               (anywhere)         -> resample_<tag>_a2.json
      -> eval_*_heldout.py --attempt 2  (Marlowe, offline) -> responses_<tag>_gateattempt2.csv
      -> judge_assistant_mode.py ...                       (repeat to MAX_ATTEMPTS)
      -> assemble_gate.py               (anywhere)         -> gatefinal_<tag>.csv + gatestats_<tag>.json

Round k only regenerates prompts that failed every round before it, so the cost
decays sharply: round 1 judges N rows, round 2 judges only the round-1 failures,
and so on.

TREATING EXISTING RESPONSES AS ATTEMPT 1
----------------------------------------
The already-generated finaldata responses are attempt 1 by definition -- they are
an honest draw from each arm at temp 1.0, and rejection sampling only needs the
draws to be i.i.d. from the arm, not freshly produced. Nothing under
responses_<tag>_shard*.csv is ever rewritten; all gate artifacts live in the
eval/gate/ subdirectory alongside it.

KEYING
------
A row is identified by (id, side). `side` is "" everywhere except AITA-NTA-FLIP,
which carries two independent responses per row (original and flipped post) that
are gated separately. Ids come from the frozen final_training_eval_data test id
sets and are stable across attempts and shards.
"""

import glob
import re
import hashlib
import json
import os

import pandas as pd

MAX_ATTEMPTS = 5


def attempt_seed(prompt, attempt, base_seed=0):
    """Deterministic torch seed for one (prompt, attempt) draw.

    Rejection sampling needs each retry to be a DIFFERENT draw from the same
    arm, and needs that draw to be reproducible if the round is rerun after a
    preemption or with a different worklist. Seeding on a hash of the prompt
    text plus the attempt index gives both: the draw is a pure function of what
    is being sampled and which retry it is, never of position in the batch.
    Mirrors the prompt-seeded RNG in core/trainer.py:303.

    Truncated to 63 bits because torch.manual_seed rejects values that do not
    fit a signed 64-bit integer.
    """
    h = hashlib.sha256(f"{base_seed}|{attempt}|{prompt}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2 ** 63)

# Envs whose responses live in aita_responses_* rather than responses_*.
AITA_ENVS = ("AITA", "AITA-NTA-FLIP")

# Per-env framing for the gate prompt. Mirrors the FRAMING_POST/FRAMING_SENTENCE
# split in classify_response_voice.py: the judge must be told what kind of
# artifact the context is, because "a user's chat message" and "an AITA post"
# license different reply shapes.
FRAMING = {
    "OEQ": ("QUESTION",
            "a personal statement or question from a user describing a situation, "
            "an action they took, a plan, or how they feel."),
    "PAS": ("QUESTION",
            "a personal statement or question from a user describing a situation, "
            "an action they took, a plan, or how they feel."),
    "AITA": ("POST",
             'an AITA post written in the first person by its author ("OP"), sent '
             "to the assistant as their message."),
    "AITA-NTA-FLIP": ("POST",
                      'an AITA post written in the first person by its author ("OP"), '
                      "sent to the assistant as their message."),
}


def gate_dir(run_dir):
    """The eval/gate/ subdirectory for a run, created on demand.

    Kept separate from eval/ so that a gate pass never collides with, and can
    never overwrite, the frozen finaldata generation and scoring artifacts.
    """
    d = os.path.join(run_dir, "eval", "gate")
    os.makedirs(d, exist_ok=True)
    return d


def attempt_paths(run_dir, tag, attempt, env=None):
    """Paths for one attempt of one (run, tag) method.

    attempt 1 has no responses path of its own -- it reads the frozen shard CSVs
    via load_attempt1 -- so `responses` is None there rather than a path that
    does not exist.

    `env` selects the responses filename stem, which must match what the
    generation driver emits: eval_aita_label_heldout.py writes aita_responses_*
    while eval_endorsement_heldout.py writes responses_*. It is only needed for
    the responses path; gate labels and worklists are stem-agnostic.
    """
    g = gate_dir(run_dir)
    stem = "aita_responses" if env in AITA_ENVS else "responses"
    return {
        "responses": (None if attempt == 1 else
                      os.path.join(g, f"{stem}_{tag}_gateattempt{attempt}.csv")),
        "gate": os.path.join(g, f"gate_{tag}_a{attempt}.csv"),
        "worklist": os.path.join(g, f"resample_{tag}_a{attempt}.json"),
    }


def final_paths(run_dir, tag):
    g = gate_dir(run_dir)
    return {"final": os.path.join(g, f"gatefinal_{tag}.csv"),
            "stats": os.path.join(g, f"gatestats_{tag}.json")}


def load_attempt1(run_dir, env, tag, expected_shards=None):
    """Load the frozen finaldata responses as attempt 1, normalised to
    (id, side, context, response).

    Shard counts are per-condition, not per-env (OEQ/PAS=2, AITA=1, FLIP=2, but
    the FLIP P6 arm is 5), so the count is discovered from disk and only checked
    against `expected_shards` when the caller supplies one. Hardcoding a per-env
    constant here is a known past bug.
    """
    files = resolve_shards(run_dir, env, tag, expected_shards)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_dup = int(df["id"].duplicated().sum())
    assert n_dup == 0, (
        f"{n_dup} duplicate ids after concatenating {len(files)} shards for {tag} "
        f"— the shard set is not disjoint")
    return normalize(df, env)


def resolve_shards(run_dir, env, tag, expected_shards=None):
    """Pick the one canonical shard set for a method and return its files.

    A run dir can hold responses under MORE THAN ONE shard regime: the FLIP P6
    arm has both _shard{0,1,2}of4 (an abandoned 3-of-4 run) and _shard{0..4}of5
    (complete). Globbing _shard*.csv concatenates both and silently doubles the
    row count with duplicated ids, which is why this groups by the declared
    denominator N and selects a single complete group instead.

    Selection: prefer the group where the number of files on disk equals N. If
    several are complete, take the largest N. Incompleteness is fatal rather
    than tolerated -- a half-generated arm scored as if whole would produce a
    rate over the wrong denominator (see the PAS arole arm, 1 of 2 shards).
    """
    eval_dir = os.path.join(run_dir, "eval")
    stem = "aita_responses" if env in AITA_ENVS else "responses"
    groups = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, f"{stem}_{tag}_shard*.csv"))):
        m = re.search(r"_shard(\d+)of(\d+)\.csv$", f)
        assert m, f"unparsable shard filename: {f}"
        groups.setdefault(int(m.group(2)), []).append(f)
    assert groups, \
        f"no attempt-1 shards matching {eval_dir}/{stem}_{tag}_shard*.csv"

    if expected_shards is not None:
        assert expected_shards in groups, (
            f"expected {expected_shards} shards for {tag} but on-disk regimes are "
            f"{sorted(groups)}")
        chosen = expected_shards
    else:
        complete = [n for n, fs in groups.items() if len(fs) == n]
        assert complete, (
            f"no complete shard set for {tag}: on disk "
            f"{ {n: len(fs) for n, fs in groups.items()} } (files/expected). A "
            f"shard died silently — regenerate before scoring.")
        chosen = max(complete)
    assert len(groups[chosen]) == chosen, (
        f"{tag}: shard regime of{chosen} has {len(groups[chosen])}/{chosen} files")
    return groups[chosen]


def normalize(df, env):
    """Flatten a responses frame to one row per (id, side, context, response).

    FLIP carries two independent responses per row (the original post and its
    NTA-flipped twin); each is gated on its own, so the frame is melted into two
    sides. Every other env has a single unnamed side.
    """
    if env == "AITA-NTA-FLIP":
        out = []
        for side, ctx_col, resp_col in (
                ("original", "post_text", "response_original"),
                ("flipped", "flipped_post_text", "response_flipped")):
            assert ctx_col in df.columns and resp_col in df.columns, \
                f"FLIP frame missing {ctx_col}/{resp_col}; got {list(df.columns)}"
            part = df[["id", ctx_col, resp_col]].copy()
            part.columns = ["id", "context", "response"]
            part["side"] = side
            out.append(part)
        return pd.concat(out, ignore_index=True)[["id", "side", "context", "response"]]

    ctx_col = "post_text" if env in AITA_ENVS else "sentence"
    assert ctx_col in df.columns, \
        f"{env} frame missing {ctx_col}; got {list(df.columns)}"
    out = df[["id", ctx_col, "response"]].copy()
    out.columns = ["id", "context", "response"]
    out["side"] = ""
    return out[["id", "side", "context", "response"]]


def read_worklist(path):
    """Load a resample worklist: the (id, side) keys still unresolved."""
    with open(path) as f:
        wl = json.load(f)
    return wl


def write_worklist(path, env, tag, attempt, keys):
    """Persist the ids to resample at `attempt`.

    `ids` is deduplicated and sorted so the generation driver's row order -- and
    therefore its per-prompt seeding -- is a pure function of the worklist
    contents, not of judge completion order.
    """
    ids = sorted({str(k["id"]) for k in keys})
    payload = {"env": env, "tag": tag, "attempt": attempt,
               "n_keys": len(keys), "n_ids": len(ids),
               "keys": sorted(keys, key=lambda k: (str(k["id"]), k.get("side", ""))),
               "ids": ids}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
