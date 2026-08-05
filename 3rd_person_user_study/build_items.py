"""Build the frozen preference-item set from the judged held-out responses.

WHAT IT PRODUCES. N_PER_CELL items for every CELL, where a cell is one
(comparison, env, ordered category pair): 2 comparisons (ours vs untrained, ours
vs neutral-mixed DPO) x 2 envs x 12 ordered pairs (4 permute 2, so both (a, b) and
(b, a)) = 48 cells. An item is one PROMPT plus the two arms' responses to that same
prompt, where the judge put our arm's response in category a and the baseline's in
category b.

BLOCKED BY ENV. The env is part of the cell, not pooled across it, because `ours`
and `neutral_mixed` are trained PER ENV -- a PAS adapter and an OEQ adapter each.
A cell pooling both envs would average two different adapter pairs with no way to
separate them afterwards. Blocking means every cell is one adapter pair and either
env can be analysed alone.

MATCHED ON PROMPT. Both responses in a pair answer the same prompt, so the
interface shows one prompt and two responses. This is what makes several cells
under-fill: a cell needs prompts on which BOTH arms happened to land in the
required categories, and the joint distribution is far from uniform (our arm is
mostly `challenges`, neutral-mixed DPO is mostly `neutral` and explicitly
affirms almost never). Blocking by env sharpens this -- it quadruples the cells
while splitting the pool -- which is why a short cell is topped up by
config.SUBSTITUTION_RULE, always from within its own (comparison, env) block.

ASSUMPTIONS, all asserted below:
  1. Every referenced scored file exists.
  2. Every endorse label parses to exactly one of {0, 1, 2, 3}.
  3. Within an env, all three arms cover the SAME HELDOUT_N prompts, once each,
     so a merge on `sentence` is a complete one-to-one join and the two arms in
     a pair really did answer the same text.
  4. Every response is non-empty.
  5. Each cell's realized count is min(N_PER_CELL, prompts still unused).

Run:
    python 3rd_person_user_study/build_items.py
"""

import hashlib
import json
import os
import random
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config
sys.path.insert(0, os.path.join(config.REPO, "osfstorage-archive", "Code"))
from evaluate_action_endorsement_stephane import first_label  # noqa: E402


def load_arm(env, arm):
    """One arm's judged responses for one env, as a frame of
    (sentence, response, label).

    Concatenates the arm's scored shards in the order CONDITIONS_BY_ENV lists
    them. Asserts assumptions 1-4 for this arm.
    """
    files = config.CONDITIONS_BY_ENV[env][config.ARMS[arm]]
    frames = []
    for name in files:
        path = os.path.join(config.V5_DIR, name)
        assert os.path.exists(path), f"missing scored frame: {path}"
        frames.append(pd.read_csv(path, usecols=["sentence", "response",
                                                 "endorse_response"]))
    df = pd.concat(frames, ignore_index=True)

    df["label"] = df["endorse_response"].map(first_label)
    assert df["label"].notna().all(), (
        f"{env}/{arm}: {int(df['label'].isna().sum())} unparsable endorse labels")
    df["label"] = df["label"].astype(int)
    assert set(df["label"]) <= set(config.CATEGORIES), (
        f"{env}/{arm}: labels outside 0-3: {sorted(set(df['label']))}")

    assert len(df) == config.HELDOUT_N, f"{env}/{arm}: n={len(df)}, want {config.HELDOUT_N}"
    assert df["sentence"].is_unique, f"{env}/{arm}: duplicate prompts"
    assert df["response"].notna().all(), f"{env}/{arm}: null response"
    assert df["response"].str.strip().str.len().gt(0).all(), f"{env}/{arm}: empty response"
    assert df["sentence"].str.strip().str.len().gt(0).all(), f"{env}/{arm}: empty prompt"

    return df[["sentence", "response", "label"]]


def paired_frame(env, arm_a, arm_b):
    """The complete one-to-one join of two arms over one env's held-out prompts.

    Asserts assumption 3: the join loses nothing, which is the only reason a
    pair can be called "two responses to the same prompt".
    """
    a = load_arm(env, arm_a)
    b = load_arm(env, arm_b)
    assert set(a["sentence"]) == set(b["sentence"]), (
        f"{env}: {arm_a} and {arm_b} cover different prompt sets")

    merged = a.merge(b, on="sentence", suffixes=("_a", "_b"), validate="one_to_one")
    assert len(merged) == config.HELDOUT_N, f"{env}: join produced {len(merged)} rows"
    merged["env"] = env
    return merged


def cell_pools(env, arm_b):
    """(cat_a, cat_b) -> frame of candidate prompts, for ONE (env, baseline) block.

    Keys are the 12 ORDERED off-diagonal category pairs. A prompt appears in
    exactly one key, because each arm assigned it exactly one label.

    NOT pooled over envs, on purpose. `ours` and `neutral_mixed` are trained per
    env, so a cell pooling PAS and OEQ would hold two different adapter pairs
    averaged together; see the N_PER_CELL comment in config.py.
    """
    pooled = paired_frame(env, "ours", arm_b)

    pools = {}
    for cat_a in sorted(config.CATEGORIES):
        for cat_b in sorted(config.CATEGORIES):
            if cat_a == cat_b:
                continue
            pools[(cat_a, cat_b)] = pooled[(pooled["label_a"] == cat_a)
                                           & (pooled["label_b"] == cat_b)]
    assert len(pools) == 12, f"expected 12 ordered pairs, got {len(pools)}"
    return pools


def shuffled_pools(pools_by_block, rng):
    """(block, pair) -> uniformly shuffled list of candidate prompts, where a
    block is (baseline, env).

    Shuffled ONCE, up front, so both passes below can just consume from the
    front of a list: taking the next unused entry of a uniform shuffle is a
    uniform draw, which is what makes the two passes read the same way.

    Sorted before shuffling because the CSV row order must not leak into the
    sample -- `random.shuffle` of an unsorted list is still deterministic given
    the seed, but it would be deterministic in the file's order.
    """
    order = {}
    for block, pools in pools_by_block.items():
        for pair in sorted(pools):
            names = sorted(pools[pair]["sentence"])
            rng.shuffle(names)
            order[(block, pair)] = names
    return order


def make_item(pools_by_block, block, pair, sentence, target, rank, origin):
    """One item. `pair` is what the judge ACTUALLY labelled the two responses;
    `target` is the cell the item is counted under, which differs from `pair`
    only for a substitute.

    `block` is (baseline, env) and the env is part of the item_id: the same
    (comparison, category pair) cell now exists once per env, so an id without the
    env would collide between the PAS and OEQ copies of that cell.
    """
    baseline, env = block
    row = pools_by_block[block][pair].set_index("sentence").loc[sentence]
    assert row["env"] == env, f"{sentence[:40]}: pool env {row['env']} != {env}"
    return {
        "item_id": f"{baseline}-{env}-{target[0]}{target[1]}-{rank}",
        "comparison": baseline,
        "env": env,
        "prompt": sentence,
        "model_a": "ours",
        "model_b": baseline,
        # The judge's labels for the two responses actually shown.
        "cat_a": pair[0],
        "cat_b": pair[1],
        # The cell this item fills. Equal to (cat_a, cat_b) unless it is a
        # substitute, in which case the cell could not be filled on its own terms
        # and this is the ONLY record of which one it stands in for.
        "target_cat_a": target[0],
        "target_cat_b": target[1],
        "origin": origin,
        "response_a": row["response_a"],
        "response_b": row["response_b"],
    }


def draw_matched(pools_by_block, order, used):
    """Pass 1: for each cell, up to N_PER_CELL prompts on which the two arms
    really did land in that cell's two categories.

    Cells are drawn scarcest-first, which is not cosmetic: it gives the thin
    cells -- the ones that decide whether the design is balanced at all -- first
    pick of the shared prompt pool, so a cell with 700 candidates cannot starve
    one with 3.

    Returns (items, free_at_draw). `free_at_draw` is how many candidates this
    cell still had when its turn came, which is the number its `matched` count
    has to be explained by -- `available` alone cannot explain a cell that lost
    candidates to an earlier, scarcer one.
    """
    items = {}
    free_at_draw = {}
    for block, pair in sorted(order, key=lambda key: (len(order[key]), key)):
        free = [name for name in order[(block, pair)] if name not in used]
        taken = free[:config.N_PER_CELL]
        used.update(taken)
        free_at_draw[(block, pair)] = len(free)
        items[(block, pair)] = [
            make_item(pools_by_block, block, pair, sentence, pair, rank, "matched")
            for rank, sentence in enumerate(taken)]
    return items, free_at_draw


def draw_substitutes(pools_by_block, order, used, matched, rng):
    """Pass 2: top every short cell up to N_PER_CELL with substitutes.

    THE RULE, and it is a real change to what a cell means. A cell that cannot be
    filled on its own terms is filled from a DIFFERENT category pair: draw a pair
    uniformly at random from the 12 ordered pairs (the two categories therefore
    always differ), then draw a prompt uniformly from that pair, rejecting
    anything already used. Both the drawn pair and the cell it stands in for are
    recorded on the item (`cat_a`/`cat_b` vs `target_cat_a`/`target_cat_b`), so a
    substitute can always be separated back out.

    Pairs with nothing unused left are filtered out rather than drawn and
    rejected -- the same distribution as rejection sampling, but it terminates.

    CONSEQUENCE TO WRITE UP: the item count is now balanced at N_PER_CELL by
    construction, but a short cell's five items are NOT five instances of that
    category pair. Nine of them could not be, because the data does not contain
    them. Balance here is in the count, not in the contrast.
    """
    substitutes = {}
    for block, pair in sorted(matched):
        shortfall = config.N_PER_CELL - len(matched[(block, pair)])
        drawn = []
        for offset in range(shortfall):
            # Restricted to `b == block`, so a substitute never crosses an env or
            # a comparison: it is still the same two adapters on the same env as
            # the cell it fills, and only the category pair differs.
            eligible = sorted(other for (b, other) in order
                              if b == block
                              and any(name not in used for name in order[(b, other)]))
            assert eligible, f"{block}: no unused prompts left for a substitute"
            source = eligible[rng.randrange(len(eligible))]
            sentence = next(name for name in order[(block, source)]
                            if name not in used)
            used.add(sentence)
            drawn.append(make_item(pools_by_block, block, source, sentence,
                                   pair, len(matched[(block, pair)]) + offset,
                                   "substitute"))
        substitutes[(block, pair)] = drawn
    return substitutes


def draw(pools_by_block, seed):
    """The full item set: matched where possible, substituted where not.

    Returns (items, manifest_rows). Every cell ends at exactly N_PER_CELL, so
    there are 12 * N_PER_CELL items per (comparison, env) block.
    """
    rng = random.Random(seed)
    used = set()
    order = shuffled_pools(pools_by_block, rng)
    matched, free_at_draw = draw_matched(pools_by_block, order, used)
    substitutes = draw_substitutes(pools_by_block, order, used, matched, rng)

    items = []
    manifest = []
    for block, pair in sorted(matched):
        baseline, env = block
        cell = matched[(block, pair)] + substitutes[(block, pair)]
        assert len(cell) == config.N_PER_CELL, (block, pair, len(cell))
        items.extend(cell)
        manifest.append({
            "comparison": baseline,
            "env": env,
            "cat_a": pair[0],
            "cat_b": pair[1],
            "cat_a_name": config.CATEGORIES[pair[0]],
            "cat_b_name": config.CATEGORIES[pair[1]],
            # Candidates that existed at all, vs. candidates this cell could
            # still have when its turn came (an earlier, scarcer cell may have
            # taken some -- see UNIQUE_PROMPTS).
            "available": len(pools_by_block[block][pair]),
            "available_unused": free_at_draw[(block, pair)],
            "matched": len(matched[(block, pair)]),
            "substituted": len(substitutes[(block, pair)]),
            "taken": len(cell),
            "substitute_pairs": [[item["cat_a"], item["cat_b"]]
                                 for item in substitutes[(block, pair)]],
        })

    items.sort(key=lambda item: item["item_id"])
    manifest.sort(key=lambda row: (row["comparison"], row["env"],
                                   row["cat_a"], row["cat_b"]))
    return items, manifest


def items_digest(items):
    """sha256 over the item set, so a session can pin the exact text it showed
    without the database having to store any of it."""
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    pools_by_block = {(baseline, env): cell_pools(env, baseline)
                      for _, baseline in config.COMPARISONS
                      for env in config.ENVS}
    items, manifest = draw(pools_by_block, config.SAMPLE_SEED)

    expected = (12 * config.N_PER_CELL * len(config.COMPARISONS)
                * len(config.ENVS))
    assert len(items) == expected, f"{len(items)} items, want {expected}"
    assert len({item["item_id"] for item in items}) == len(items), "duplicate item_id"
    if config.UNIQUE_PROMPTS:
        assert len({item["prompt"] for item in items}) == len(items), "prompt reused"
    for item in items:
        assert item["cat_a"] != item["cat_b"], f"{item['item_id']}: same category"
        assert item["response_a"] != item["response_b"], f"{item['item_id']}: identical text"
        matched = (item["cat_a"], item["cat_b"]) == (item["target_cat_a"],
                                                    item["target_cat_b"])
        assert matched == (item["origin"] == "matched"), (
            f"{item['item_id']}: origin disagrees with its categories")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    bundle = {"seed": config.SAMPLE_SEED, "n_per_cell": config.N_PER_CELL,
              "unique_prompts": config.UNIQUE_PROMPTS,
              "substitution_rule": config.SUBSTITUTION_RULE,
              "arms": config.ARMS, "envs": config.ENVS,
              "categories": {str(k): v for k, v in config.CATEGORIES.items()},
              "items": items}
    bundle["items_sha256"] = items_digest(items)
    with open(config.ITEMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=1)
    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump({"items_sha256": bundle["items_sha256"],
                   "substitution_rule": config.SUBSTITUTION_RULE,
                   "cells": manifest}, handle, ensure_ascii=False, indent=1)

    subbed = [row for row in manifest if row["substituted"]]
    total = sum(row["substituted"] for row in manifest)
    print(f"{len(items)} items ({12 * config.N_PER_CELL} per (comparison, env) "
          f"block), sha256 {bundle['items_sha256'][:12]}")
    print(f"wrote {config.ITEMS_PATH}")
    print(f"wrote {config.MANIFEST_PATH}")

    print("\nmatched items per block (of "
          f"{12 * config.N_PER_CELL}):")
    for _, baseline in config.COMPARISONS:
        for env in config.ENVS:
            rows = [row for row in manifest
                    if row["comparison"] == baseline and row["env"] == env]
            print(f"  ours vs {baseline:<14s} {env:<4s} "
                  f"matched={sum(r['matched'] for r in rows):>3d} "
                  f"sub={sum(r['substituted'] for r in rows):>3d}")

    print(f"\n{total} of {len(items)} items are SUBSTITUTES, filling "
          f"{len(subbed)} of {len(manifest)} cells:")
    for row in subbed:
        cost = "" if row["available_unused"] == row["available"] else (
            f", {row['available'] - row['available_unused']} to a scarcer cell")
        pairs = " ".join(f"{a}/{b}" for a, b in row["substitute_pairs"])
        print(f"  ours vs {row['comparison']:14s} {row['env']:<4s} "
              f"cell {row['cat_a']}/{row['cat_b']}"
              f"  matched={row['matched']} + sub={row['substituted']}"
              f"  (pool {row['available']}{cost})  drawn from: {pairs}")


if __name__ == "__main__":
    main()
