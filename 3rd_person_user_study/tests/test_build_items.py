"""Tests for the item-set construction.

The load-time assertions in build_items.py cover the input contract (files
exist, labels parse, the arms cover the same prompts). These tests cover the
things an assertion cannot: that the committed artifact actually matches the
CSVs it claims to come from, that the sampling is reproducible, and that the
design's own invariants hold item by item.

The central one is test_every_item_matches_its_source_csvs. It re-derives each
item from the scored files independently of the sampler and checks the response
text and the endorsement label on BOTH sides. Without it, a mislabelled or
mis-joined item would look perfectly well-formed.
"""

import os

import pytest

import build_items
import config


@pytest.fixture(scope="module")
def arms():
    """(env, arm) -> {sentence: (response, label)}, read straight from the
    scored CSVs. The reference the items are checked against."""
    table = {}
    for env in config.ENVS:
        for arm in config.ARMS:
            frame = build_items.load_arm(env, arm)
            table[(env, arm)] = {row.sentence: (row.response, row.label)
                                 for row in frame.itertuples()}
    return table


def test_every_item_matches_its_source_csvs(bundle, arms):
    for item in bundle["items"]:
        key_a = (item["env"], item["model_a"])
        key_b = (item["env"], item["model_b"])
        assert item["prompt"] in arms[key_a], f"{item['item_id']}: prompt not in {key_a}"
        assert item["prompt"] in arms[key_b], f"{item['item_id']}: prompt not in {key_b}"

        response_a, label_a = arms[key_a][item["prompt"]]
        response_b, label_b = arms[key_b][item["prompt"]]
        assert item["response_a"] == response_a, f"{item['item_id']}: response_a drifted"
        assert item["response_b"] == response_b, f"{item['item_id']}: response_b drifted"
        assert item["cat_a"] == label_a, f"{item['item_id']}: cat_a is not the judge label"
        assert item["cat_b"] == label_b, f"{item['item_id']}: cat_b is not the judge label"


def test_model_a_is_always_ours(bundle):
    """The whole design is "ours in category a, the baseline in category b".
    If either side could be the baseline, every cell would mean two things."""
    for item in bundle["items"]:
        assert item["model_a"] == "ours", item["item_id"]
        assert item["model_b"] == item["comparison"], item["item_id"]
        assert item["model_b"] in config.ARMS, item["item_id"]
        assert item["model_b"] != "ours", item["item_id"]


def test_categories_within_a_pair_always_differ(bundle):
    for item in bundle["items"]:
        assert item["cat_a"] != item["cat_b"], item["item_id"]
        assert item["cat_a"] in config.CATEGORIES
        assert item["cat_b"] in config.CATEGORIES


def test_no_prompt_is_used_twice(bundle):
    """UNIQUE_PROMPTS: no rater sees the same scenario -- or the same `ours`
    response -- twice, including across the two comparisons."""
    assert config.UNIQUE_PROMPTS, "test encodes the UNIQUE_PROMPTS design"
    prompts = [item["prompt"] for item in bundle["items"]]
    assert len(set(prompts)) == len(prompts)


def test_item_ids_are_unique_and_name_the_cell_they_fill(bundle):
    """The id names the TARGET cell, not the drawn one: that is what keeps
    "5 per cell" readable off the ids of a set containing substitutes."""
    seen = set()
    for item in bundle["items"]:
        assert item["item_id"] not in seen
        seen.add(item["item_id"])
        assert item["item_id"].startswith(
            f"{item['comparison']}-{item['env']}-"
            f"{item['target_cat_a']}{item['target_cat_b']}-")


#: 4 categories permute 2 = both (a, b) and (b, a).
ORDERED_PAIRS = 12

#: One cell per (comparison, env, ordered pair). DERIVED, not written down: with
#: PAS only this is 24, and it was 48 while OEQ was in the draw. A hardcoded
#: number here would have to be edited in step with config.ENVS, and the version
#: that failed to be edited would be the one asserting the design is intact.
CELLS = len(config.COMPARISONS) * len(config.ENVS) * ORDERED_PAIRS


def test_every_cell_holds_exactly_n_per_cell_items(bundle, manifest):
    """The point of the substitution rule: 12 ordered pairs x N per
    (comparison, env) BLOCK."""
    counted = {}
    for item in bundle["items"]:
        key = (item["comparison"], item["env"],
               item["target_cat_a"], item["target_cat_b"])
        counted[key] = counted.get(key, 0) + 1
    assert set(counted.values()) == {config.N_PER_CELL}
    assert len(counted) == CELLS
    for _, baseline in config.COMPARISONS:
        for env in config.ENVS:
            per = sum(count for key, count in counted.items()
                      if key[0] == baseline and key[1] == env)
            assert per == ORDERED_PAIRS * config.N_PER_CELL, (baseline, env)
    assert len(bundle["items"]) == CELLS * config.N_PER_CELL


def test_origin_is_exactly_whether_the_pair_matches_the_cell(bundle):
    """`origin` is redundant with the two label pairs on purpose -- and a
    substitute that recorded itself as matched would be invisible in analysis, so
    the redundancy is checked rather than trusted."""
    for item in bundle["items"]:
        same_cell = (item["cat_a"], item["cat_b"]) == (item["target_cat_a"],
                                                       item["target_cat_b"])
        assert item["origin"] in ("matched", "substitute"), item["item_id"]
        assert same_cell == (item["origin"] == "matched"), item["item_id"]


def test_substitutes_are_still_well_formed_pairs(bundle):
    """A substitute is a real pair from a real cell -- just a different one. Its
    two categories still differ, which is what the design asks for."""
    subs = [item for item in bundle["items"] if item["origin"] == "substitute"]
    assert subs, "no substitutes: the fixture predates the substitution rule"
    for item in subs:
        assert item["cat_a"] != item["cat_b"], item["item_id"]
        assert item["target_cat_a"] != item["target_cat_b"], item["item_id"]
        assert (item["cat_a"], item["cat_b"]) != (item["target_cat_a"],
                                                  item["target_cat_b"])


def test_a_substitute_never_crosses_an_env_or_a_comparison(bundle, arms):
    """THE REASON THE DESIGN IS BLOCKED BY ENV.

    `ours` and `neutral_mixed` are trained per env, so an item's two responses only
    come from one adapter pair if the item stays inside one env. A substitute drawn
    from a different env would put a PAS-adapter response in an OEQ cell and the
    pooling this blocking exists to prevent would be back -- invisibly, since only
    `env` records it.

    Checked against the SOURCE CSVs, not against the item's own `env` field: the
    field agreeing with itself would prove nothing. An item's prompt has to appear
    in the scored files of both its arms FOR ITS OWN ENV, which is only true if the
    responses really came from that env's run.
    """
    subs = [item for item in bundle["items"] if item["origin"] == "substitute"]
    assert subs, "no substitutes: nothing to check"
    for item in subs:
        assert item["env"] in config.ENVS, item["item_id"]
        for arm in ("model_a", "model_b"):
            frame = arms[(item["env"], item[arm])]
            assert item["prompt"] in frame, (
                f"{item['item_id']}: prompt is not in the {item['env']} "
                f"{item[arm]} run, so the response is from another env")
        # And it is NOT in the other env's run -- the two prompt pools are
        # disjoint, so a leak would show up here.
        for other in [env for env in config.ENVS if env != item["env"]]:
            assert item["prompt"] not in arms[(other, item["model_a"])], (
                f"{item['item_id']}: prompt appears in {other} too")


def test_a_cell_is_only_substituted_into_when_it_ran_dry(manifest):
    """Substitution must be a LAST resort: a cell may only be topped up after it
    has taken every matched prompt it could. Without this, the sampler could
    quietly prefer easy substitutes over harder real matches."""
    for row in manifest["cells"]:
        if row["substituted"]:
            assert row["matched"] == row["available_unused"], row
            assert row["matched"] < config.N_PER_CELL, row


def test_the_substitution_rule_is_recorded_with_the_data(bundle, manifest):
    """It is a methods decision, so it travels with the artifact rather than
    living only in the code that produced it."""
    assert bundle["substitution_rule"] == config.SUBSTITUTION_RULE
    assert manifest["substitution_rule"] == config.SUBSTITUTION_RULE
    assert "uniformly" in config.SUBSTITUTION_RULE


def test_all_cells_are_accounted_for(manifest):
    """12 ordered category pairs x every comparison x every env. A cell that
    filled to 0 must still appear -- a missing row would read as "not part of the
    design" rather than "empty"."""
    cells = {(row["comparison"], row["env"], row["cat_a"], row["cat_b"])
             for row in manifest["cells"]}
    expected = {(baseline, env, a, b)
                for _, baseline in config.COMPARISONS
                for env in config.ENVS
                for a in config.CATEGORIES for b in config.CATEGORIES if a != b}
    assert cells == expected
    assert len(cells) == CELLS


def test_manifest_counts_agree_with_the_item_set(bundle, manifest):
    matched = {}
    subbed = {}
    for item in bundle["items"]:
        key = (item["comparison"], item["env"],
               item["target_cat_a"], item["target_cat_b"])
        bucket = matched if item["origin"] == "matched" else subbed
        bucket[key] = bucket.get(key, 0) + 1
    for row in manifest["cells"]:
        key = (row["comparison"], row["env"], row["cat_a"], row["cat_b"])
        assert matched.get(key, 0) == row["matched"], key
        assert subbed.get(key, 0) == row["substituted"], key
        assert row["matched"] + row["substituted"] == row["taken"], key
    assert sum(row["taken"] for row in manifest["cells"]) == len(bundle["items"])


def test_matched_counts_are_capped_by_the_data(manifest):
    """The matched half is still exactly min(target, prompts still unused) -- the
    substitution rule tops a cell up, it does not license inventing a match."""
    for row in manifest["cells"]:
        assert row["taken"] == config.N_PER_CELL, row
        assert row["available_unused"] <= row["available"], row
        assert row["matched"] == min(config.N_PER_CELL, row["available_unused"]), row


def test_the_manifest_names_the_pair_every_substitute_came_from(bundle, manifest):
    """`substitute_pairs` is the per-cell provenance record, so the write-up can
    say what each short cell was actually filled with."""
    for row in manifest["cells"]:
        assert len(row["substitute_pairs"]) == row["substituted"], row
        for pair in row["substitute_pairs"]:
            assert pair[0] != pair[1], row
            assert pair != [row["cat_a"], row["cat_b"]], row
    recorded = sorted(tuple(pair) for row in manifest["cells"]
                      for pair in row["substitute_pairs"])
    actual = sorted((item["cat_a"], item["cat_b"]) for item in bundle["items"]
                    if item["origin"] == "substitute")
    assert recorded == actual


def test_the_committed_item_set_is_what_the_seed_produces(bundle):
    """Reproducibility: rerunning the sampler must return the committed items
    exactly, so the artifact is derivable from the CSVs plus the seed alone."""
    pools = {(baseline, env): build_items.cell_pools(env, baseline)
             for _, baseline in config.COMPARISONS for env in config.ENVS}
    items, _ = build_items.draw(pools, config.SAMPLE_SEED)
    # main() does both; the set partition is part of the artifact, so a redraw
    # that skipped it would compare items missing their set_index.
    build_items.assign_sets(items, config.SAMPLE_SEED)
    assert items == bundle["items"]


def test_a_different_seed_draws_a_different_sample(bundle):
    """Guards against a sampler that ignores its seed -- which would still look
    deterministic and still pass the test above."""
    pools = {(baseline, env): build_items.cell_pools(env, baseline)
             for _, baseline in config.COMPARISONS for env in config.ENVS}
    items, _ = build_items.draw(pools, config.SAMPLE_SEED + 1)
    assert [item["prompt"] for item in items] != [i["prompt"] for i in bundle["items"]]


def test_the_digest_covers_the_response_text(bundle):
    """The sha256 the page and the database agree on has to change when the TEXT
    changes, not just when the item list does -- otherwise a redeploy could show
    different responses under the same pinned digest."""
    assert build_items.items_digest(bundle["items"]) == bundle["items_sha256"]
    tampered = [dict(item) for item in bundle["items"]]
    tampered[0]["response_a"] = tampered[0]["response_a"] + " "
    assert build_items.items_digest(tampered) != bundle["items_sha256"]


def test_responses_are_non_empty_and_distinct(bundle):
    for item in bundle["items"]:
        assert item["prompt"].strip(), item["item_id"]
        assert item["response_a"].strip(), item["item_id"]
        assert item["response_b"].strip(), item["item_id"]
        assert item["response_a"] != item["response_b"], item["item_id"]


def test_load_arm_rejects_an_unknown_condition():
    with pytest.raises(KeyError):
        build_items.load_arm("PAS", "not_an_arm")


def test_load_arm_rejects_an_unknown_env():
    with pytest.raises(KeyError):
        build_items.load_arm("AITA", "ours")


def test_every_arm_covers_the_full_heldout_split():
    """Assumption 3, exercised directly: all three arms answer the same 1000
    prompts per env, so a pair really is two responses to one prompt."""
    for env in config.ENVS:
        frames = {arm: build_items.load_arm(env, arm) for arm in config.ARMS}
        sentences = {arm: set(frame["sentence"]) for arm, frame in frames.items()}
        assert all(len(s) == config.HELDOUT_N for s in sentences.values())
        assert len(set(map(frozenset, sentences.values()))) == 1, (
            f"{env}: arms cover different prompt sets")


def test_the_two_envs_do_not_share_prompts():
    """Pooling PAS and OEQ is only sound if a prompt cannot be counted twice."""
    pas = set(build_items.load_arm("PAS", "ours")["sentence"])
    oeq = set(build_items.load_arm("OEQ", "ours")["sentence"])
    assert not (pas & oeq)


def test_items_come_from_both_envs(bundle):
    """Not an invariant of the design, but a change worth noticing: if pooling
    ever stopped contributing from one env, every cell would silently be
    single-env."""
    assert {item["env"] for item in bundle["items"]} == set(config.ENVS)


def test_the_sets_partition_the_item_set(bundle):
    """8 disjoint sets of SET_SIZE, every item in exactly one. A rater sees one
    set, so an item in two sets would be rated by two links at once and one in
    none would never be rated at all."""
    n_sets = len(bundle["items"]) // config.SET_SIZE
    assert bundle["n_sets"] == n_sets and bundle["set_size"] == config.SET_SIZE
    members = {}
    for item in bundle["items"]:
        members.setdefault(item["set_index"], []).append(item["item_id"])
    assert sorted(members) == list(range(1, n_sets + 1))
    assert {len(v) for v in members.values()} == {config.SET_SIZE}
    seen = [i for ids in members.values() for i in ids]
    assert len(seen) == len(set(seen)) == len(bundle["items"])


def test_every_set_carries_both_comparisons(bundle):
    """The partition is balanced by comparison, not a free shuffle: a set that
    came out all-`untrained` would contribute nothing to the `neutral_mixed`
    contrast, and the raters sent to that link would be wasted on half the
    design."""
    per_set = {}
    for item in bundle["items"]:
        per_set.setdefault(item["set_index"], []).append(item["comparison"])
    for index, comparisons in sorted(per_set.items()):
        for _, baseline in config.COMPARISONS:
            count = comparisons.count(baseline)
            assert count >= config.SET_SIZE // len(config.COMPARISONS) - 1, \
                f"set {index} has only {count} {baseline} items"
