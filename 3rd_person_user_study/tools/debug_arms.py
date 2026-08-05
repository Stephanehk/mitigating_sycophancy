"""The item -> arm mapping, in the two forms the debug label view needs it.

The rating page never receives this mapping: static/items.json is published with
the responses keyed `a`/`b` and no arm names (see build_static.py), and the
database resolves a screen side to a model itself. So anything that wants to SEE
which arm wrote which response has to bring the mapping with it, from the full
data/items.json that stays local.

Two consumers, one mapping:

  - tools/local_server.py --label-arms serves arms_map() at /__debug_arms.json;
  - this module run as a script prints a console snippet -- the mapping plus
    tools/debug_labels.js -- to paste into devtools on the DEPLOYED page:

        python 3rd_person_user_study/tools/debug_arms.py > /tmp/labels.js

    Pasting it labels the page in that one browser tab. Nothing is uploaded and
    nothing about the published site changes, which is the whole point: a rater
    loading the same URL still cannot tell the arms apart.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY)

import build_items
import config

LABELS_JS = os.path.join(_HERE, "debug_labels.js")


def arms_map(items):
    """{item_id: {"a": model_a, "b": model_b}} over full (unpublished) items.

    Takes the items themselves rather than a path so the caller that already
    holds the bundle does not read it twice. Assumes every item carries
    model_a/model_b -- true of data/items.json and false of the published
    static/items.json, which is exactly the confusion worth failing on.
    """
    items = list(items)
    assert items, "no items to map"
    for item in items:
        assert "model_a" in item and "model_b" in item, (
            f"{item.get('item_id')} has no arm names -- this looks like the "
            "published static/items.json, not data/items.json")
    return {item["item_id"]: {"a": item["model_a"], "b": item["model_b"]}
            for item in items}


def load_items():
    """The full item bundle, with its recorded digest checked against the items.

    A stale digest would mean the mapping describes a different build than the
    page is showing, and the labels would silently name the wrong arms.
    """
    assert os.path.exists(config.ITEMS_PATH), (
        f"missing {config.ITEMS_PATH} -- run build_items.py first")
    with open(config.ITEMS_PATH, encoding="utf-8") as handle:
        bundle = json.load(handle)
    digest = build_items.items_digest(bundle["items"])
    assert digest == bundle["items_sha256"], "data/items.json digest is stale"
    return bundle


def console_snippet(bundle):
    """debug_labels.js with the mapping prepended, ready to paste into devtools.

    Carries the digest as a comment so a snippet found later can be checked
    against the item set the page is serving.
    """
    with open(LABELS_JS, encoding="utf-8") as handle:
        labels = handle.read()
    mapping = json.dumps(arms_map(bundle["items"]), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return (f"/* Arm labels for item build {bundle['items_sha256']}.\n"
            " * Paste into the devtools console on the rating page. Local to\n"
            " * this tab: nothing is sent anywhere and the published page is\n"
            " * unchanged. Re-paste after a reload. */\n"
            f"window.__DEBUG_ARMS = {mapping};\n"
            f"{labels}")


def main():
    sys.stdout.write(console_snippet(load_items()))


if __name__ == "__main__":
    main()
