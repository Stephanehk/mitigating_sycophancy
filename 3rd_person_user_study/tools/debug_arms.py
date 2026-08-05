"""The item -> arm mapping behind the `?labels=1` overlay.

static/items.json is published with the responses keyed `a`/`b` and no arm names
(see build_static.py), and app.js resolves nothing: a screen side becomes a model
inside `pref_record`, from the session's own draw. So the mapping has to come
from somewhere else, and this is the one place it is built -- out of the FULL
data/items.json, which keeps the arm names the published item set drops.

Two consumers, one mapping, so the label a page shows cannot disagree with the
arm the database records:

  - tools/build_static.py writes it to static/__debug_arms.json at build time,
    which is what `?labels=1` fetches on the deployed site;
  - tools/local_server.py serves it at the same path, so the local stub and the
    deployed page run the identical overlay.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY)

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
    digest = config.items_digest(bundle["items"])
    assert digest == bundle["items_sha256"], "data/items.json digest is stale"
    return bundle


