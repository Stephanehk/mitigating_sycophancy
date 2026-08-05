"""Emit the static assets the rating page fetches at load.

  static/items.json   the item set, STRIPPED of every label. The page gets
                      item_id, the prompt and the two responses keyed `a`/`b`
                      -- and nothing about which arm produced which, or what the
                      endorsement judge called it. `data/items.json` keeps the
                      full record; only this reduced view is published.
  static/config.json  the Supabase project URL and anon key, from the
                      environment so real values are never committed.

  static/__debug_arms.json  the item -> arm mapping, and
  static/debug_labels.js    the overlay that renders it.
                      Published, but fetched ONLY by `?labels=1` (see app.js).
                      A rater on the plain URL requests neither, so the page
                      they load is unchanged by their existence.

All are read with a relative path from index.html, so the directory can be
opened directly during development and served as a site root in production --
the same arrangement as study/static.

WHY THE MAPPING IS PUBLISHED AT ALL. `?labels=1` has to work on the deployed
site, with no local checkout and nothing pasted into a console. That means the
names have to be fetchable, so the item set is no longer secret from a
determined visitor -- only from the rating page itself, which never asks for
them. The protection that remains is that a rater has no reason to, and the
data-side protection is unchanged: `pref_record` resolves a screen side to a
model in the database from the session's own draw, so what a rater could learn
cannot alter what their judgment is recorded as.

Run:
    python 3rd_person_user_study/tools/build_static.py
"""

import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY)

import config

sys.path.insert(0, _HERE)

import debug_arms

# Exactly the keys the page may see. Anything not listed -- comparison, env,
# model_a, model_b, cat_a, cat_b -- stays server-side.
CLIENT_KEYS = ["item_id", "prompt", "response_a", "response_b"]


def client_bundle():
    """The published view of the item set, with its digest.

    The digest is over the FULL item set (as computed by build_items.py), not
    over this reduced view, because it is what pref_start_session compares
    against -- the page and the database must be talking about the same build.
    """
    assert os.path.exists(config.ITEMS_PATH), (
        f"missing {config.ITEMS_PATH} -- run build_items.py first")
    with open(config.ITEMS_PATH, encoding="utf-8") as handle:
        bundle = json.load(handle)

    items = []
    for item in bundle["items"]:
        items.append({key: item[key] for key in CLIENT_KEYS})
    leaked = set(items[0]) - set(CLIENT_KEYS) if items else set()
    assert not leaked, f"client bundle would publish {sorted(leaked)}"
    return {"items_sha256": bundle["items_sha256"], "items": items}


def write_items():
    payload = client_bundle()
    path = os.path.join(config.STATIC_DIR, "items.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return path, os.path.getsize(path)


def write_config():
    """Where the page should send its RPCs. Fails rather than emitting a file
    that would 200 with an unusable key -- a page that loads and then silently
    cannot record anything is the worst failure mode here.

    The file is gitignored (see ../.gitignore) and generated at deploy time, so
    this is the only place the credential enters the artifact.
    """
    url = config.SUPABASE_URL.strip()
    anon = config.SUPABASE_ANON_KEY.strip()
    assert url, "PREF_SUPABASE_URL is not set"
    assert anon, (
        "PREF_SUPABASE_ANON_KEY is not set. This is the PUBLIC anon/publishable "
        "key (sb_publishable_... or the anon JWT), NOT the service_role key -- "
        "that one must never be built into the page.")
    # The same guard study/tools/build_static.py carries. Anything written here is
    # readable in page source, so a service_role key handed to the build by
    # mistake has to fail the build rather than ship.
    assert "secret" not in anon and "service_role" not in anon, (
        "PREF_SUPABASE_ANON_KEY looks like a SECRET key (%r). Use the anon / "
        "publishable key; the secret key would be readable in page source."
        % (anon[:20],))
    path = os.path.join(config.STATIC_DIR, "config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"supabase_url": url.rstrip("/"),
                   "supabase_anon_key": anon}, handle, indent=2)
    return path


def write_debug_labels():
    """The `?labels=1` pair: the arm mapping, and the overlay that renders it.

    Both are generated (and gitignored) rather than committed into static/, so
    tools/debug_labels.js stays the single copy of the labelling code and cannot
    drift from the one tools/local_server.py serves.

    Assumes the FULL data/items.json -- debug_arms.arms_map asserts on a bundle
    with no arm names, which is what the published items.json is.
    """
    bundle = debug_arms.load_items()
    arms_path = os.path.join(config.STATIC_DIR, "__debug_arms.json")
    with open(arms_path, "w", encoding="utf-8") as handle:
        json.dump(debug_arms.arms_map(bundle["items"]), handle,
                  ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    labels_path = os.path.join(config.STATIC_DIR, "debug_labels.js")
    shutil.copyfile(debug_arms.LABELS_JS, labels_path)
    return arms_path, labels_path


def main():
    os.makedirs(config.STATIC_DIR, exist_ok=True)
    path, size = write_items()
    print(f"wrote {path} ({size / 1024:.0f} KB)")
    print(f"wrote {write_config()}")
    for path in write_debug_labels():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
