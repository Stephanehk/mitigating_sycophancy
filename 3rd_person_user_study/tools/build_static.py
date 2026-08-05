"""Emit the static assets the rating page fetches at load.

  static/items.json   the item set, STRIPPED of every label. The page gets
                      item_id, the prompt and the two responses keyed `a`/`b`
                      -- and nothing about which arm produced which, or what the
                      endorsement judge called it. `data/items.json` keeps the
                      full record; only this reduced view is published.
  static/config.json  the Supabase project URL and anon key, from the
                      environment so real values are never committed.

  static/consent.json the consent form, IMPORTED FROM study/content.py rather
                      than copied, so the two studies cannot show participants
                      differently-worded consent.

NOT PUBLISHED ANY MORE: the item -> arm mapping and the `?labels=1` overlay.
They were emitted here so the labelled view worked on the deployed site with
nothing pasted into a console. With the study going live that trade is off --
the mapping is the answer key, and a rater who found it would know which arm
wrote which response. tools/local_server.py still serves both, so `?labels=1`
works against a local checkout; nothing reachable from the deployed URL does.

All are read with a relative path from index.html, so the directory can be
opened directly during development and served as a site root in production --
the same arrangement as study/static.

Run:
    python 3rd_person_user_study/tools/build_static.py
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY)

import config

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
                   "supabase_anon_key": anon,
                   "min_item_seconds": config.MIN_ITEM_SECONDS,
                   "min_reasoning_chars": config.MIN_REASONING_CHARS,
                   "completion_code": config.completion_code()}, handle, indent=2)
    return path


def consent():
    """The consent screen's text, read out of study/content.py.

    IMPORTED, NOT COPIED. Both studies recruit the same Prolific pool under the
    same protocol, and a consent form that had drifted between them is the kind
    of difference nobody notices until it matters. study/content.py is the one
    place it is authored.

    THE WORDING DOES NOT ALL FIT THIS STUDY, and that is deliberately not
    patched up here -- it is IRB-facing text, so the mismatches are surfaced for
    sign-off rather than silently edited. See MISMATCHES below and the README.
    """
    sys.path.insert(0, os.path.join(config.REPO, "study"))
    import content
    assert content.CONSENT_CHOICES, "study/content.py has no consent choices"
    return {"title": content.CONSENT_TITLE,
            "body": content.CONSENT_BODY,
            "choices": content.CONSENT_CHOICES}


#: Spans of the imported consent text that describe study/ and not this study.
#: Asserted present by the tests so they cannot be forgotten: if one is edited
#: in study/content.py the test fails and this list has to be revisited.
CONSENT_MISMATCHES = [
    # This study shows pre-generated responses; nobody talks to a chatbot.
    "talking with a chatbot about an interpersonal conflict you experienced",
    # ~5 minutes of rating, not twenty.
    "approximately twenty minutes",
    # Payment is per-study and has not been set for this one.
    "$2 USD",
]


def write_consent():
    path = os.path.join(config.STATIC_DIR, "consent.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(consent(), handle, ensure_ascii=False)
    return path


def main():
    os.makedirs(config.STATIC_DIR, exist_ok=True)
    path, size = write_items()
    print(f"wrote {path} ({size / 1024:.0f} KB)")
    print(f"wrote {write_config()}")
    print(f"wrote {write_consent()}")


if __name__ == "__main__":
    main()
