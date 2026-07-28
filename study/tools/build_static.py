"""Emit static/content.json — every string and question the SPA renders.

The bundle is 18KB and identical for every participant: content_bundle() takes
no arguments. Serving it from an endpoint meant re-sending it on every session
start for no reason. Amplify can serve it as a static asset instead, cached by
the CDN, which is one fewer thing the browser waits on before the consent page
appears.

Generated rather than hand-maintained so content.py, questionnaires.py and
scenarios.py stay the single source of truth for the instrument. Run as part of
the Amplify build (see amplify.yml), and committed too so the SPA works when
opened directly during development.

    python tools/build_static.py

Also emits static/config.json — the Supabase project URL and anon key the
client needs to reach the RPCs. Both are public by design: the anon key is
visible in page source on every Supabase app, and it can do nothing on its own
because RLS blocks every table and each RPC authenticates its own caller. The
values come from the environment so the file is never committed with real ones.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY)

import config
import views

_STATIC = os.path.join(_STUDY, "static")


def write_content():
    """The instrument, as one JSON file."""
    bundle = views.content_bundle()
    path = os.path.join(_STATIC, "content.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, separators=(",", ":"))
    return path, os.path.getsize(path)


def write_config():
    """Where the client should send its RPCs.

    Fails loudly on a missing anon key rather than emitting a file that would
    500 in the browser at run time — an Amplify build with no credentials
    configured should not produce a deployable-looking artifact.
    """
    anon = os.environ.get("STUDY_SUPABASE_ANON_KEY", "").strip()
    url = config.SUPABASE_URL
    assert url, "STUDY_SUPABASE_URL is not set"
    assert anon, (
        "STUDY_SUPABASE_ANON_KEY is not set. This is the PUBLIC anon/publishable "
        "key (sb_publishable_... or the anon JWT), NOT the service_role key — "
        "that one must never be built into the page.")
    assert "secret" not in anon and "service_role" not in anon, (
        "STUDY_SUPABASE_ANON_KEY looks like a SECRET key (%r). Use the anon / "
        "publishable key; the secret key would be readable in page source."
        % (anon[:20],))

    payload = {
        "supabase_url": url.rstrip("/"),
        "supabase_anon_key": anon,
        "redirect_delay_seconds": config.REDIRECT_DELAY_SECONDS,
    }
    path = os.path.join(_STATIC, "config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path, os.path.getsize(path)


def main():
    content_path, content_size = write_content()
    print("wrote %s (%.1f KB)" % (content_path, content_size / 1024))
    config_path, config_size = write_config()
    print("wrote %s (%d bytes)" % (config_path, config_size))


if __name__ == "__main__":
    main()
