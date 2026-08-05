"""Serve static/ locally with an IN-MEMORY stand-in for the four Supabase RPCs.

FOR LOOKING AT THE PAGE. Nothing it records survives the process, and it is not
a staging environment -- do not point a rater at it.

WHY IT EXISTS. The page has no offline mode and should not get one: a fallback
that let it run without a backend would be a second code path, and the one that
matters would be the one nobody looked at. So instead of loosening the page, this
serves the SAME unmodified static/ files against a stub that implements the RPC
contract, and hands the page a config.json pointing at itself. What you see is
the real client.

The stub deliberately mirrors the SQL rather than simplifying it -- the digest
check, the required Prolific ID, the per-session order and side draw,
side-to-arm resolution, upsert on re-answer, and next-unanswered resume. A
divergence here would mean the page looked fine locally and broke against
Postgres.

    python 3rd_person_user_study/tools/local_server.py [--port 8000] [--label-arms]

Then open http://localhost:8000/. Over SSH:
    ssh -L 8000:localhost:8000 <host>

--label-arms is a DEBUG view: it names the arm behind each response above the two
panels. It is off by default and changes nothing in static/ -- see
tools/debug_labels.js for how, and why it is bolted on from outside the page
rather than added to it.
"""

import argparse
import http.server
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.dirname(_HERE)
sys.path.insert(0, _STUDY)

import build_items
import config

#: session_token -> {"session_id", "assignments", "completed"}
SESSIONS = {}
#: (session_id, item_id) -> the recorded row, as pref_judgments would hold it
JUDGMENTS = {}

#: --label-arms. OFF by default, and the default has to stay the blind page: the
#: whole value of this stub is that what you look at locally is what a rater gets.
LABEL_ARMS = False


def load_items():
    with open(config.ITEMS_PATH, encoding="utf-8") as handle:
        bundle = json.load(handle)
    digest = build_items.items_digest(bundle["items"])
    assert digest == bundle["items_sha256"], "data/items.json digest is stale"
    return {item["item_id"]: item for item in bundle["items"]}, digest


ITEMS, DIGEST = load_items()


def start_session(args):
    """pref_start_session. Refuses a page holding a different item build, refuses
    a blank Prolific ID, and draws the order and the per-item side exactly as the
    SQL does."""
    if args.get("p_items_sha256") != DIGEST:
        return 400, {"message": f"item set mismatch: page has "
                                f"{args.get('p_items_sha256')}, stub expects {DIGEST}"}
    prolific_pid = (args.get("p_prolific_pid") or "").strip()[:200]
    if not prolific_pid:
        return 400, {"message": "prolific id is required"}
    session_id = "local%04d" % (len(SESSIONS) + 1)
    token = "tok-" + session_id
    order = sorted(ITEMS)
    random.shuffle(order)
    assignments = [{"item_id": item_id, "flipped": random.random() < 0.5,
                    "position": position}
                   for position, item_id in enumerate(order)]
    SESSIONS[token] = {"session_id": session_id, "assignments": assignments,
                       "prolific_pid": prolific_pid, "completed": False}
    print(f"[start] {session_id}: {len(assignments)} items, pid {prolific_pid!r}")
    return 200, {"session_id": session_id, "session_token": token,
                 "items": [{"item_id": a["item_id"], "flipped": a["flipped"]}
                           for a in assignments]}


def session_for_token(args):
    return SESSIONS.get(args.get("p_token"))


def record(args):
    """pref_record. The browser sends a SIDE; the arms are resolved here, from
    this session's own draw -- the same division of trust as the RPC."""
    session = session_for_token(args)
    if session is None:
        return 401, {"message": "unknown session", "code": "P0002"}
    if session["completed"]:
        return 400, {"message": "session already complete"}
    if args.get("p_choice") not in config.CHOICES:
        return 400, {"message": f"bad choice: {args.get('p_choice')}"}

    item_id = args.get("p_item_id")
    assignment = next((a for a in session["assignments"] if a["item_id"] == item_id),
                      None)
    if assignment is None:
        return 400, {"message": f"item {item_id} not assigned to this session"}

    item = ITEMS[item_id]
    left, right = ((item["model_b"], item["model_a"]) if assignment["flipped"]
                   else (item["model_a"], item["model_b"]))
    choice = args["p_choice"]
    preferred = {"left": left, "right": right}.get(choice)
    JUDGMENTS[(session["session_id"], item_id)] = {
        "choice": choice, "left_model": left, "right_model": right,
        "preferred_model": preferred, "ms_elapsed": args.get("p_ms_elapsed")}
    answered = sum(1 for key in JUDGMENTS if key[0] == session["session_id"])
    print(f"[{answered:>3}/{len(session['assignments'])}] {item_id:<24} "
          f"{choice:<10} left={left:<14} right={right:<14} -> {preferred}")
    return 200, {"answered": answered}


def progress(args):
    session = session_for_token(args)
    if session is None:
        return 401, {"message": "unknown session", "code": "P0002"}
    total = len(session["assignments"])
    answered = sum(1 for key in JUDGMENTS if key[0] == session["session_id"])
    unanswered = [a["position"] for a in session["assignments"]
                  if (session["session_id"], a["item_id"]) not in JUDGMENTS]
    return 200, {"answered": answered, "total": total,
                 "next_position": min(unanswered) if unanswered else total}


def complete(args):
    session = session_for_token(args)
    if session is None:
        return 401, {"message": "unknown session", "code": "P0002"}
    session["completed"] = True
    answered = sum(1 for key in JUDGMENTS if key[0] == session["session_id"])
    print(f"[done] {session['session_id']}: {answered} judgments")
    return 200, {"answered": answered}


RPCS = {"pref_start_session": start_session, "pref_record": record,
        "pref_progress": progress, "pref_complete": complete}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *handler_args, **handler_kwargs):
        super().__init__(*handler_args, directory=config.STATIC_DIR,
                         **handler_kwargs)

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        # Overridden so the committed production config.json is not the thing the
        # local page reads -- it would send every RPC to the real project.
        #
        # Built from the request's OWN Host header rather than hardcoding
        # localhost: whatever name the page was fetched under is the one the RPCs
        # have to go back to, or the browser treats them as cross-origin and CORS
        # blocks every one. Reaching the same server as 127.0.0.1 used to be
        # enough to break it.
        if path == "/config.json":
            host = self.headers.get("Host") or f"localhost:{self.server.server_port}"
            self.send_json(200, {"supabase_url": f"http://{host}",
                                 "supabase_anon_key": "local-stub"})
            return

        # The two debug routes 404 unless --label-arms, so the arm names are not
        # reachable at all on the default (blind) page.
        if path == "/__debug_arms.json":
            if not LABEL_ARMS:
                self.send_json(404, {"message": "run with --label-arms"})
                return
            self.send_json(200, {item_id: {"a": item["model_a"],
                                           "b": item["model_b"]}
                                 for item_id, item in ITEMS.items()})
            return

        if path == "/__debug_labels.js":
            if not LABEL_ARMS:
                self.send_error(404)
                return
            self.send_asset(os.path.join(_HERE, "debug_labels.js"),
                            "application/javascript")
            return

        if LABEL_ARMS and path in ("/", "/index.html"):
            self.send_index_with_labels()
            return

        super().do_GET()

    def send_asset(self, path, content_type):
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_index_with_labels(self):
        """static/index.html with the debug script appended.

        Injected here rather than added to static/: the file on disk stays exactly
        the one that gets deployed, so --label-arms cannot leak into a build.
        """
        path = os.path.join(config.STATIC_DIR, "index.html")
        with open(path, encoding="utf-8") as handle:
            html = handle.read()
        marker = "</body>"
        assert marker in html, f"no {marker} in index.html to inject before"
        html = html.replace(
            marker, '  <script src="__debug_labels.js"></script>\n' + marker, 1)
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        prefix = "/rest/v1/rpc/"
        name = self.path[len(prefix):] if self.path.startswith(prefix) else None
        if name not in RPCS:
            self.send_json(404, {"message": f"no such rpc: {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        args = json.loads(self.rfile.read(length) or b"{}")
        status, payload = RPCS[name](args)
        self.send_json(status, payload)

    def log_message(self, *log_args):
        return  # the RPC prints above are the useful log


def main():
    global LABEL_ARMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--label-arms", action="store_true",
                        help="DEBUG: name the arm above each response. Not the "
                             "page a rater sees -- never use it to judge items.")
    parsed = parser.parse_args()
    LABEL_ARMS = parsed.label_arms
    assert os.path.exists(os.path.join(config.STATIC_DIR, "items.json")), (
        "run tools/build_static.py first")
    print(f"{len(ITEMS)} items, digest {DIGEST[:12]}")
    if LABEL_ARMS:
        print("--label-arms: ARM NAMES ARE ON SCREEN. This is not the page a "
              "rater sees; anything you judge here is judged unblinded.")
    print(f"IN-MEMORY STUB -- nothing is saved. http://localhost:{parsed.port}/")
    http.server.ThreadingHTTPServer(("127.0.0.1", parsed.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
