"""End-to-end tests for the screen flow: Prolific ID, three slides, then rating.

These drive the REAL static/ files in real headless Chromium against the REAL
stub from tools/local_server.py, booted in-process on an ephemeral port. Still
offline -- the stub is the whole backend here, and no Supabase call is made (see
tests/conftest.py).

WHY NOT SOURCE-LEVEL CHECKS. Every property below is one that a regex over app.js
would either miss or misreport:

  - that the keyboard numbers select the button they sit under, which is a fact
    about the RENDERED order and the KEYS map agreeing, not about either one;
  - that a keypress on an instruction slide records nothing, which is about when
    the handler is armed;
  - that the two response panels are the same height with the buttons on screen,
    which is the anti-cue property and only exists after layout;
  - that a blank Prolific ID cannot start a session, which is enforced in two
    places (the page and the stub) and has to hold end to end.
"""

import json
import os
import threading

import pytest

import config
import debug_arms


PID = "FLOWTESTPID"

#: The intro is three slides. Asserted rather than discovered, so adding one to
#: app.js without updating these tests fails instead of going untested.
SLIDE_COUNT = 3


@pytest.fixture(scope="module")
def server():
    """The stub from tools/local_server.py on an ephemeral port.

    Imported rather than reimplemented: a second stand-in written for the tests
    could pass while the one a human actually looks at was broken.
    """
    if not os.path.exists(os.path.join(config.STATIC_DIR, "items.json")):
        pytest.skip("static/items.json not built")
    import http.server

    import local_server

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), local_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", local_server
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        instance = driver.chromium.launch()
        yield instance
        instance.close()


def open_page(browser, base, width=1280, height=900, query=""):
    """A fresh page at the ID screen, with console errors collected onto it."""
    page = browser.new_page(viewport={"width": width, "height": height})
    page.errors = []
    page.on("pageerror", lambda exc: page.errors.append(str(exc)))
    page.on("console",
            lambda msg: msg.type == "error" and page.errors.append(msg.text))
    page.goto(base + "/" + query, wait_until="networkidle")
    return page


def enter_id(page, pid=PID):
    page.fill("input[type=text]", pid)
    page.click(".actions button")
    page.wait_for_timeout(400)


def pass_slides(page):
    """Click Continue through the intro, asserting the rating screen is not up
    early. Returns the slide texts in the order they were shown."""
    seen = []
    for _ in range(SLIDE_COUNT):
        seen.append(page.inner_text(".slide").strip())
        assert page.query_selector(".pair") is None, "rating screen shown during intro"
        page.click(".actions button")
        page.wait_for_timeout(300)
    return seen


# --- the Prolific ID -------------------------------------------------------

def test_a_blank_prolific_id_starts_no_session(server, browser):
    """The gate is the whole reason the session is drawn here and not at load: a
    session with no ID is neither payable nor joinable."""
    base, stub = server
    before = len(stub.SESSIONS)
    page = open_page(browser, base)
    page.click(".actions button")
    page.wait_for_timeout(400)

    assert page.query_selector("input[type=text]") is not None, "left the ID screen"
    assert page.query_selector(".error") is not None, "no message shown"
    assert len(stub.SESSIONS) == before, "a session was started without an ID"
    assert page.errors == []
    page.close()


def test_the_prolific_id_reaches_the_backend_trimmed(server, browser):
    base, stub = server
    page = open_page(browser, base)
    enter_id(page, "  " + PID + "  ")
    recorded = [s["prolific_pid"] for s in stub.SESSIONS.values()]
    assert PID in recorded, recorded
    assert page.errors == []
    page.close()


def test_the_prolific_id_is_prefilled_from_the_url(server, browser):
    """Prolific sends the rater in with ?PROLIFIC_PID=..., and a rater who has to
    copy it by hand is a rater who mistypes it."""
    base, _ = server
    page = open_page(browser, base, query="?PROLIFIC_PID=FROMURL42")
    assert page.input_value("input[type=text]") == "FROMURL42"
    page.close()


def test_digits_can_be_typed_into_the_id_field(server, browser):
    """The choice keys are bound on `document` and call preventDefault. Gating
    them on the stage is what keeps 1-4 from being swallowed here -- Prolific IDs
    contain digits."""
    base, _ = server
    page = open_page(browser, base)
    page.click("input[type=text]")
    page.keyboard.type("1234")
    assert page.input_value("input[type=text]") == "1234"
    page.close()


# --- the intro -------------------------------------------------------------

def test_the_intro_is_three_slides_then_the_rating_screen(server, browser):
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    seen = pass_slides(page)

    assert len(seen) == SLIDE_COUNT
    assert "two different responses" in seen[0]
    assert "your friend was writing the prompt" in seen[1]
    assert "Can" in seen[2] and "tell" in seen[2]
    assert page.query_selector(".pair") is not None, "rating screen did not arrive"
    assert page.errors == []
    page.close()


def test_a_choice_key_on_an_instruction_slide_records_nothing(server, browser):
    """The order is drawn BEFORE the slides are shown, so a handler gated on the
    item set being loaded would let a keypress here answer item 1."""
    base, stub = server
    page = open_page(browser, base)
    enter_id(page)
    before = len(stub.JUDGMENTS)
    for key in ["1", "2", "3", "4", "ArrowLeft", "ArrowRight"]:
        page.keyboard.press(key)
        page.wait_for_timeout(120)
    assert page.query_selector(".slide") is not None, "left the intro"
    assert page.query_selector(".pair") is None
    assert len(stub.JUDGMENTS) == before, "a judgment was recorded during the intro"
    page.close()


# --- the rating screen -----------------------------------------------------

def test_the_buttons_read_left_same_right_then_cant_tell(server, browser):
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    pass_slides(page)
    labels = [b.inner_text().strip()
              for b in page.query_selector_all(".choices button")]
    assert labels[:3] == ["Left", "Same", "Right"], labels
    assert "tell" in labels[3]

    # Can't tell sits in the middle column of the second row, under Same.
    boxes = page.evaluate("""() => [...document.querySelectorAll('.choices button')]
        .map(b => { const r = b.getBoundingClientRect();
                    return {mid: Math.round(r.left + r.width / 2),
                            top: Math.round(r.top)}; })""")
    same, cant = boxes[1], boxes[3]
    assert cant["top"] > same["top"], "Can't tell is not on a second row"
    assert abs(cant["mid"] - same["mid"]) <= 2, \
        f"Can't tell is not centred under Same: {cant['mid']} vs {same['mid']}"
    page.close()


def test_the_number_keys_select_the_button_they_sit_under(server, browser):
    """1-4 are not printed on the buttons, so the only thing they can agree with
    is the layout. When Same moved into the middle, `2` had to move with it."""
    base, stub = server
    page = open_page(browser, base)
    # Its OWN id: several tests in this module start a session, and looking one up
    # by the shared PID finds whichever ran first, not this one's.
    own_pid = "KEYORDERPID"
    enter_id(page, own_pid)
    pass_slides(page)

    expected = ["left", "same", "right", "cant_tell"]
    for index, key in enumerate(["1", "2", "3", "4"]):
        page.keyboard.press(key)
        page.wait_for_timeout(350)
        assert page.inner_text(".counter").startswith(str(index + 2) + " /"), \
            f"key {key} did not advance"

    session_id = next(s["session_id"] for s in stub.SESSIONS.values()
                      if s["prolific_pid"] == own_pid)
    recorded = [row["choice"] for key, row in stub.JUDGMENTS.items()
                if key[0] == session_id]
    assert recorded[:4] == expected, recorded[:4]
    page.close()


def test_the_prompt_is_framed_as_a_question_about_a_friend(server, browser):
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    pass_slides(page)
    lines = [e.inner_text().strip() for e in page.query_selector_all(".framing")]
    assert lines[0] == "If your friend asked an AI chatbot the following:"
    assert lines[1] == "which response would you rather the chatbot show them?"
    page.close()


@pytest.mark.parametrize("width,height", [(1440, 900), (1280, 900), (1280, 700),
                                          (900, 650)])
def test_the_panels_are_equal_and_the_buttons_stay_on_screen(server, browser,
                                                             width, height):
    """The anti-cue property, and the one the old fixed max-height did not give.

    EQUAL HEIGHT matters because our arm's responses are markedly shorter on PAS:
    a panel that simply ended higher than its neighbour would be a cue about which
    arm it is. BUTTONS ON SCREEN matters because a rater who has to scroll the page
    to reach them will scroll a response panel by accident.
    """
    base, _ = server
    page = open_page(browser, base, width=width, height=height)
    enter_id(page)
    pass_slides(page)

    for _ in range(6):
        geometry = page.evaluate("""() => {
          const panels = [...document.querySelectorAll('.pair .response')]
            .map(e => { const r = e.getBoundingClientRect();
                        return [Math.round(r.height), Math.round(r.width)]; });
          const c = document.querySelector('.choices').getBoundingClientRect();
          return {panels, bottom: Math.round(c.bottom),
                  viewport: window.innerHeight,
                  scrolls: document.documentElement.scrollHeight
                           > window.innerHeight + 1};
        }""")
        heights = [p[0] for p in geometry["panels"]]
        widths = [p[1] for p in geometry["panels"]]
        assert len(heights) == 2
        assert len(set(heights)) == 1, f"unequal panel heights {heights}"
        assert len(set(widths)) == 1, f"unequal panel widths {widths}"
        assert geometry["bottom"] <= geometry["viewport"], \
            f"buttons below the fold: {geometry['bottom']} > {geometry['viewport']}"
        assert not geometry["scrolls"], "the rating page scrolls"
        page.click(".choices button[data-choice=left]")
        page.wait_for_timeout(280)

    assert page.errors == []
    page.close()


def test_the_panels_are_bigger_than_the_old_fixed_cap(server, browser):
    """The layout change was asked for as 'make the display boxes bigger', so the
    floor is the height the previous fixed rule gave: max-height: 44vh."""
    base, _ = server
    page = open_page(browser, base, width=1280, height=900)
    enter_id(page)
    pass_slides(page)
    old_cap = 0.44 * 900
    for _ in range(8):
        height = page.evaluate(
            "() => document.querySelector('.pair .response')"
            ".getBoundingClientRect().height")
        assert height > old_cap, f"panel {height:.0f}px is under the old {old_cap:.0f}px"
        page.click(".choices button[data-choice=left]")
        page.wait_for_timeout(260)
    page.close()


# --- resuming --------------------------------------------------------------

def test_a_reload_resumes_the_rating_and_does_not_replay_the_intro(server, browser):
    """A returning rater has already given their ID and read the instructions;
    replaying them would only stand between them and the item they were on."""
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    pass_slides(page)
    for _ in range(3):
        page.click(".choices button[data-choice=same]")
        page.wait_for_timeout(280)
    counter = page.inner_text(".counter")

    page.reload(wait_until="networkidle")
    page.wait_for_timeout(600)
    assert page.query_selector("input[type=text]") is None, "asked for the ID again"
    assert page.query_selector(".slide") is None, "replayed the intro"
    assert page.query_selector(".pair") is not None
    assert page.inner_text(".counter") == counter, "resumed at the wrong item"
    assert page.errors == []
    page.close()


# --- the --label-arms debug view -------------------------------------------

def test_the_default_page_carries_no_arm_labels(server, browser):
    """The blindness of the DEFAULT local page is the whole value of this stub: if
    looking at it locally showed something a rater cannot see, it would stop being
    a preview of what a rater gets."""
    base, _ = server
    page = open_page(browser, base)
    assert "__debug_labels.js" not in page.content(), "debug script injected"
    enter_id(page)
    pass_slides(page)
    assert page.query_selector(".debug-arms") is None
    assert page.evaluate(
        "() => fetch('__debug_arms.json').then(r => r.status)") == 404, \
        "arm names are reachable on the blind page"
    page.close()


def test_label_arms_names_the_side_the_stub_resolves(server, browser, monkeypatch):
    """The debug labels have to be RIGHT, or they are worse than nothing.

    Checked against the stub's own side-to-arm resolution -- the thing that writes
    `left_model`/`right_model` -- rather than against the same `flipped` the labels
    were built from, so the two halves of the mapping are confirmed to compose.
    """
    base, stub = server
    monkeypatch.setattr(stub, "LABEL_ARMS", True)

    page = open_page(browser, base)
    assert "__debug_labels.js" in page.content(), "debug script not injected"
    own_pid = "LABELCHECKPID"
    enter_id(page, own_pid)
    pass_slides(page)

    session_id = next(s["session_id"] for s in stub.SESSIONS.values()
                      if s["prolific_pid"] == own_pid)
    on_screen = []
    for _ in range(5):
        labels = [e.inner_text().strip()
                  for e in page.query_selector_all(".debug-arms span")]
        assert len(labels) == 2, f"expected two labels, got {labels}"
        on_screen.append(tuple(labels))
        page.click(".choices button[data-choice=left]")
        page.wait_for_timeout(320)

    resolved = [(row["left_model"], row["right_model"])
                for key, row in stub.JUDGMENTS.items() if key[0] == session_id]
    assert on_screen == resolved[:5], f"labels {on_screen} != resolved {resolved[:5]}"
    assert page.errors == []
    page.close()


def test_the_console_snippet_labels_the_unmodified_page(server, browser):
    """The path used on the DEPLOYED site: no injected script tag and no
    __debug_arms.json route, just the snippet evaluated in the console.

    Run against the blind page (LABEL_ARMS stays off) so what is being labelled
    is byte-for-byte what a rater is served -- which is the situation on Pages.
    """
    base, stub = server
    snippet = debug_arms.console_snippet(debug_arms.load_items())

    page = open_page(browser, base)
    assert "__debug_labels.js" not in page.content(), "not the blind page"
    own_pid = "SNIPPETCHECKPID"
    enter_id(page, own_pid)
    pass_slides(page)
    page.evaluate(snippet)

    session_id = next(s["session_id"] for s in stub.SESSIONS.values()
                      if s["prolific_pid"] == own_pid)
    on_screen = []
    for _ in range(3):
        labels = [e.inner_text().strip()
                  for e in page.query_selector_all(".debug-arms span")]
        assert len(labels) == 2, f"expected two labels, got {labels}"
        on_screen.append(tuple(labels))
        page.click(".choices button[data-choice=left]")
        page.wait_for_timeout(320)

    resolved = [(row["left_model"], row["right_model"])
                for key, row in stub.JUDGMENTS.items() if key[0] == session_id]
    assert on_screen == resolved[:3], f"labels {on_screen} != resolved {resolved[:3]}"
    assert page.errors == []
    page.close()


def test_the_stub_and_the_page_agree_on_the_item_digest(server):
    """A page holding a different item build must be refused, not silently
    recorded against text nobody can reconstruct."""
    _, stub = server
    with open(os.path.join(config.STATIC_DIR, "items.json"), encoding="utf-8") as fh:
        published = json.load(fh)
    assert published["items_sha256"] == stub.DIGEST
    status, payload = stub.start_session({"p_items_sha256": "stale",
                                          "p_prolific_pid": PID})
    assert status == 400 and "mismatch" in payload["message"]
