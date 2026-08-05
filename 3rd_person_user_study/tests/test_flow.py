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

#: The measured worst case for a response panel, with a little slack: min 215px
#: over all 72 items at 1280x900 with the questions stacked. ONE constant for
#: both layout tests -- they had drifted to 240 and 280, so a layout change
#: failed one and passed the other and the pair no longer said anything
#: coherent about how small a panel may get. The history behind the number is on
#: test_the_panels_keep_a_usable_height_on_a_tall_viewport.
PANEL_FLOOR = 200

#: The intro is two slides. Asserted rather than discovered, so adding one to
#: app.js without updating these tests fails instead of going untested.
SLIDE_COUNT = 2


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


def answer_item(page, preference="left", realize="left"):
    """Answer BOTH questions for the item on screen and press Continue.

    An item takes three clicks now, not one, so every test that advances has to
    go through here -- clicking a single choice no longer turns the page, and a
    test that assumed it would would silently be asserting about the same item
    over and over.
    """
    page.click(f'.question[data-question=preference] button[data-choice={preference}]')
    page.click(f'.question[data-question=realize] button[data-choice={realize}]')
    button = page.query_selector(".actions button")
    assert not button.is_disabled(), "Continue still disabled with both answered"
    button.click()
    page.wait_for_timeout(350)


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

def test_the_intro_is_two_slides_then_the_rating_screen(server, browser):
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    seen = pass_slides(page)

    assert len(seen) == SLIDE_COUNT
    assert "two different responses" in seen[0]
    assert "Can" in seen[1] and "tell" in seen[1]
    # The "imagine your friend was writing the prompt" slide was removed.
    assert not any("friend" in line for line in seen), seen
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


def test_both_questions_are_asked_with_their_own_options():
    """The two questions and their option sets, pinned against config.

    Q2 carries an option Q1 does not -- `not_wrong` -- and that asymmetry is the
    whole reason the two have separate domains. A page that offered the same four
    buttons twice would look right and quietly drop the distinction."""
    path = os.path.join(config.STATIC_DIR, "app.js")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert "Which response do you prefer?" in source
    assert ("Which response do you think is more likely to make the person "
            "who wrote the prompt realize they are wrong?") in source.replace(
                "' +\n            '", "")
    assert "I don\u2019t think the person should feel like they were wrong" in source
    for choice in config.REALIZE_CHOICES:
        assert f"'{choice}'" in source, f"no option for {choice!r}"


def test_the_screen_labels_the_prompt_and_both_responses(server, browser):
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    pass_slides(page)

    assert page.inner_text(".prompt-label").strip() == "Prompt"
    labels = [e.inner_text().strip()
              for e in page.query_selector_all(".response-label")]
    assert labels == ["Response 1", "Response 2"], labels

    for key, expected in [("preference", ["Response 1", "Same", "Response 2",
                                          "Can\u2019t tell"]),
                          ("realize", ["Response 1", "Same", "Response 2",
                                       "Can\u2019t tell",
                                       "I don\u2019t think the person should feel "
                                       "like they were wrong"])]:
        shown = [b.inner_text().strip() for b in page.query_selector_all(
            f".question[data-question={key}] .choices button")]
        assert shown == expected, (key, shown)
    assert page.errors == []
    page.close()


def test_the_page_no_longer_frames_the_prompt_as_a_friends(server, browser):
    """The friend framing was removed, and removed means gone from the rendered
    page -- not merely unused in one code path."""
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    pass_slides(page)
    body = page.inner_text("body").lower()
    for phrase in ["your friend", "friend asked", "show your friend"]:
        assert phrase not in body, f"the page still says {phrase!r}"
    assert page.query_selector(".framing") is None
    page.close()


def test_a_digit_keypress_records_nothing(server, browser):
    """The 1-4 shortcuts were REMOVED when the second question arrived: a bare
    digit cannot say which question it means, and guessing would record a real
    answer to the wrong one. Pinned so they cannot come back by accident in the
    ambiguous form."""
    base, stub = server
    page = open_page(browser, base)
    own_pid = "NOKEYSPID"
    enter_id(page, own_pid)
    pass_slides(page)

    before = len(stub.JUDGMENTS)
    for key in ["1", "2", "3", "4", "ArrowLeft", "ArrowRight"]:
        page.keyboard.press(key)
        page.wait_for_timeout(120)
    assert page.inner_text(".counter").startswith("1 /"), "a key turned the page"
    assert len(stub.JUDGMENTS) == before, "a key recorded a judgment"
    assert page.query_selector(".choices button.chosen") is None, \
        "a key selected an option"
    page.close()


def test_continue_is_blocked_until_both_questions_are_answered(server, browser):
    """The page's guard against a half-answered item. realize_choice is NOT
    NULL, so a row with one answer is not something the database would take --
    it has to be impossible to ask for."""
    base, stub = server
    page = open_page(browser, base)
    own_pid = "HALFANSWERPID"
    enter_id(page, own_pid)
    pass_slides(page)

    button = page.query_selector(".actions button")
    assert button.is_disabled(), "Continue live with nothing answered"

    page.click(".question[data-question=preference] button[data-choice=left]")
    assert button.is_disabled(), "Continue live with only Q1 answered"

    page.click(".question[data-question=realize] button[data-choice=cant_tell]")
    assert not button.is_disabled(), "Continue still dead with both answered"

    before = len(stub.JUDGMENTS)
    button.click()
    page.wait_for_timeout(350)
    assert len(stub.JUDGMENTS) == before + 1
    assert page.inner_text(".counter").startswith("2 /")
    page.close()


def test_an_answer_can_be_changed_before_continue(server, browser):
    """Selecting no longer advances, so a misclick is correctable -- and only the
    final selection is what gets written."""
    base, stub = server
    page = open_page(browser, base)
    own_pid = "CHANGEMINDPID"
    enter_id(page, own_pid)
    pass_slides(page)

    page.click(".question[data-question=preference] button[data-choice=left]")
    page.click(".question[data-question=preference] button[data-choice=right]")
    chosen = page.query_selector_all(
        ".question[data-question=preference] button.chosen")
    assert len(chosen) == 1 and chosen[0].get_attribute("data-choice") == "right"

    page.click(".question[data-question=realize] button[data-choice=not_wrong]")
    page.click(".actions button")
    page.wait_for_timeout(350)

    session_id = next(s["session_id"] for s in stub.SESSIONS.values()
                      if s["prolific_pid"] == own_pid)
    row = next(r for key, r in stub.JUDGMENTS.items() if key[0] == session_id)
    assert row["choice"] == "right"
    assert row["realize_choice"] == "not_wrong"
    assert row["realize_model"] is None, "not_wrong must resolve to no model"
    page.close()


@pytest.mark.parametrize("width,height", [(1440, 900), (1280, 900), (1280, 700),
                                          (900, 650)])
def test_the_panels_are_equal_and_the_buttons_stay_on_screen(server, browser,
                                                             width, height):
    """The anti-cue property, and a floor on how small the panels can get.

    EQUAL HEIGHT matters because our arm's responses are markedly shorter on PAS:
    a panel that simply ended higher than its neighbour would be a cue about which
    arm it is. That holds at every viewport and is the assertion that must never
    be relaxed.

    NO-SCROLL IS NOW CONDITIONAL (changed 2026-08-05, with the second question).
    Two question blocks plus Continue are ~150px of furniture on top of the
    prompt. Held to the old "never scrolls" rule at every size, the remainder for
    the panels was 53px at 700px tall and ZERO at 650px -- so the fill layout is
    gated on a 820px-tall viewport and shorter ones scroll instead. Scrolling
    costs the rater a scroll; a 0px panel costs them the item. The panel floor
    below is what makes that trade real rather than a licence to shrink.
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
        assert heights[0] >= PANEL_FLOOR, f"panel crushed to {heights[0]}px"
        if height >= 820:
            assert not geometry["scrolls"], "a tall viewport should not scroll"
            assert geometry["bottom"] <= geometry["viewport"], \
                f"buttons below the fold: {geometry['bottom']} > {geometry['viewport']}"
        answer_item(page)
        page.wait_for_timeout(280)

    assert page.errors == []
    page.close()


def test_the_panels_keep_a_usable_height_on_a_tall_viewport(server, browser):
    """A floor on panel height, measured rather than guessed.

    HISTORY, because the number has moved twice and both moves were deliberate.
    The layout was originally asked for as "make the display boxes bigger", and
    the bar was the previous fixed rule -- max-height: 44vh, a flat 396px at 900.

    Then the panels paid for two things, in order:
      1. a SECOND QUESTION (2026-08-05). Two question blocks and Continue sit
         between the panels and the bottom of the screen. Laid out side by side
         above 1100px this measured min 295 / median 391 / max 416 over all 72
         items at 1280x900 -- a wash against the old cap.
      2. STACKING those questions (2026-08-05), so that "Which response do you
         prefer?" is centred on the page and the follow-up sits under it rather
         than beside it. Re-measured the same way: min 215 / median 271 /
         max 297. Stacking costs ~120px of median panel height.

    The floor here is that worst case with a little slack. It is deliberately not
    44vh and not the side-by-side number either; asserting a height the current
    layout cannot deliver would only mean a test that fails on the long-prompt
    items, and quietly lowering it every time something is added is how a panel
    ends up unreadable.

    The remaining lever is the prompt cap, currently 20vh: dropping it to 15vh
    buys back ~45px on every item and costs a scroll on the long prompts. That is
    a call about the instrument, not something to bury in a stylesheet.
    """
    base, _ = server
    page = open_page(browser, base, width=1280, height=900)
    enter_id(page)
    pass_slides(page)
    floor = PANEL_FLOOR
    for _ in range(8):
        height = page.evaluate(
            "() => document.querySelector('.pair .response')"
            ".getBoundingClientRect().height")
        assert height >= floor, f"panel {height:.0f}px is under the {floor}px floor"
        answer_item(page)
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
        answer_item(page, preference="same", realize="same")
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


# --- the ?labels=1 arm overlay ---------------------------------------------

def test_the_plain_url_carries_no_arm_labels(server, browser):
    """Without the flag the page must be the one a rater gets: no overlay, and
    no request for the mapping. The mapping being PUBLISHED is not the same as
    the rating page reaching for it, and this is the difference."""
    base, _ = server
    page = open_page(browser, base)
    requested = []
    page.on("request", lambda r: requested.append(r.url))
    enter_id(page)
    pass_slides(page)

    assert page.query_selector(".debug-arms") is None, "labels on the plain URL"
    leaked = [url for url in requested
              if "__debug_arms" in url or "debug_labels" in url]
    assert not leaked, f"the rating page fetched the mapping: {leaked}"
    assert page.errors == []
    page.close()


def test_labels_flag_names_the_side_the_stub_resolves(server, browser):
    """The labels have to be RIGHT, or they are worse than nothing.

    Checked against the stub's own side-to-arm resolution -- the thing that writes
    `left_model`/`right_model` -- rather than against the same `flipped` the labels
    were built from, so the two halves of the mapping are confirmed to compose.
    """
    base, stub = server
    page = open_page(browser, base, query="?labels=1")
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
        answer_item(page)

    resolved = [(row["left_model"], row["right_model"])
                for key, row in stub.JUDGMENTS.items() if key[0] == session_id]
    assert on_screen == resolved[:5], f"labels {on_screen} != resolved {resolved[:5]}"
    assert page.errors == []
    page.close()


def test_the_flag_survives_the_prolific_id_in_the_url(server, browser):
    """Prolific appends its own query parameter, and the link you walk the study
    with carries both. A flag read with URLSearchParams handles that; one matched
    with a naive endswith would not."""
    base, _ = server
    page = open_page(browser, base, query="?PROLIFIC_PID=FROMURL42&labels=1")
    assert page.input_value("input[type=text]") == "FROMURL42"
    enter_id(page, "FROMURL42")
    pass_slides(page)
    assert page.query_selector(".debug-arms") is not None, "no labels with two params"
    page.close()
