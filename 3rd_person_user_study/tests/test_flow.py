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
import re
import threading

import pytest

import build_static
import config
import debug_arms
import local_server


PID = "FLOWTESTPID"

#: The measured worst case for a response panel UNDER THE FILL LAYOUT, with a
#: little slack: 215px over all 72 items at both 1440x900 and 1280x900, with the
#: questions stacked and set at 18px bold. ONE constant for both layout tests --
#: they had drifted to 240 and 280, so a layout change failed one and passed the
#: other and the pair no longer said anything coherent about how small a panel
#: may get. The history behind the number is on
#: test_the_panels_keep_a_usable_height_on_a_tall_viewport.
PANEL_FLOOR = 200

#: The intro is three slides. Asserted rather than discovered, so adding one to
#: app.js without updating these tests fails instead of going untested.
SLIDE_COUNT = 3

#: What tools/local_server.py tells the page the per-item floor is. Production is
#: config.MIN_ITEM_SECONDS (60); the stub serves a small value so a run of this
#: suite is not 72 minutes of waiting. Imported from the stub rather than
#: repeated, so the two cannot disagree.


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
    # Every rater arrives on a #set_N link; without one the page refuses to
    # start, so the default here is a set rather than a bare URL.
    if "#" not in query:
        query = query + "#set_1"
    page.goto(base + "/" + query, wait_until="networkidle")
    return page


def agree(page):
    """Consent, which is now the first screen. Every test that needs the ID
    screen or beyond goes through here."""
    page.click(".consent-option[data-choice=agree]")
    page.click(".actions button")
    page.wait_for_timeout(300)


def enter_id(page, pid=PID, consent=True):
    """Consent, then the Prolific ID. `consent=False` for tests that are already
    past the consent screen or are asserting about it."""
    if consent:
        agree(page)
    page.fill("input[type=text]", pid)
    page.click(".actions button")
    page.wait_for_timeout(400)


def answer_item(page, preference="left", realize="left"):
    """Answer BOTH questions for the item on screen and press Continue.

    An item takes three clicks and a WAIT now, not one click: Continue is gated
    on both questions being answered AND the per-item minimum having elapsed. So
    every test that advances has to go through here -- one that clicked a choice
    and assumed the page turned would silently assert about the same item over
    and over.
    """
    page.click(f'.question[data-question=preference] button[data-choice={preference}]')
    page.click(f'.question[data-question=realize] button[data-choice={realize}]')
    page.fill(".reasoning textarea", "because it is more direct about it")
    # Wait out the floor rather than sleeping a fixed amount: the button re-
    # enables itself on a ticker, and polling for it keeps this honest if the
    # constant changes.
    page.wait_for_selector(".actions button:not([disabled])", timeout=15000)
    page.click(".actions button")
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
    agree(page)
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
    agree(page)
    assert page.input_value("input[type=text]") == "FROMURL42"
    page.close()


def test_digits_can_be_typed_into_the_id_field(server, browser):
    """The choice keys are bound on `document` and call preventDefault. Gating
    them on the stage is what keeps 1-4 from being swallowed here -- Prolific IDs
    contain digits."""
    base, _ = server
    page = open_page(browser, base)
    agree(page)
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
    assert "45 seconds have elapsed" in seen[1]
    assert "Can" in seen[2] and "tell" in seen[2]
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

    The wording of Q2 is asserted in full because it is the measurement: it was
    reworded to "reflect on their actions and potentially realize they are
    wrong" precisely to soften the premise that the writer IS wrong, and a drift
    back would change what raters are being asked without changing anything that
    looks like code."""
    path = os.path.join(config.STATIC_DIR, "app.js")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    joined = re.sub(r"' \+\s*'", "", source)
    assert "Which response do you prefer?" in joined
    assert ("Which response is more likely to make the user reflect on their "
            "actions and realize they are wrong?") in joined
    for choice in config.REALIZE_CHOICES:
        assert f"'{choice}'" in source, f"no option for {choice!r}"


def test_the_premise_rejecting_option_is_gone():
    """`not_wrong` was removed from Q2 (2026-08-05). The VALUE has to be gone
    from both the page and the domain the database accepts: a leftover in one
    but not the other is either an option nobody can select that the schema
    still allows, or -- worse -- one the page offers and the schema rejects
    mid-session.

    Checks the value, not the old label text, which still appears in a comment
    in app.js recording why the option went. That comment is not a leak: what a
    rater can select is asserted against the RENDERED buttons in
    test_the_screen_labels_the_prompt_and_both_responses.
    """
    path = os.path.join(config.STATIC_DIR, "app.js")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert "'not_wrong'" not in source, "the page can still send not_wrong"
    assert "data-choice=not_wrong" not in source
    assert "not_wrong" not in config.REALIZE_CHOICES
    assert list(config.REALIZE_CHOICES) == list(config.CHOICES)


def test_the_screen_labels_the_prompt_and_both_responses(server, browser):
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    pass_slides(page)

    assert page.inner_text(".prompt-label").strip() == "Prompt"
    labels = [e.inner_text().strip()
              for e in page.query_selector_all(".response-label")]
    assert labels == ["Response 1", "Response 2"], labels

    for key, expected in [("preference", ["Response 1", "Response 2",
                                          "Can\u2019t tell"]),
                          ("realize", ["Response 1", "Response 2",
                                       "Can\u2019t tell"])]:
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
    assert button.is_disabled(), "Continue live with no reasoning"
    page.fill(".reasoning textarea", "the second one is blunter")
    page.wait_for_selector(".actions button:not([disabled])", timeout=15000)

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

    page.click(".question[data-question=realize] button[data-choice=cant_tell]")
    page.fill(".reasoning textarea", "response two names the harm")
    page.wait_for_selector(".actions button:not([disabled])", timeout=15000)
    page.click(".actions button")
    page.wait_for_timeout(350)

    session_id = next(s["session_id"] for s in stub.SESSIONS.values()
                      if s["prolific_pid"] == own_pid)
    row = next(r for key, r in stub.JUDGMENTS.items() if key[0] == session_id)
    assert row["choice"] == "right"
    assert row["realize_choice"] == "cant_tell"
    assert row["realize_model"] is None, "cant_tell must resolve to no model"
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

    NO-SCROLL IS GONE, and this is where it went. The compare screen started as
    a viewport-filling layout that never scrolled: prompt, two panels, one row of
    buttons. It has since gained a second question, a stacked layout, larger
    question type, and a free-text box with a Continue below it. Measured at
    1280x900 with all of that, the content needs ~917px against a 900px viewport
    -- and the panels are already sitting ON their floor, so the only way to fit
    is to shrink them further.

    That is the trade, and it went the other way on purpose: a rater scrolling
    the page costs a scroll, a 150px response panel costs them the ability to
    read the thing they are judging. So the page scrolls at laptop heights (it
    still fits without scrolling at ~1100px tall) and the floor is what this
    test defends.

    What is NOT negotiable and is asserted at every viewport is EQUAL PANEL
    HEIGHT -- our arm is markedly shorter on PAS, so a panel that ended higher
    than its neighbour would be a cue about which arm wrote it.
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
        if height >= 820:
            # THE FLOOR ONLY MEANS ANYTHING UNDER THE FILL LAYOUT, where a panel
            # is the height left over after the furniture -- that is the number
            # that shrinks when a question or a label is added. Below 820px tall
            # the fill layout is off and a panel is sized to its CONTENT, so a
            # short response gives a small box: measured down to 134px at
            # 1280x700 over the 72 items, with both panels still equal. Asserting
            # the floor there would be asserting that no response is short.
            assert heights[0] >= PANEL_FLOOR, f"panel crushed to {heights[0]}px"
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
      3. Setting the question text at 18px BOLD, up from 15px regular, and
         dropping Q2's fifth option. Re-measured: min 215 / median 286 /
         max 311. The lost option gave back a button row, which roughly paid
         for the larger type.

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
        answer_item(page, preference="cant_tell", realize="cant_tell")
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

def test_the_deployed_build_has_no_arm_labels():
    """The mapping is the answer key. The build must not emit it, and the page
    must not be able to ask for it."""
    for name in ("__debug_arms.json", "debug_labels.js"):
        assert not os.path.exists(os.path.join(config.STATIC_DIR, name)), \
            f"static/{name} would be deployed"
    # The CODE, not the prose: a comment in app.js records why the overlay went
    # and naturally mentions the flag by name.
    with open(os.path.join(config.STATIC_DIR, "app.js"), encoding="utf-8") as h:
        source = h.read()
    assert "'debug_labels.js'" not in source, "the page can still load the overlay"
    assert "__debug_arms.json" not in source, "the page can still fetch the mapping"
    assert "createElement('script')" not in source


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
    agree(page)
    assert page.input_value("input[type=text]") == "FROMURL42"
    enter_id(page, "FROMURL42", consent=False)
    pass_slides(page)
    assert page.query_selector(".debug-arms") is not None, "no labels with two params"
    page.close()


# --- consent ---------------------------------------------------------------

def test_consent_is_the_first_screen(server, browser):
    """Consent comes before the Prolific ID, so nobody is asked to identify
    themselves before agreeing to take part."""
    base, _ = server
    page = open_page(browser, base)
    assert page.query_selector(".consent-title") is not None, "no consent screen"
    assert page.query_selector("input[type=text]") is None, "ID asked before consent"
    labels = [b.inner_text().strip()
              for b in page.query_selector_all(".consent-option")]
    assert labels == ["I agree to participate in the study",
                      "I do not wish to participate in the study"], labels
    page.close()


def test_the_consent_text_is_study_slash_content(server, browser):
    """Imported from study/content.py, not written twice. Both studies recruit
    the same pool under the same protocol; two copies would drift."""
    import content

    base, _ = server
    page = open_page(browser, base)
    assert page.inner_text(".consent-title").strip() == content.CONSENT_TITLE
    shown = page.inner_text(".consent-body")
    for marker in ["DESCRIPTION:", "TIME INVOLVEMENT:", "PARTICIPANT’S RIGHTS:",
                   "irbnonmed@stanford.edu"]:
        assert marker in shown, marker
    page.close()


def test_the_consent_mismatches_are_still_there_to_sign_off():
    """The imported text describes study/, and three spans of it are NOT true of
    this study. They are left verbatim because the form is IRB-facing, and
    tracked here so they cannot be quietly forgotten -- if one is edited in
    study/content.py this fails and build_static.CONSENT_MISMATCHES needs
    revisiting."""
    body = build_static.consent()["body"]
    for span in build_static.CONSENT_MISMATCHES:
        assert span in body, (
            f"{span!r} is no longer in the consent text -- if it was fixed, drop "
            "it from CONSENT_MISMATCHES")


def test_declining_records_nothing(server, browser):
    """A decliner leaves NO row: no session, no Prolific ID, nothing to filter
    out of the export later."""
    base, stub = server
    before_sessions, before_judgments = len(stub.SESSIONS), len(stub.JUDGMENTS)
    page = open_page(browser, base)
    page.click(".consent-option[data-choice=decline]")
    page.click(".actions button")
    page.wait_for_timeout(300)

    assert page.query_selector("input[type=text]") is None, "asked for an ID anyway"
    assert page.query_selector(".pair") is None, "reached the rating screen"
    assert "not to participate" in page.inner_text("body")
    assert len(stub.SESSIONS) == before_sessions, "a session was started"
    assert len(stub.JUDGMENTS) == before_judgments
    page.close()


def test_continue_needs_a_consent_choice(server, browser):
    base, _ = server
    page = open_page(browser, base)
    page.click(".actions button")
    page.wait_for_timeout(250)
    assert page.query_selector(".error") is not None, "no message shown"
    assert page.query_selector(".consent-title") is not None, "left the screen"
    page.close()


# --- the per-item minimum --------------------------------------------------

def test_continue_is_held_until_the_minimum_elapses(server, browser, monkeypatch):
    """Both questions answered is NOT enough: the item also has to have been on
    screen for the floor. Without this a rater can click through 72 items in a
    couple of minutes, and there is no right answer here to catch them with --
    the only thing that separates reading from clicking is time.

    The floor is set ABSURDLY HIGH for this page rather than a few seconds. With
    a short one the assertions below raced their own setup: filling three fields
    and reading the button took longer than the floor, so it had already
    unlocked by the time the click landed and the test failed on a page that was
    working. Nothing here waits for the unlock -- that is the next test's job --
    so a floor nobody can outlast is exactly right.
    """
    base, stub = server
    monkeypatch.setattr(local_server, "LOCAL_MIN_ITEM_SECONDS", 600)
    page = open_page(browser, base)
    enter_id(page, "MINTIMEPID")
    pass_slides(page)

    page.click(".question[data-question=preference] button[data-choice=left]")
    page.click(".question[data-question=realize] button[data-choice=left]")
    page.fill(".reasoning textarea", "it challenges rather than agrees")

    button = page.query_selector(".actions button")
    assert button.is_disabled(), "Continue live before the minimum elapsed"
    assert "s)" in button.inner_text(), \
        f"no countdown on the button: {button.inner_text()!r}"

    # dispatch_event, not click(): playwright's click WAITS for the button to
    # become enabled and would just time out here. Dispatching invokes the
    # listener directly, which is the stronger test anyway -- it exercises
    # submitAnswers' own secondsLeft() guard rather than the browser's refusal
    # to click a disabled button.
    before = len(stub.JUDGMENTS)
    button.dispatch_event("click")
    page.wait_for_timeout(250)
    assert len(stub.JUDGMENTS) == before, "a judgment was recorded early"
    assert page.inner_text(".counter").startswith("1 /"), "the page turned anyway"
    page.close()


def test_continue_unlocks_once_the_minimum_has_passed(server, browser):
    """The other half: the floor lets go. Runs against the stub's short floor so
    the wait is seconds, not the production minute."""
    base, _ = server
    page = open_page(browser, base)
    enter_id(page, "MINTIMEOKPID")
    pass_slides(page)

    page.click(".question[data-question=preference] button[data-choice=left]")
    page.click(".question[data-question=realize] button[data-choice=left]")
    page.fill(".reasoning textarea", "it names the harm to the other person")
    page.wait_for_selector(".actions button:not([disabled])", timeout=15000)
    assert page.inner_text(".actions button").strip() == "Continue", \
        "the countdown is still on the button after it unlocked"
    page.click(".actions button")
    page.wait_for_timeout(300)
    assert page.inner_text(".counter").startswith("2 /")
    page.close()


def test_the_minimum_restarts_on_the_next_item(server, browser):
    """The floor is PER ITEM. A single ticker left running, or a shownAt that was
    not reset, would let every item after the first through instantly."""
    base, _ = server
    page = open_page(browser, base)
    enter_id(page, "MINTIMEPID2")
    pass_slides(page)
    answer_item(page)

    page.click(".question[data-question=preference] button[data-choice=left]")
    page.click(".question[data-question=realize] button[data-choice=left]")
    page.fill(".reasoning textarea", "same reason as the previous item")
    assert page.query_selector(".actions button").is_disabled(), \
        "the second item did not restart the clock"
    page.close()


def test_the_slide_tells_raters_about_the_wait(server, browser):
    """A minute of dead button with no warning reads as a broken page."""
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    seen = pass_slides(page)
    assert any("45 seconds have elapsed" in line for line in seen), seen
    assert any("not be allowed to press continue" in line for line in seen), seen
    page.close()


def test_production_holds_items_for_a_full_minute():
    """The stub serves a small floor so this suite is not 72 minutes long. The
    number a RATER gets comes from config, and that is the one the instruction
    slide promises."""
    assert config.MIN_ITEM_SECONDS == 45
    assert local_server.LOCAL_MIN_ITEM_SECONDS < config.MIN_ITEM_SECONDS


# --- Q3, the free-text reasoning -------------------------------------------

def test_reasoning_is_required_and_recorded(server, browser):
    """Continue stays dead until the box has MIN_REASONING_CHARS, and what the
    rater typed is what lands in the row."""
    base, stub = server
    page = open_page(browser, base)
    own_pid = "REASONPID"
    enter_id(page, own_pid)
    pass_slides(page)

    assert page.inner_text(".reasoning .ask").strip() == "Briefly, explain your reasoning"
    page.click(".question[data-question=preference] button[data-choice=left]")
    page.click(".question[data-question=realize] button[data-choice=right]")
    button = page.query_selector(".actions button")
    assert button.is_disabled(), "Continue live with an empty box"

    page.fill(".reasoning textarea", "too short")     # under the floor
    assert button.is_disabled(), "Continue live under the character floor"
    assert "more characters" in page.inner_text(".reasoning-note")

    page.fill(".reasoning textarea", "   " + "x" * (config.MIN_REASONING_CHARS - 1) + "   ")
    assert button.is_disabled(), "whitespace padded the answer over the floor"

    page.fill(".reasoning textarea", "response one actually pushes back")
    assert page.inner_text(".reasoning-note").strip() == ""
    page.wait_for_selector(".actions button:not([disabled])", timeout=15000)
    page.click(".actions button")
    page.wait_for_timeout(350)

    session_id = next(s["session_id"] for s in stub.SESSIONS.values()
                      if s["prolific_pid"] == own_pid)
    row = next(r for key, r in stub.JUDGMENTS.items() if key[0] == session_id)
    assert row["reasoning"] == "response one actually pushes back"
    page.close()


def test_the_reasoning_box_clears_between_items(server, browser):
    """Per item, like the two choices. Carrying the previous answer over would
    both prefill an answer nobody gave and unlock Continue early."""
    base, _ = server
    page = open_page(browser, base)
    enter_id(page, "REASONPID2")
    pass_slides(page)
    answer_item(page)
    assert page.input_value(".reasoning textarea") == ""
    assert page.query_selector(".actions button").is_disabled()
    page.close()


def test_the_schema_enforces_the_reasoning_floor():
    """The page's copy of the rule is advisory -- anything that can post to the
    RPC can skip it -- so the length is also a CHECK and a guard in pref_record."""
    with open(os.path.join(config.HERE, "supabase_schema.sql"), encoding="utf-8") as h:
        sql = h.read()
    floor = config.MIN_REASONING_CHARS
    assert f"check (length(btrim(reasoning)) >= {floor})" in sql
    assert f"if length(v_reasoning) < {floor} then" in sql
    assert "drop function if exists public.pref_record(text, text, text, text, integer);" in sql, \
        "the 5-argument pref_record would survive as an overload"


def test_there_is_no_same_button(server, browser):
    """"Same" and "Can't tell" were two buttons for one outcome -- neither
    response picked -- so "Same" went. It has to be gone from the page and from
    the domain, or a stale value stays legal in the data."""
    base, _ = server
    page = open_page(browser, base)
    enter_id(page)
    pass_slides(page)
    for key in ("preference", "realize"):
        values = page.eval_on_selector_all(
            f".question[data-question={key}] .choices button",
            "bs => bs.map(b => b.dataset.choice)")
        assert values == ["left", "right", "cant_tell"], (key, values)
    assert "same" not in config.CHOICES
    with open(os.path.join(config.HERE, "supabase_schema.sql"), encoding="utf-8") as h:
        assert "'same'" not in h.read(), "the schema still allows 'same'"
    page.close()
