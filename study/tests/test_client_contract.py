"""Contract tests between the client and the database.

The client owns the flow now, which means a whole class of bug that used to be
impossible is suddenly possible: the browser can drift from the server without
anything failing until a participant hits it in production. Nothing else in this
suite executes app.js, so these tests parse it and assert the three couplings
that would break silently:

  1. Every RPC the client calls is granted to anon (an ungranted one 403s).
  2. The payload the client builds has exactly the fields submit_study reads —
     no missing field, no field invented and then dropped.
  3. The client's stage list matches the questionnaire pages in the bundle.

esprima parses the file rather than a regex reading it, so a renamed function or
a reformatted literal cannot make a check silently vacuous.
"""

import os
import sys

import re

import esprima
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import questionnaires

_STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "static")
#: The client is two files: app.js holds state and the stage machine, screens.js
#: holds the page renderers. Both are parsed, because api() calls and payload
#: writes are spread across the pair.
_CLIENT_FILES = ("app.js", "screens.js", "widgets.js")


def _parse(name):
    with open(os.path.join(_STATIC, name), encoding="utf-8") as handle:
        # range=True so literals can be recovered in SOURCE order; the generic
        # walk below visits attributes alphabetically, which is not the order
        # they appear in the file.
        return esprima.parseScript(handle.read(), options={"range": True})


@pytest.fixture(scope="module")
def tree():
    """The app.js AST — state, stage machine, transport."""
    return _parse("app.js")


@pytest.fixture(scope="module")
def trees():
    """Every client file's AST, keyed by filename."""
    return {name: _parse(name) for name in _CLIENT_FILES}


def _walk(node):
    """Yield every AST node, depth first."""
    if isinstance(node, list):
        for item in node:
            yield from _walk(item)
        return
    if not hasattr(node, "type"):
        return
    yield node
    for key in dir(node):
        if key.startswith("_") or key == "type":
            continue
        yield from _walk(getattr(node, key))


def _find_function(tree, name):
    """The FunctionDeclaration named `name`. Asserts it exists."""
    for node in _walk(tree):
        if node.type == "FunctionDeclaration" and node.id and node.id.name == name:
            return node
    raise AssertionError("static/app.js has no function %r — did it get renamed?" % (name,))


def _find_function_expression(tree, screen_name):
    """The function assigned to APP.screens.<screen_name>."""
    for node in _walk(tree):
        if (node.type == "AssignmentExpression"
                and node.left.type == "MemberExpression"
                and getattr(node.left.property, "name", None) == screen_name):
            return node.right
    return None


def _string_or_ident_names(nodes):
    """Every identifier name appearing anywhere inside `nodes`."""
    names = set()
    for node in nodes:
        for inner in _walk(node):
            if inner.type == "Identifier":
                names.add(inner.name)
    return names


def _string_literals(node):
    """Every string literal under `node`, in SOURCE order."""
    found = [n for n in _walk(node)
             if n.type == "Literal" and isinstance(n.value, str)]
    found.sort(key=lambda n: n.range[0])
    return [n.value for n in found]


# --- 1. Endpoints ----------------------------------------------------------

def _rpc_calls(tree):
    """Function-name literals passed to rpc(...) or APP.rpc(...)."""
    called = set()
    for node in _walk(tree):
        if node.type != "CallExpression" or not node.arguments:
            continue
        callee = node.callee
        name = getattr(callee, "name", None)
        if name is None and callee.type == "MemberExpression":
            name = getattr(callee.property, "name", None)
        if name == "rpc" and node.arguments[0].type == "Literal":
            called.add(node.arguments[0].value)
    return called


def _declared_rpcs():
    """RPC names granted to `anon` in the generated schema.

    Read from the grants rather than the create statements on purpose: a
    function that exists but was never granted is unreachable from the browser,
    and that is exactly the failure this test should catch.
    """
    path = os.path.join(os.path.dirname(_STATIC), "supabase_app_schema.sql")
    with open(path, encoding="utf-8") as handle:
        sql = handle.read()
    return set(re.findall(r"grant execute on function public\.([a-z_]+)\(", sql))


def test_every_rpc_the_client_calls_is_granted_to_anon(trees):
    called = set()
    for tree in trees.values():
        called |= _rpc_calls(tree)
    assert called, "found no rpc() calls — the extraction is broken, not the client"

    granted = _declared_rpcs()
    missing = called - granted
    assert not missing, (
        "client calls RPCs that are not granted to anon (so they would 403 in "
        "the browser): %s" % (sorted(missing),))


def test_the_client_never_calls_a_worker_rpc(trees):
    """Worker RPCs need the shared secret, which must never reach a browser."""
    called = set()
    for tree in trees.values():
        called |= _rpc_calls(tree)
    worker_rpcs = {name for name in called if name.startswith("worker_")}
    assert not worker_rpcs, "client calls worker RPCs: %s" % (sorted(worker_rpcs),)
    for name, tree in trees.items():
        for node in _walk(tree):
            if node.type == "Literal" and isinstance(node.value, str):
                assert "worker_token" not in node.value, \
                    "%s mentions worker_token: %r" % (name, node.value)


def test_the_client_makes_exactly_three_kinds_of_call(trees):
    """The whole point of the redesign. A fourth means a page turn regressed."""
    called = set()
    for tree in trees.values():
        called |= _rpc_calls(tree)
    assert called == {"start_session", "start_response", "response_status",
                      "retry_response", "submit_study"}, sorted(called)


def test_only_the_three_expected_screens_talk_to_the_database(trees):
    """Every OTHER page turn must be local, which is the latency fix itself."""
    calls = _rpc_calls(trees["screens.js"])
    assert calls == {"start_response", "response_status", "retry_response",
                     "submit_study"}, (
        "a screen gained or lost a database call: %s" % (sorted(calls),))
    assert _rpc_calls(trees["widgets.js"]) == set(), \
        "widgets.js must stay pure DOM — a widget that fetches is a page-turn stall"


def test_the_client_never_embeds_a_secret_key(trees):
    """The anon key is public and expected in page source; a secret one is not."""
    for name, tree in trees.items():
        for node in _walk(tree):
            if node.type == "Literal" and isinstance(node.value, str):
                assert "sb_secret_" not in node.value and "service_role" not in node.value, \
                    "%s embeds what looks like a secret key: %r" % (name, node.value[:40])


# --- 2. Payload shape ------------------------------------------------------

#: The fields submit_study reads out of the client's payload.
EXPECTED_PAYLOAD_FIELDS = {
    "consent", "prolific_pid", "eligibility", "selected_scenario", "screening",
    "perspective_text", "additional_thoughts", "answers", "client_timings",
}


def test_the_client_payload_has_exactly_the_fields_the_validator_reads(tree):
    fresh = _find_function(tree, "freshData")
    returned = None
    for node in _walk(fresh):
        if node.type == "ReturnStatement" and node.argument.type == "ObjectExpression":
            returned = node.argument
            break
    assert returned is not None, "freshData() does not return an object literal"

    keys = {prop.key.name if prop.key.type == "Identifier" else prop.key.value
            for prop in returned.properties}
    assert keys == EXPECTED_PAYLOAD_FIELDS, (
        "client payload fields drifted from flow.validate_submission.\n"
        "  only in client: %s\n  only in server: %s"
        % (sorted(keys - EXPECTED_PAYLOAD_FIELDS),
           sorted(EXPECTED_PAYLOAD_FIELDS - keys)))


def test_submit_study_actually_reads_every_field_the_client_sends():
    """Guards the other direction: a field the client sends but nobody reads.

    The validator is plpgsql now, so this greps the generated SQL. Crude, but it
    catches the real failure — renaming a field on one side only, which would
    silently drop it into the payload blob and out of its promoted column.
    """
    path = os.path.join(os.path.dirname(_STATIC), "supabase_app_schema.sql")
    with open(path, encoding="utf-8") as handle:
        sql = handle.read()
    body = sql[sql.index("function public.submit_study"):]
    for field in EXPECTED_PAYLOAD_FIELDS:
        assert "'%s'" % field in body, (
            "submit_study never reads %r, so the client is sending a field that "
            "is silently dropped" % (field,))


# --- 3. Stage list ---------------------------------------------------------

def test_the_client_spine_matches_the_servers_questionnaire_pages(tree):
    """buildSpine() concatenates the fixed stages around the server's page list.

    The questionnaire pages are pulled from the bundle at runtime, so what is
    checkable here is the FIXED part — and that it brackets the pages in the
    right places.
    """
    spine = _find_function(tree, "buildSpine")
    literals = _string_literals(spine)
    expected_prefix = ["consent", "prolific_id", "prelude", "screening",
                       "perspective", "response_intro", "response"]
    assert literals[:len(expected_prefix)] == expected_prefix
    assert literals[len(expected_prefix):] == ["debrief", "additional_thoughts", "complete"]


def test_the_client_knows_the_same_three_terminal_stages_the_bundle_carries(tree):
    import views
    terminal = None
    for node in _walk(tree):
        if (node.type == "VariableDeclarator" and getattr(node.id, "name", None) == "TERMINAL"
                and node.init.type == "ArrayExpression"):
            terminal = [element.value for element in node.init.elements]
    assert terminal is not None, "app.js has no TERMINAL list"
    assert set(terminal) == set(views.content_bundle()["terminal"])


def test_the_back_lock_stage_is_the_response_page(tree):
    """The one-way door. If this moves, the stimulus becomes editable after it is seen."""
    for node in _walk(tree):
        if (node.type == "VariableDeclarator"
                and getattr(node.id, "name", None) == "BACK_LOCK_STAGE"):
            assert node.init.value == "response"
            return
    raise AssertionError("app.js no longer defines BACK_LOCK_STAGE")


def test_the_dv_pages_the_client_special_cases_still_exist():
    """screens.js shows the response above any page whose id starts with 'dv_'."""
    dv_pages = [p for p in questionnaires.QUESTIONNAIRE_PAGE_IDS if p.startswith("dv_")]
    assert dv_pages, ("no questionnaire page starts with 'dv_' any more, so the "
                      "client would stop showing the response above the DV items")


# --- 4. The two-file split -------------------------------------------------

def test_every_stage_on_the_spine_has_a_renderer(trees):
    """A stage with no screen renders the "unexpected state" error to a participant.

    screens.js registers renderers as `APP.screens.<name> = ...`; the router
    falls back to `questionnaire` for anything in the page list and `terminal`
    for the three end states, so those are the exempt names.
    """
    registered = set()
    for node in _walk(trees["screens.js"]):
        if (node.type == "AssignmentExpression"
                and node.left.type == "MemberExpression"
                and getattr(node.left.object, "type", None) == "MemberExpression"
                and getattr(node.left.object.property, "name", None) == "screens"):
            registered.add(node.left.property.name)
    assert registered, "found no APP.screens registrations — extraction is broken"

    fixed = set(_string_literals(_find_function(trees["app.js"], "buildSpine")))
    fixed.discard("complete")  # rendered by the terminal screen
    missing = fixed - registered
    assert not missing, "stages on the spine with no renderer: %s" % (sorted(missing),)
    assert {"questionnaire", "terminal"} <= registered


def test_screens_js_only_reaches_into_app_through_the_declared_surface(trees):
    """APP is the seam between the two files; an undeclared member is a typo.

    `screens` is written to rather than read, so it is allowed alongside the
    keys app.js exports.
    """
    exported = set()
    for node in _walk(trees["app.js"]):
        if (node.type == "VariableDeclarator" and getattr(node.id, "name", None) == "APP"
                and node.init.type == "ObjectExpression"):
            exported = {prop.key.name for prop in node.init.properties}
    assert exported, "app.js no longer defines the APP object literal"

    used = set()
    for node in _walk(trees["screens.js"]):
        if (node.type == "MemberExpression"
                and getattr(node.object, "name", None) == "APP"
                and not node.computed):
            used.add(node.property.name)
    unknown = used - exported
    assert not unknown, "screens.js uses APP members app.js does not export: %s" % (
        sorted(unknown),)


# --- 5. Deployed asset paths -----------------------------------------------

def test_no_asset_path_starts_with_static(trees):
    """amplify.yml serves study/static AS the site root.

    index.html and app.js originally asked for '/static/app.js' and friends —
    correct under the old FastAPI mount, and a 404 for every asset on the first
    Amplify deploy. Paths must be relative to index.html, not rooted at /static.
    """
    for name, tree in trees.items():
        for node in _walk(tree):
            if node.type == "Literal" and isinstance(node.value, str):
                assert not node.value.startswith("/static/"), (
                    "%s references %r, which does not exist once study/static is "
                    "served as the site root" % (name, node.value))

    with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as handle:
        html = handle.read()
    # Attributes only — the comment above them legitimately mentions the old
    # path to explain why it must not come back.
    for ref in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert not ref.startswith("/static/"), (
            "index.html references %r, which 404s in production" % (ref,))


def test_every_asset_index_html_references_is_in_the_artifact_directory():
    """A src/href that resolves to nothing is a blank page for a participant.

    config.json is exempt: it is generated by the Amplify build and gitignored,
    so it is legitimately absent from a checkout.
    """
    with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as handle:
        html = handle.read()
    referenced = set(re.findall(r'(?:src|href)="([^"]+)"', html))
    for ref in referenced:
        if ref.startswith(("http://", "https://", "#", "data:")):
            continue
        assert os.path.exists(os.path.join(_STATIC, ref)), (
            "index.html references %r but %s/%s does not exist" % (ref, _STATIC, ref))


def test_the_client_fetches_only_assets_that_the_build_produces():
    """app.js fetches these by name; build_static.py must emit exactly them."""
    with open(os.path.join(_STATIC, "app.js"), encoding="utf-8") as handle:
        source = handle.read()
    fetched = set(re.findall(r"asset\('([^']+)'\)", source))
    assert fetched == {"config.json", "content.json"}, sorted(fetched)

    builder = os.path.join(os.path.dirname(_STATIC), "tools", "build_static.py")
    with open(builder, encoding="utf-8") as handle:
        build_source = handle.read()
    for name in fetched:
        assert '"%s"' % name in build_source, (
            "app.js fetches %r but build_static.py never writes it" % (name,))


# --- 6. Rendering the model's reply ----------------------------------------

def test_the_model_reply_is_rendered_through_markdown_not_raw(trees):
    """The reply must go through W.aiResponse, never straight into innerHTML.

    Qwen emits `###`, `**bold**` and lists, so plain text left the markers
    visible. But the reply is MODEL output, not study copy, so it cannot be
    trusted HTML either — aiResponse is the one path that escapes first.
    """
    source = open(os.path.join(_STATIC, "screens.js"), encoding="utf-8").read()
    assert "W.aiResponse(" in source, "screens.js no longer renders the reply via aiResponse"
    assert "className: 'ai-response', text:" not in source, \
        "screens.js renders the reply as plain text again"
    for node in _walk(trees["screens.js"]):
        if (node.type == "ObjectExpression"
                and any(getattr(p.key, "name", None) == "html" for p in node.properties)):
            for prop in node.properties:
                if getattr(prop.key, "name", None) == "html":
                    src = _string_literals(prop.value)
                    assert "shownResponse" not in "".join(src), \
                        "the model reply is being passed through `html`"


def test_markdown_escapes_before_it_transforms(trees):
    """The ordering IS the security property.

    escapeHtml must run over the raw input before any tag-producing transform,
    so a reply containing <script> renders as characters rather than executing.

    Checked against the AST, not the source text. The first version of this test
    grepped for "escapeHtml(" and was defeated by a mutation that deleted the
    call and left the name in a comment — it passed while the injection hole was
    open. Comments do not appear in the AST, so this cannot repeat.
    """
    markdown = _find_function(trees["widgets.js"], "markdown")
    param = markdown.params[0].name

    escape_calls = [n for n in _walk(markdown)
                    if n.type == "CallExpression"
                    and getattr(n.callee, "name", None) == "escapeHtml"]
    assert escape_calls, "markdown() never calls escapeHtml"

    # It must escape the PARAMETER, not some already-derived string.
    on_input = [c for c in escape_calls
                if param in _string_or_ident_names(c.arguments)]
    assert on_input, (
        "escapeHtml is called but not on markdown()'s input %r — the raw text "
        "would reach the output unescaped" % param)

    first_escape = min(c.range[0] for c in on_input)
    producers = [n for n in _walk(markdown)
                 if n.type == "Literal" and isinstance(n.value, str)
                 and "<" in n.value]
    assert producers, "markdown() produces no tags at all — extraction is broken"
    earliest = min(n.range[0] for n in producers)
    assert first_escape < earliest, (
        "markdown() builds a tag before escaping its input; model output could "
        "inject HTML")

    # inlineMarkdown must only ever be handed already-escaped text.
    inline = _find_function(trees["widgets.js"], "inlineMarkdown")
    assert not [n for n in _walk(inline)
                if n.type == "CallExpression"
                and getattr(n.callee, "name", None) == "escapeHtml"], (
        "inlineMarkdown escapes internally; escaping belongs once, up front")


def test_markdown_introduces_no_link_or_image_tags():
    """No anchors and no images from model output — each is an attack surface
    and the model has no reason to emit either."""
    source = open(os.path.join(_STATIC, "widgets.js"), encoding="utf-8").read()
    body = source[source.index("function markdown("):source.index("function aiResponse(")]
    for tag in ("<a ", "<a>", "<img", "href", "src="):
        assert tag not in body, "markdown() emits %r" % tag


# --- 7. The Likert header ---------------------------------------------------

def test_numbered_scales_stack_the_word_above_the_number(trees):
    """Numbers must share a line and columns must be equal width.

    Only the endpoints and midpoint carry an anchor word, so a single-line
    header made the labelled columns wider, pushed the numbers to different
    heights, and left the radio buttons unevenly spaced.
    """
    source = open(os.path.join(_STATIC, "widgets.js"), encoding="utf-8").read()
    assert "matrix-head-word" in source and "matrix-head-number" in source, \
        "optionHeader no longer builds a two-row header"
    assert "option.label + '\\n' + option.sublabel" not in source, \
        "the single-line header is back"

    css = open(os.path.join(_STATIC, "styles.css"), encoding="utf-8").read()
    assert "table-layout: fixed" in css, \
        "without a fixed table layout the option columns are not equal width"
    assert ".matrix-head-word" in css and "min-height" in css, \
        "the word row needs a reserved height or the numbers will not align"


def test_every_numbered_scale_in_the_instrument_has_the_shape_the_header_assumes():
    """optionHeader treats `sublabel` as the number and `label` as the anchor
    word. Verify that actually holds for every question that uses one."""
    import questionnaires
    for question in questionnaires.all_questions():
        options = question["options"]
        if not any(o.get("sublabel") for o in options):
            continue
        for option in options:
            if option.get("sublabel"):
                assert option["sublabel"].strip().isdigit(), (
                    "%s: sublabel %r is not the number the header renders"
                    % (question["id"], option["sublabel"]))
            else:
                assert str(option["label"]).strip().isdigit(), (
                    "%s: option %r has no sublabel, so its label must BE the "
                    "number" % (question["id"], option["label"]))


# --- 8. The completion code -------------------------------------------------

def test_the_terminal_page_shows_a_code_and_never_redirects(trees):
    """A redirect that is slow or blocked strands a participant who finished."""
    source = open(os.path.join(_STATIC, "screens.js"), encoding="utf-8").read()
    assert "completion-code" in source, "the terminal page no longer shows a code"
    for banned in ("window.location.href", "redirect_url", "Return to Prolific",
                   "prolific.com"):
        assert banned not in source, "screens.js still redirects or links out: %r" % banned
    app = open(os.path.join(_STATIC, "app.js"), encoding="utf-8").read()
    assert "redirect_url" not in app and "prolific.com" not in app


def test_the_code_is_shown_only_after_the_submit_succeeds():
    """A participant holding the code must, by construction, be recorded."""
    source = open(os.path.join(_STATIC, "screens.js"), encoding="utf-8").read()
    terminal = source[source.index("APP.screens.terminal"):]
    finish_at = terminal.index("function finish()")
    submit_at = terminal.index("submit_study")
    assert finish_at < submit_at, "expected finish() defined above the submit call"
    # finish() must be reached only from the resolve path or the unauthorized
    # branch, never called unconditionally before submitting.
    before_submit = terminal[:submit_at]
    assert "finish();" not in before_submit, \
        "the code is shown before the submit is attempted"


def test_the_bundle_carries_a_code_wherever_one_is_owed():
    """Completers and screen-outs are owed a code; a decliner is not — they did
    not take part. test_views.py owns the decliner case in detail."""
    import views
    terminal = views.content_bundle()["terminal"]
    assert set(terminal) == {"complete", "screened_out", "declined"}
    for name in ("complete", "screened_out"):
        assert terminal[name].get("code"), "%s has no completion code" % name
    assert "code" not in terminal["declined"]
    for name, entry in terminal.items():
        assert "redirect_url" not in entry, "%s still carries a redirect URL" % name
        assert "redirected" not in entry["message"].lower(), (
            "%s still promises a redirect that no longer happens" % name)


def test_the_card_widens_for_the_response_and_its_questions(trees):
    """720px is right for prose and wrong for a 10-column Likert grid.

    The CARD widens; the grid stays inside it. An earlier version broke the
    table out with a full-bleed transform, which widened the table but left it
    overhanging the card's white background.

    The wide set is the response page plus every questionnaire page — whether or
    not that page happens to contain a grid — so the card does not resize
    between consecutive questions about the same reply.
    """
    css = open(os.path.join(_STATIC, "styles.css"), encoding="utf-8").read()
    assert "--maxw-wide" in css and ".page.wide" in css, \
        "no wide-card mode defined"

    wrap = css[css.index(".matrix-wrap {"):]
    wrap = wrap[:wrap.index("}")]
    for escaped in ("transform:", "position: relative", "100vw"):
        assert escaped not in wrap, (
            "the grid is breaking out of the card again (%r); widen the card "
            "instead" % escaped)

    source = open(os.path.join(_STATIC, "app.js"), encoding="utf-8").read()
    assert "'wide'" in source, "nothing applies the wide class"

    # Checked against the AST, not the text. A substring version of this test
    # was defeated by a mutation that commented the body out and left the
    # identifier behind in the comment — green while the behaviour was gone.
    wide = _find_function(trees["app.js"], "isWideStage")
    returns = [n for n in _walk(wide) if n.type == "ReturnStatement"]
    assert returns, "isWideStage returns nothing"
    assert not any(r.argument is not None and r.argument.type == "Literal"
                   and r.argument.value is False for r in returns), \
        "isWideStage has a `return false` short-circuit"

    literals = _string_literals(wide)
    assert "response" in literals, "the response page must use the same width"
    members = {n.property.name for n in _walk(wide)
               if n.type == "MemberExpression" and not n.computed}
    assert "questionnaire_pages" in members, (
        "the wide set must be derived from the questionnaire page list, not "
        "just the pages that happen to contain a grid, or the card resizes "
        "between consecutive DV pages")


def test_the_reply_stays_readable_inside_the_wide_card():
    """A 1180px line length is unreadable and the participant has to read it."""
    css = open(os.path.join(_STATIC, "styles.css"), encoding="utf-8").read()
    assert ".page.wide .ai-response" in css, \
        "the reply is not constrained inside the wide card"
    rule = css[css.index(".page.wide .ai-response"):]
    rule = rule[:rule.index("}")]
    assert "max-width" in rule


def test_the_widest_scale_still_gets_usable_header_room():
    """Guards against someone adding a scale with more options than fits.

    Ten columns at the 1140px cap is about 78px each, which holds the longest
    anchor word in the instrument. Many more than that and the overlap returns.
    """
    import questionnaires
    for question in questionnaires.all_questions():
        if question["type"] != "matrix":
            continue
        columns = len(question["options"])
        # Inside the wide card: 1180 page cap, less page and card padding.
        per_column = (1180 - 40 - 64) * 0.70 / columns
        assert per_column >= 60, (
            "%s has %d option columns, leaving %.0fpx per header — the anchor "
            "words will overlap again" % (question["id"], columns, per_column))


def test_the_terminal_page_handles_a_missing_code():
    """The bundle omits `code` for a decliner, so the page must branch on it
    rather than render an empty code box."""
    source = open(os.path.join(_STATIC, "screens.js"), encoding="utf-8").read()
    finish = source[source.index("function finish()"):]
    finish = finish[:finish.index("\n    function ")]
    assert "if (conf.code)" in finish, (
        "finish() renders the code unconditionally; a decliner would get an "
        "empty box")
    assert "no_code_instruction" in finish, (
        "with no code, a decliner must be told what to do instead")


def test_the_prolific_id_is_required(trees):
    """A submission with no Prolific ID cannot be matched to a participant, so
    it is neither payable nor usable.

    The source survey used Qualtrics' soft "request response" and accepted a
    blank; this does not. Enforced client-side, like every other required
    answer — the server sees the payload only at the end, and rejecting a
    finished session over it would be worse than the gap.
    """
    screen = _find_function_expression(trees["screens.js"], "prolific_id")
    assert screen is not None, "no prolific_id screen"
    # A guard clause that returns early on a blank value must exist.
    returns = [n for n in _walk(screen)
               if n.type == "ReturnStatement" and n.argument is None]
    assert returns, "prolific_id advances unconditionally; the field is not required"
    members = {n.property.name for n in _walk(screen)
               if n.type == "MemberExpression" and not n.computed}
    assert "required_error" in members, \
        "no error message is shown when the field is left blank"


def test_the_bundle_carries_the_prolific_required_message():
    import views
    assert views.content_bundle()["prolific"].get("required_error"), \
        "the client needs a message to show when the ID is blank"
