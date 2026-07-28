"""Contract tests between static/app.js and the Python server.

The client owns the flow now, which means a whole class of bug that used to be
impossible is suddenly possible: the browser can drift from the server without
anything failing until a participant hits it in production. Nothing else in this
suite executes app.js, so these tests parse it and assert the three couplings
that would break silently:

  1. Every endpoint the client calls exists on the server.
  2. The payload the client builds has exactly the fields the server's validator
     reads — no missing field, no field invented and then dropped.
  3. The client's stage list matches the questionnaire pages the server sends.

esprima parses the file rather than a regex reading it, so a renamed function or
a reformatted literal cannot make a check silently vacuous.
"""

import os
import sys

import esprima
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import questionnaires
import server

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


def _string_literals(node):
    """Every string literal under `node`, in SOURCE order."""
    found = [n for n in _walk(node)
             if n.type == "Literal" and isinstance(n.value, str)]
    found.sort(key=lambda n: n.range[0])
    return [n.value for n in found]


# --- 1. Endpoints ----------------------------------------------------------

def _api_calls(tree):
    """Endpoint literals passed to api(...) or APP.api(...)."""
    called = set()
    for node in _walk(tree):
        if node.type != "CallExpression" or not node.arguments:
            continue
        callee = node.callee
        name = getattr(callee, "name", None)
        if name is None and callee.type == "MemberExpression":
            name = getattr(callee.property, "name", None)
        if name == "api" and node.arguments[0].type == "Literal":
            called.add(node.arguments[0].value)
    return called


def test_every_endpoint_the_client_calls_exists_on_the_server(trees):
    called = set()
    for tree in trees.values():
        called |= _api_calls(tree)
    assert called, "found no api() calls — the extraction is broken, not the client"

    served = {route.path for route in server.app.routes}
    missing = called - served
    assert not missing, "client calls endpoints the server does not serve: %s" % (
        sorted(missing),)


def test_the_client_never_calls_a_worker_endpoint(trees):
    """Those are authenticated by a shared secret that must never reach a browser."""
    for name, tree in trees.items():
        for node in _walk(tree):
            if node.type == "Literal" and isinstance(node.value, str):
                assert not node.value.startswith("/api/worker"), \
                    "%s references a worker endpoint: %r" % (name, node.value)


def test_the_client_touches_the_server_at_only_five_places(trees):
    """The whole point of the redesign. A sixth call means a page turn regressed."""
    called = set()
    for tree in trees.values():
        called |= _api_calls(tree)
    assert called == {"/api/session", "/api/response/start", "/api/response/status",
                      "/api/response/retry", "/api/submit"}


def test_only_the_three_expected_screens_talk_to_the_server(trees):
    """Every OTHER page turn must be local, which is the latency fix itself."""
    calls = _api_calls(trees["screens.js"])
    assert calls == {"/api/response/start", "/api/response/status",
                     "/api/response/retry", "/api/submit"}, (
        "a screen gained or lost a server call: %s" % (sorted(calls),))
    assert _api_calls(trees["widgets.js"]) == set(), \
        "widgets.js must stay pure DOM — a widget that fetches is a page-turn stall"


# --- 2. Payload shape ------------------------------------------------------

#: The fields flow.validate_submission reads out of the client's payload.
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


def test_the_validator_actually_reads_every_field_the_client_sends():
    """Guards the other direction: a field the client sends but nobody reads."""
    import inspect

    import flow
    source = inspect.getsource(flow.validate_submission)
    source += inspect.getsource(flow.validate_screening_answers)
    for field in EXPECTED_PAYLOAD_FIELDS:
        assert '"%s"' % field in source, (
            "flow.validate_submission never reads %r, so the client is sending "
            "a field that is silently dropped" % (field,))


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
