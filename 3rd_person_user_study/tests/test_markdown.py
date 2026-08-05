"""Tests for the response-panel markdown rendering.

Two halves, and both matter for different reasons.

1. THE SUBSET IS SUFFICIENT. static/markdown.js handles the constructs the frozen
   item set actually contains, not all of CommonMark. That is only safe while the
   claim holds, so `test_items_contain_no_unsupported_construct` scans every
   prompt and response for anything the renderer would pass through as literal
   text. Rebuild the item set from responses containing a table and this fails,
   instead of a rater seeing `| a | b |`.

2. THE RENDERER IS CORRECT. The rest run the real file in real Chromium and
   assert on the resulting DOM -- including that markup in the model's output
   stays inert. A regex-level test of the source could not show that.
"""

import json
import os
import re

import pytest

import config

_HERE = os.path.dirname(os.path.abspath(__file__))
MARKDOWN_JS = os.path.join(config.STATIC_DIR, "markdown.js")

# Constructs static/markdown.js does NOT implement. Each would render as its own
# literal source text, which is a silent, arm-correlated presentation artifact.
UNSUPPORTED = {
    "fenced code block": r"```",
    "pipe table": r"(?m)^\s*\|.*\|\s*$",
    # A setext heading is text UNDERLINED by --- or ===. A bare rule on its own,
    # after a blank line, is a horizontal rule and IS supported -- so the
    # preceding non-blank line is the whole distinction, and matching the
    # underline alone would flag all 126 real rules.
    "setext heading": r"(?m)^[^\n\S]*\S[^\n]*\n {0,3}(?:={3,}|-{3,})[^\n\S]*$",
    "reference link": r"(?m)^\s*\[[^\]]+\]:\s*\S+",
    "image": r"!\[[^\]]*\]\(",
    "raw html tag": r"<(?:/?[a-zA-Z][a-zA-Z0-9]*)(?:\s[^>\n]*)?>",
    # Entities the renderer does NOT decode. It handles the XML predefined five
    # plus &nbsp;, which is everything the item set contains; anything else would
    # show as its own source text.
    "undecoded html entity": r"&(?!amp;|lt;|gt;|quot;|apos;|nbsp;|#39;)"
                             r"(?:[a-zA-Z]+|#\d+);",
    "autolink": r"<https?://[^>\s]+>",
    "footnote": r"\[\^[^\]]+\]",
    "task list": r"(?m)^\s*[-*+] \[[ xX]\]",
    "highlight": r"==[^=\n]+==",
}

# An indented code block needs 4+ spaces on a line that is NOT list content. Every
# such line in the item set so far is a nested bullet, which the renderer handles,
# so the check is written to tell the two apart rather than to flag both.
LIST_MARKER = re.compile(r"^ *(?:[-*+]|\d+[.)]) ")
INDENTED = re.compile(r"^(?: {4,}|\t)\S")


@pytest.fixture(scope="module")
def all_text(bundle):
    """Every string a rater will see rendered."""
    strings = []
    for item in bundle["items"]:
        strings.extend([item["prompt"], item["response_a"], item["response_b"]])
    return strings


def test_items_contain_no_unsupported_construct(all_text):
    found = {}
    for name, pattern in UNSUPPORTED.items():
        hits = [text for text in all_text if re.search(pattern, text)]
        if hits:
            found[name] = len(hits)
    assert not found, (
        f"item text uses markdown the renderer does not handle: {found}. "
        "Either extend static/markdown.js or accept literal source text on screen.")


def test_no_indented_block_is_outside_a_list(all_text):
    """Separated from the scan above because the two cases look identical to a
    regex: a 4-space-indented nested bullet renders correctly, a 4-space-indented
    paragraph would render as a code block (i.e. as its own source)."""
    offenders = []
    for text in all_text:
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if not INDENTED.match(line) or LIST_MARKER.match(line):
                continue
            previous = next((lines[back] for back in range(index - 1, -1, -1)
                             if lines[back].strip()), "")
            if not LIST_MARKER.match(previous):
                offenders.append(line[:60])
    assert not offenders, f"indented non-list lines would render as code: {offenders}"


def test_entity_decoding_covers_what_the_items_contain(all_text):
    """The renderer decodes a fixed set. This asserts the set is the right one for
    this corpus rather than a guess."""
    found = set()
    for text in all_text:
        found.update(re.findall(r"&(?:[a-zA-Z]+|#\d+);", text))
    assert found <= {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;", "&#39;",
                     "&nbsp;"}, found
    assert found, "no entities at all: the decoding path is untested by real data"


#: Constructs the renderer handles, and how to spot one in the item text.
CONSTRUCTS = {
    "heading": r"(?m)^ {0,3}#{1,6} ",
    "bold": r"\*\*\S",
    "bullet list": r"(?m)^ *[-*+] ",
    "ordered list": r"(?m)^ *\d+[.)] ",
    "rule": r"(?m)^ {0,3}-{3,}\s*$",
    "strikethrough": r"~~\S",
}

#: Handled by markdown.js but ABSENT from the current item set, so the renderer's
#: behaviour on them is exercised only by the unit tests below and not by real
#: data. Recorded rather than dropped: the support stays (a later item set may
#: contain it), and test_absent_constructs_are_still_absent fails if one shows up,
#: which is the signal to move it back above.
#:
#: strikethrough left when the item set moved to the voice2 `finaldata` series
#: (2026-08-05); the previous `mixed_P6_perspshift_distill_gp` items had it.
NOT_IN_CORPUS = {"strikethrough"}


def test_the_supported_constructs_are_actually_present(all_text):
    """The mirror of the test above: if none of the handled constructs appear, the
    renderer is untested by the real data and the subset claim is vacuous."""
    for name, pattern in CONSTRUCTS.items():
        if name in NOT_IN_CORPUS:
            continue
        assert any(re.search(pattern, text) for text in all_text), name


def test_absent_constructs_are_still_absent(all_text):
    """NOT_IN_CORPUS is a claim about the data, so it has to be checked. A
    construct that reappears must go back under the test above rather than sit
    in a list saying the data does not contain it."""
    for name in NOT_IN_CORPUS:
        assert not any(re.search(CONSTRUCTS[name], text) for text in all_text), (
            f"{name} IS in the item set now -- remove it from NOT_IN_CORPUS so "
            "the renderer is held to it")


# --- the renderer, in a real browser ---------------------------------------

@pytest.fixture(scope="module")
def render():
    """MD.render, callable from Python, returning the resulting innerHTML."""
    playwright = pytest.importorskip("playwright.sync_api")
    with open(MARKDOWN_JS, encoding="utf-8") as handle:
        source = handle.read()
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch()
        page = browser.new_page()
        page.set_content("<div id=out></div>")
        page.add_script_tag(content=source)

        def call(markdown):
            return page.evaluate(
                """(text) => {
                    const out = document.getElementById('out');
                    out.replaceChildren(window.MD.render(text));
                    return out.innerHTML;
                }""", markdown)

        yield call
        browser.close()


def test_headings_render_at_their_level(render):
    assert render("## Two") == "<h2>Two</h2>"
    assert render("#### Four") == "<h4>Four</h4>"
    assert "<h3><strong>Lead</strong></h3>" == render("### **Lead**")


def test_emphasis(render):
    assert render("a **b** c") == "<p>a <strong>b</strong> c</p>"
    assert render("a *b* c") == "<p>a <em>b</em> c</p>"


def test_underscores_inside_words_are_left_alone(render):
    """The one departure from CommonMark that could bite: word-internal
    underscores are common in these responses (URLs), intentional emphasis is
    not."""
    assert render("found_out_that_x") == "<p>found_out_that_x</p>"


def test_backslash_escapes_do_not_open_emphasis(render):
    assert render(r"a \*not em\* b") == "<p>a *not em* b</p>"
    assert render(r"x\_y\_z") == "<p>x_y_z</p>"


def test_lists(render):
    assert render("- a\n- b") == "<ul><li>a</li><li>b</li></ul>"
    assert render("1. a\n2. b") == "<ol><li>a</li><li>b</li></ol>"


def test_nested_lists_hang_off_their_parent_item(render):
    assert render("- a\n  - b\n- c") == (
        "<ul><li>a<ul><li>b</li></ul></li><li>c</li></ul>")


def test_a_wrapped_list_item_stays_in_its_item(render):
    """The model wraps long bullets. Treating the continuation as a new paragraph
    would break the list in half."""
    assert render("- first line\n  continued\n- second") == (
        "<ul><li>first line continued</li><li>second</li></ul>")


def test_strikethrough(render):
    assert render("I ~~forgot~~ told him") == "<p>I <s>forgot</s> told him</p>"
    assert render("a ~ b") == "<p>a ~ b</p>"          # a lone tilde is literal


def test_rules_and_blockquotes(render):
    assert render("---") == "<hr>"
    assert render("> quoted\n> more") == "<blockquote><p>quoted<br>more</p></blockquote>"


def test_inline_code(render):
    assert render("use `x = 1` here") == "<p>use <code>x = 1</code> here</p>"


def test_bold_inside_backticks_stays_literal(render):
    assert render("`**raw**`") == "<p><code>**raw**</code></p>"


def test_a_single_newline_becomes_a_line_break(render):
    assert render("one\ntwo") == "<p>one<br>two</p>"
    assert render("one\n\ntwo") == "<p>one</p><p>two</p>"


def test_unmatched_emphasis_is_left_as_text(render):
    """39 of the item strings have an odd number of `**`. Leaving the stray marker
    literal is wrong-looking; swallowing the rest of the response is worse."""
    assert render("a ** b") == "<p>a ** b</p>"


def test_markup_in_the_model_output_is_inert(render):
    """The renderer never touches innerHTML and only creates a fixed set of
    elements, so this is structural, not a filter that could be bypassed."""
    html = render("<script>alert(1)</script> <img src=x onerror=y>")
    assert "<script" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html


def test_entities_are_decoded_to_characters_not_markup(render):
    assert render("Tom &amp; Jerry") == "<p>Tom &amp; Jerry</p>"     # the CHARACTER &
    assert render("5 &lt; 6") == "<p>5 &lt; 6</p>"                   # the CHARACTER <
    # One pass only, so a double-escaped entity decodes one level and stops.
    assert render("&amp;lt;") == "<p>&amp;lt;</p>"
    # And a decoded character can never have become a tag.
    assert render("&lt;script&gt;alert(1)&lt;/script&gt;").count("<script") == 0


def test_links_render_as_text_and_are_not_clickable(render):
    """A rater who follows a link loses the session."""
    html = render("see [the post](https://reddit.com/r/x)")
    assert "<a " not in html and "href" not in html
    assert "the post" in html


def test_link_text_is_itself_processed(render):
    r"""The item set's one link is a URL whose text carries `\_` escapes. Emitting
    the text verbatim put the backslashes on screen."""
    assert render(r"[a\_b\_c](http://x)") == "<p>a_b_c</p>"
    assert render("[**bold link**](http://x)") == "<p><strong>bold link</strong></p>"


def test_recursing_into_link_text_does_not_disturb_the_outer_scan(render):
    """The regression the recursion could have caused: a shared global regex would
    carry lastIndex out of the inner call and the outer loop would resume at the
    wrong offset, silently dropping the tail of the line."""
    assert render(r"start [a\_b](http://x) middle **end**") == (
        "<p>start a_b middle <strong>end</strong></p>")
    assert render("[one](http://x) and [two](http://y) and *three*") == (
        "<p>one and two and <em>three</em></p>")


def test_every_item_renders_without_error(render, all_text):
    """The real corpus, not just the fixtures: 360 strings through the renderer,
    each producing something non-empty."""
    for text in all_text:
        html = render(text)
        assert html.strip(), f"empty render for: {text[:60]!r}"
        assert "<script" not in html.lower()
