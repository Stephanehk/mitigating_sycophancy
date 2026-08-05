/* A deliberately small Markdown renderer for the response panels.
 *
 * WHY NOT A LIBRARY. There is no build step and no package manager here, so a
 * library means a vendored blob to audit and keep. The item set is frozen and its
 * markdown was surveyed: ATX headings (## through #####), `**bold**`, `*em*`,
 * `~~strike~~`, unordered and ordered lists including nesting, `---` rules, `>`
 * blockquotes, one inline-code span, one link, and `&amp;`/`&lt;` entities. No
 * fenced code, no tables, no reference links, no images, no setext headings, no
 * raw HTML, no task lists.
 *
 * That survey is the ASSUMPTION, and it is checked rather than trusted:
 * tests/test_markdown.py fails if the committed items contain a construct this
 * file does not handle. Rebuild the item set with a table in it and the test
 * tells you, instead of a rater seeing `| a | b |` on screen.
 *
 * IT RETURNS DOM NODES, NOT AN HTML STRING, and nothing here ever touches
 * innerHTML. All text goes in through textContent, and only a fixed set of
 * element names is ever created, so model output cannot inject markup no matter
 * what it contains -- the escaping bug class is absent rather than handled.
 *
 * Two deliberate departures from CommonMark:
 *   - `_underscore_` is NOT emphasis. It appears once in the whole item set, and
 *     inside a URL, where treating it as emphasis is simply wrong. Word-internal
 *     underscores are far more common than intentional ones here.
 *   - a single newline inside a paragraph becomes <br>, not a space. These are
 *     chat responses; the line breaks the model produced are intended.
 */
window.MD = (function () {
  'use strict';

  var HR = /^ {0,3}(-{3,}|\*{3,}|_{3,})\s*$/;
  var HEADING = /^ {0,3}(#{1,6}) +(.*)$/;
  var QUOTE = /^ {0,3}> ?(.*)$/;
  var ITEM = /^( *)(?:([-*+])|(\d+)[.)]) +(.*)$/;

  /* One pass over a line's inline markup. Alternatives are ordered so the
   * greedier, more specific ones win: code before emphasis (so `**` inside
   * backticks stays literal), the backslash escape before everything (so `\*`
   * cannot open emphasis), bold before italic (so `**x**` is not read as an
   * empty italic followed by `x`). */
  /* A SOURCE STRING, not a shared /g/ regex object. inlineNodes recurses into link
   * text, and a shared global regex carries `lastIndex` across calls -- the inner
   * call would leave it pointing into the inner string and the outer loop would
   * resume from the wrong offset. A fresh instance per call has no such state. */
  var INLINE_SOURCE = "`([^`\\n]+)`"
                    + "|\\\\([^\\n])"
                    + "|\\*\\*(\\S(?:[^\\n]*?\\S)?)\\*\\*"
                    + "|~~(\\S(?:[^\\n]*?\\S)?)~~"
                    + "|\\*(\\S(?:[^\\n]*?\\S)?)\\*"
                    + "|\\[([^\\]\\n]+)\\]\\([^)\\n]*\\)";

  /* The item set contains literal `&amp;` (8 strings) and `&lt;` (1) -- an
   * artifact of whatever pipeline produced the source text, not something an
   * author typed. Decoding them is safe here in a way it would not be in an
   * HTML-string renderer: the result goes into a text node, so `&lt;` becomes the
   * CHARACTER `<` and can never become a tag.
   *
   * ONE PASS, so `&amp;lt;` decodes to the text `&lt;` and not to `<`. Decoding
   * happens at node creation, i.e. AFTER inline markup is tokenized, so a
   * numeric entity cannot smuggle in an emphasis marker. */
  var ENTITIES = { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
                   '&apos;': "'", '&#39;': "'", '&nbsp;': ' ' };
  var ENTITY = /&(?:amp|lt|gt|quot|apos|nbsp|#39);/g;

  function decode(value) {
    return String(value).replace(ENTITY, function (match) {
      return ENTITIES[match];
    });
  }

  function textNode(value) {
    return document.createTextNode(decode(value));
  }

  function wrap(tag, value) {
    var node = document.createElement(tag);
    node.textContent = decode(value);
    return node;
  }

  /* Inline markup of one line, as an array of nodes.
   *
   * A link renders as its TEXT and is not made clickable: a rater who navigates
   * away mid-task loses the session, and there is nothing in these responses
   * worth following. */
  function inlineNodes(line) {
    var nodes = [];
    var cursor = 0;
    var pattern = new RegExp(INLINE_SOURCE, 'g');
    var match;
    while ((match = pattern.exec(line)) !== null) {
      if (match.index > cursor) {
        nodes.push(textNode(line.slice(cursor, match.index)));
      }
      if (match[1] !== undefined) {
        nodes.push(wrap('code', match[1]));
      } else if (match[2] !== undefined) {
        nodes.push(textNode(match[2]));          // \x -> x, literally
      } else if (match[3] !== undefined) {
        nodes.push(wrap('strong', match[3]));
      } else if (match[4] !== undefined) {
        // `~~forgot~~` -- one occurrence, in a prompt copied off Reddit, where the
        // author plainly meant it struck through rather than tilde-wrapped.
        nodes.push(wrap('s', match[4]));
      } else if (match[5] !== undefined) {
        nodes.push(wrap('em', match[5]));
      } else {
        /* Link TEXT only -- the href is dropped, because a rater who navigates
         * away mid-task loses the session and there is nothing in these responses
         * worth following.
         *
         * Processed RECURSIVELY, not emitted as a raw text node. The one link in
         * the item set is a Reddit URL whose text carries `\_` escapes, and
         * emitting it verbatim put the backslashes on screen. Terminates because
         * the inner text is strictly shorter than the match. */
        inlineNodes(match[6]).forEach(function (child) { nodes.push(child); });
      }
      cursor = match.index + match[0].length;
    }
    if (cursor < line.length) { nodes.push(textNode(line.slice(cursor))); }
    return nodes;
  }

  function fill(node, lines) {
    lines.forEach(function (line, index) {
      if (index) { node.append(document.createElement('br')); }
      inlineNodes(line).forEach(function (child) { node.append(child); });
    });
    return node;
  }

  function isBlank(line) { return !line.trim(); }

  function isBlockStart(line) {
    return HR.test(line) || HEADING.test(line) || QUOTE.test(line)
        || ITEM.test(line);
  }

  /* A run of list items, with nesting driven by leading indent.
   *
   * A non-blank, indented line that is not itself an item is a WRAPPED item --
   * appended to the open <li> rather than ending the list, which is what the
   * model's own line wrapping produces. */
  function parseList(lines, start, parent) {
    var stack = [];
    var index = start;
    while (index < lines.length) {
      var match = ITEM.exec(lines[index]);
      if (!match) {
        var open = stack.length && stack[stack.length - 1].list.lastElementChild;
        if (open && !isBlank(lines[index]) && !isBlockStart(lines[index])
            && /^ {2,}/.test(lines[index])) {
          open.append(textNode(' '));
          inlineNodes(lines[index].trim()).forEach(function (child) {
            open.append(child);
          });
          index += 1;
          continue;
        }
        break;
      }
      var indent = match[1].length;
      var ordered = match[3] !== undefined;

      while (stack.length > 1 && indent < stack[stack.length - 1].indent) {
        stack.pop();
      }
      var top = stack.length ? stack[stack.length - 1] : null;
      if (!top || indent > top.indent) {
        var list = document.createElement(ordered ? 'ol' : 'ul');
        // Nest under the item it belongs to; fall back to the list itself when a
        // list opens already indented and there is no item to hang it on.
        (top ? (top.list.lastElementChild || top.list) : parent).append(list);
        stack.push({ indent: indent, list: list, ordered: ordered });
      } else if (top.ordered !== ordered) {
        // Same level, different marker: a sibling list, not a continuation.
        var sibling = document.createElement(ordered ? 'ol' : 'ul');
        (top.list.parentNode || parent).append(sibling);
        stack[stack.length - 1] = { indent: indent, list: sibling,
                                    ordered: ordered };
      }
      stack[stack.length - 1].list.append(fill(document.createElement('li'),
                                               [match[4]]));
      index += 1;
    }
    return index;
  }

  /* Consecutive lines matching `test`, collected and mapped through `strip`. */
  function run(lines, start, test, strip) {
    var collected = [];
    var index = start;
    while (index < lines.length && test(lines[index])) {
      collected.push(strip(lines[index]));
      index += 1;
    }
    return { lines: collected, next: index };
  }

  function render(source) {
    var fragment = document.createDocumentFragment();
    var lines = String(source).split('\n');
    var index = 0;
    while (index < lines.length) {
      var line = lines[index];
      if (isBlank(line)) { index += 1; continue; }

      if (HR.test(line)) {
        fragment.append(document.createElement('hr'));
        index += 1;
      } else if (HEADING.test(line)) {
        var heading = HEADING.exec(line);
        fragment.append(fill(document.createElement('h' + heading[1].length),
                             [heading[2]]));
        index += 1;
      } else if (QUOTE.test(line)) {
        var quoted = run(lines, index, function (candidate) {
          return QUOTE.test(candidate);
        }, function (candidate) { return QUOTE.exec(candidate)[1]; });
        var quote = document.createElement('blockquote');
        quote.append(fill(document.createElement('p'), quoted.lines));
        fragment.append(quote);
        index = quoted.next;
      } else if (ITEM.test(line)) {
        index = parseList(lines, index, fragment);
      } else {
        var paragraph = run(lines, index, function (candidate) {
          return !isBlank(candidate) && !isBlockStart(candidate);
        }, function (candidate) { return candidate; });
        fragment.append(fill(document.createElement('p'), paragraph.lines));
        index = paragraph.next;
      }
    }
    return fragment;
  }

  return { render: render, inlineNodes: inlineNodes };
}());
