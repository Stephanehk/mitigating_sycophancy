/* DEBUG ONLY -- names the arm behind each response on screen.
 *
 * LOADED ONLY BY `?labels=1`. app.js injects a script tag for this file when the
 * flag is in the URL and never otherwise, so the plain rating URL a rater opens
 * requests neither this nor the mapping, and renders exactly as it did before the
 * flag existed. It is published (by tools/build_static.py) but not reached.
 *
 * IT IS STILL BOLTED ON FROM OUTSIDE, and that is deliberate. app.js renders `a`
 * and `b` and sends a screen side; it has no notion of an arm, and the published
 * items.json carries no arm names. Teaching the renderer about arms would make
 * every code path a labelled-or-not path. Instead this watches the DOM app.js
 * produced and adds a bar above it, so there is one render, one item set, and one
 * thing that knows the mapping -- this file.
 *
 * WHERE THE MAPPING COMES FROM. Two halves, neither of which app.js knows:
 *   - which arms an item's `a`/`b` are: from __debug_arms.json, built by
 *     arms_map() in tools/debug_arms.py out of the FULL data/items.json, and
 *     served both by tools/local_server.py and from the deployed site;
 *   - which of the two is on the left: from `flipped` in the session the page
 *     stored in sessionStorage, i.e. the same per-session draw the database made.
 * Composing them here reproduces exactly what pref_record does server-side --
 * which is why a label can be trusted to name the arm the judgment is recorded
 * against, rather than merely agreeing with itself.
 *
 * `window.__DEBUG_ARMS`, if something has already set it, wins over the fetch.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'pref_session';
  var arms = null;

  var style = document.createElement('style');
  style.textContent = [
    '.debug-arms {',
    '  display: flex; gap: 16px; justify-content: space-between;',
    '  font: 600 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;',
    '  letter-spacing: 0.02em; margin: 0 0 6px;',
    '}',
    '.debug-arms span {',
    '  flex: 1 1 0; text-align: center; color: #fff; background: #b5522b;',
    '  border-radius: 6px; padding: 3px 8px;',
    '}'
  ].join('\n');
  document.head.append(style);

  /* The item at the position currently on screen, with its side draw.
   *
   * The position is read off the counter rather than tracked, because app.js owns
   * the index and does not publish it -- and the counter is the same number, one
   * indexed. Returns null on any screen that is not a rating screen. */
  function current() {
    var counter = document.querySelector('.counter');
    var pair = document.querySelector('.pair');
    if (!counter || !pair || !arms) { return null; }

    var position = parseInt(counter.textContent, 10) - 1;
    var raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw || !(position >= 0)) { return null; }

    var assignment = JSON.parse(raw).items[position];
    if (!assignment) { return null; }
    var pairing = arms[assignment.item_id];
    if (!pairing) { return null; }

    return {
      pair: pair,
      left: assignment.flipped ? pairing.b : pairing.a,
      right: assignment.flipped ? pairing.a : pairing.b
    };
  }

  function label() {
    var shown = current();
    if (!shown || document.querySelector('.debug-arms')) { return; }
    var bar = document.createElement('div');
    bar.className = 'debug-arms';
    [shown.left, shown.right].forEach(function (name) {
      var tag = document.createElement('span');
      tag.textContent = name;
      bar.append(tag);
    });
    /* Immediately above the two panels, so each name sits over its own column --
       and outside `.pair`, which is a two-column grid that would take a third
       child as a grid item. */
    shown.pair.parentNode.insertBefore(bar, shown.pair);
  }

  /* app.js rebuilds #root on every page turn, which drops the bar with it. */
  new MutationObserver(label).observe(document.getElementById('root'),
                                      { childList: true, subtree: true });

  /* Beside index.html on the deployed site, and a route on the local stub. */
  if (window.__DEBUG_ARMS) {
    arms = window.__DEBUG_ARMS;
    label();
  } else {
    fetch('__debug_arms.json').then(function (response) {
      return response.json();
    }).then(function (loaded) {
      arms = loaded;
      label();
    });
  }
}());
