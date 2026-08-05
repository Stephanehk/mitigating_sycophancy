/* LOCAL DEBUG ONLY -- names the arm behind each response on screen.
 *
 * THIS FILE IS NOT PART OF THE PAGE. It lives in tools/, never in static/, and is
 * served only by tools/local_server.py when it is run with --label-arms, which
 * injects a script tag for it into index.html on the way out. Nothing in static/
 * references it and no deployment can pick it up.
 *
 * That indirection is the point. The rating page is blind by construction:
 * static/items.json is published with the responses keyed `a`/`b` and no arm
 * names, app.js is tested to never mention an arm, and the database resolves a
 * screen side to a model itself. Adding a labelled mode to app.js would put the
 * arm names into the deployed bundle and make the blindness a runtime flag rather
 * than a property of what is published. So the labels are bolted on from outside
 * instead, and the file a rater is served stays the blind one.
 *
 * WHERE THE MAPPING COMES FROM. Two halves, neither of which app.js knows:
 *   - which arms an item's `a`/`b` are: from the stub's /__debug_arms.json, which
 *     it reads out of the FULL data/items.json;
 *   - which of the two is on the left: from `flipped` in the session the page
 *     stored in sessionStorage, i.e. the same per-session draw the database made.
 * Composing them here reproduces exactly what pref_record does server-side.
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

  fetch('__debug_arms.json').then(function (response) {
    return response.json();
  }).then(function (loaded) {
    arms = loaded;
    label();
  });
}());
