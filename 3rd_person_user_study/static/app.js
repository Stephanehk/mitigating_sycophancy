/* The rating page.
 *
 * A Prolific ID, three instruction slides, then one prompt with two responses to
 * it and four buttons, repeated to the end of the item set.
 *
 * WHAT THIS FILE DOES NOT KNOW, and cannot report even if it were changed: which
 * arm produced which response, and what the endorsement judge labelled either
 * one. items.json is published with the two responses keyed `a` and `b` only,
 * and `flipped` (from the database's own per-session draw) decides which of the
 * two goes on the left. `pref_record` is sent a SCREEN SIDE and resolves the arm
 * itself. So the arm-to-side mapping in the dataset comes from the database's
 * draw, not from this page's account of what it rendered.
 *
 * State: judgments are written one at a time, as they are made. A rater who
 * closes the tab keeps everything they answered -- the opposite of study/, which
 * holds one payload to the end. There is no per-page cost to avoid here, because
 * a judgment IS the page turn.
 *
 * The session token lives in sessionStorage so a reload resumes the same
 * session and the same order rather than drawing a second one. A reload DURING
 * the intro, before the session exists, restarts the intro -- there is nothing
 * yet to resume.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'pref_session';

  /* The intro, in order. Text lives here rather than in index.html because the
   * page is rendered entirely from script -- index.html holds only the mount
   * point. */
  var SLIDES = [
    'We are going to show you a prompt and two different responses from an ' +
      'AI chatbot.',
    'Imagine that your friend was writing the prompt. Your job is to select ' +
      'which response you would rather the chatbot show your friend.',
    'Some prompts don’t contain all the information you might normally ' +
      'have. Use your judgement to determine the best response, or select ' +
      '“Can’t tell”.'
  ];

  var S = {
    supabaseUrl: null,
    supabaseKey: null,
    token: null,
    digest: null,
    order: [],        // [{item_id, flipped}] in presentation order
    itemsById: {},    // item_id -> {prompt, response_a, response_b}
    slide: 0,         // which SLIDES entry is up, while stage === 'intro'
    index: 0,
    shownAt: 0,
    busy: false,
    stage: 'loading'  // loading | prolific | intro | compare | done
  };

  var root = document.getElementById('root');

  /* The rating screen is the only one laid out against the viewport height (see
   * styles.css), so the stage has to be visible to CSS.
   *
   * NB: app.js is served to raters, so tests/test_generators.py forbids the item
   * metadata field names over the WHOLE source, comments included. If a comment
   * here trips it, reword the comment -- do not loosen the test. */
  function stage(name) {
    S.stage = name;
    document.body.classList.toggle('compare', name === 'compare');
  }

  // --- plumbing ------------------------------------------------------------

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }

  /* A scrollable box of rendered markdown, wrapped in the element that carries
   * the "there is more below" affordance.
   *
   * The cue lives on the WRAPPER, not the scroller: a gradient and a badge
   * positioned inside the scrolling element would scroll away with the content,
   * which is the one place they must not be.
   *
   * tabindex makes the box keyboard-scrollable. Up/Down are not bound to any
   * choice, so they reach it; the choice keys are handled on `document` and still
   * fire while it has focus. */
  function pane(className, text) {
    var scroller = el('div', className + ' scroller');
    scroller.tabIndex = 0;
    scroller.append(window.MD.render(text));

    var hint = el('div', 'scroll-hint', 'more below ↓');
    hint.setAttribute('aria-hidden', 'true');

    var box = el('div', 'pane');
    box.append(scroller, hint);
    watchOverflow(box, scroller);
    return box;
  }

  /* Show the cue only while there is something below the fold.
   *
   * Measured after layout rather than guessed from text length: whether a
   * response overflows depends on the rendered height, which depends on the
   * viewport. `requestAnimationFrame` is the wait for that first layout --
   * measuring synchronously here reports scrollHeight === clientHeight for a
   * subtree that has not been laid out yet.
   *
   * A 2px slack absorbs sub-pixel rounding, which otherwise leaves the cue stuck
   * on at the very bottom of a box that is only just scrollable. */
  function watchOverflow(box, scroller) {
    function update() {
      var overflowing = scroller.scrollHeight - scroller.clientHeight > 2;
      var atBottom = scroller.scrollTop + scroller.clientHeight
                     >= scroller.scrollHeight - 2;
      box.classList.toggle('overflowing', overflowing);
      box.classList.toggle('at-bottom', atBottom);
    }
    scroller.addEventListener('scroll', update);
    window.addEventListener('resize', update);
    window.requestAnimationFrame(update);
  }

  function asset(path) {
    return fetch(path, { cache: 'no-cache' }).then(function (response) {
      if (!response.ok) { return Promise.reject('Could not load ' + path); }
      return response.json();
    });
  }

  /* A Supabase RPC. The anon key is public by design and can do nothing on its
   * own: RLS blocks every table and each function authenticates its caller. */
  function rpc(name, args) {
    return fetch(S.supabaseUrl + '/rest/v1/rpc/' + name, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'apikey': S.supabaseKey,
        'Authorization': 'Bearer ' + S.supabaseKey
      },
      body: JSON.stringify(args || {})
    }).then(function (response) {
      return response.json().then(function (parsed) {
        if (response.ok) { return parsed; }
        return Promise.reject((parsed && (parsed.message || parsed.details)) ||
                              ('Server error (' + response.status + ')'));
      }, function () {
        return Promise.reject('Server error (' + response.status + ')');
      });
    }, function () {
      return Promise.reject('Could not reach the server.');
    });
  }

  function fail(message) {
    root.replaceChildren(el('p', 'error', String(message)));
  }

  // --- rendering -----------------------------------------------------------

  /* The four buttons: Left / Same / Right on one row, Can't tell centred on a
   * second row beneath Same.
   *
   * The DOM order is the READING order, and the grid places them (styles.css) --
   * so the tab order and the keyboard numbers below both follow what is on
   * screen, which they would not if the layout were done by reordering the grid
   * cells instead. `cant-tell` is the hook the stylesheet uses to drop it into
   * the middle column of the second row. */
  function choiceButtons() {
    var choices = el('div', 'choices');
    [['left', 'Left'], ['same', 'Same'], ['right', 'Right'],
     ['cant_tell', 'Can’t tell']].forEach(function (choice) {
      var button = el('button', choice[0] === 'cant_tell' ? 'cant-tell' : null,
                      choice[1]);
      button.dataset.choice = choice[0];
      button.addEventListener('click', function () { answer(choice[0]); });
      choices.append(button);
    });
    return choices;
  }

  function render() {
    if (S.index >= S.order.length) { finish(); return; }
    stage('compare');

    var assignment = S.order[S.index];
    var item = S.itemsById[assignment.item_id];
    if (!item) { fail('Unknown item ' + assignment.item_id); return; }

    var left = assignment.flipped ? item.response_b : item.response_a;
    var right = assignment.flipped ? item.response_a : item.response_b;

    var pair = el('div', 'pair');
    pair.append(pane('response', left), pane('response', right));

    /* The question is split ACROSS the prompt -- lead-in above, the actual ask
     * below -- so the prompt reads as the thing being quoted and the rater is
     * left looking at the question, not at the scenario, when they choose. */
    root.replaceChildren(
      el('div', 'counter', (S.index + 1) + ' / ' + S.order.length),
      el('p', 'framing', 'If your friend asked an AI chatbot the following:'),
      pane('prompt', item.prompt),
      el('p', 'framing ask',
         'which response would you rather the chatbot show them?'),
      pair,
      choiceButtons());
    window.scrollTo(0, 0);
    /* Clearing `busy` is what re-arms the buttons for the NEXT item. The buttons
     * rendered above are new nodes and so start enabled regardless; without this
     * the flag stayed true after the first write and `answer` returned early
     * from then on. */
    S.busy = false;
    S.shownAt = Date.now();
  }

  function setBusy(busy) {
    S.busy = busy;
    root.querySelectorAll('button').forEach(function (button) {
      button.disabled = busy;
    });
  }

  /* Send one judgment, then advance. Advancing only AFTER the write succeeds is
   * the point: a rater must never move past an item whose answer was lost, and
   * on failure the same item is still on screen to be re-answered. */
  function answer(choice) {
    if (S.busy) { return; }
    setBusy(true);
    rpc('pref_record', {
      p_token: S.token,
      p_item_id: S.order[S.index].item_id,
      p_choice: choice,
      p_ms_elapsed: Date.now() - S.shownAt
    }).then(function () {
      S.index += 1;
      render();
    }, function (message) {
      setBusy(false);
      showError(message);
    });
  }

  /* pref_complete only stamps a timestamp -- every judgment is already written --
   * so a failure here costs a completion time, not data. It is still surfaced
   * rather than swallowed, because a token that has stopped working is worth
   * seeing. */
  function finish() {
    stage('done');
    root.replaceChildren(el('p', 'status', 'Done.'));
    rpc('pref_complete', { p_token: S.token }).then(null, function (message) {
      root.append(el('p', 'error', String(message)));
    });
  }

  // --- intro ---------------------------------------------------------------

  function showError(message) {
    var previous = root.querySelector('.error');
    if (previous) { previous.remove(); }
    root.insertBefore(el('p', 'error', String(message)), root.firstChild);
  }

  /* A centred row holding one button. */
  function actions(button) {
    var row = el('div', 'actions');
    row.append(button);
    return row;
  }

  /* A slide: one line of instruction and a Continue button. */
  function slideScreen() {
    stage('intro');
    var button = el('button', 'primary', 'Continue');
    button.addEventListener('click', function () {
      S.slide += 1;
      if (S.slide < SLIDES.length) { slideScreen(); } else { render(); }
    });
    root.replaceChildren(el('div', 'slide', SLIDES[S.slide]), actions(button));
    window.scrollTo(0, 0);
    button.focus();
  }

  /* The Prolific ID, prefilled from the URL Prolific sends the rater in on.
   *
   * REQUIRED, and it is what starts the session: a rater who never gives one
   * never gets an order drawn, so the sessions table cannot fill with rows that
   * are neither payable nor joinable. The trade is that the item-digest check
   * fires here rather than at load -- which is the better place for it anyway,
   * since a stale deployment then fails before anyone reads the instructions. */
  function prolificScreen() {
    stage('prolific');
    var field = el('input');
    field.type = 'text';
    field.placeholder = 'Your Prolific ID';
    field.value = prolificFromUrl();

    var button = el('button', 'primary', 'Continue');

    function submit() {
      if (S.busy) { return; }
      var value = field.value.trim();
      if (!value) {
        showError('Please enter your Prolific ID.');
        field.focus();
        return;
      }
      S.busy = true;
      button.disabled = true;
      startSession(value.slice(0, 200)).then(function () {
        S.busy = false;
        S.slide = 0;
        slideScreen();
      }, function (message) {
        S.busy = false;
        button.disabled = false;
        showError(message);
      });
    }

    button.addEventListener('click', submit);
    field.addEventListener('keydown', function (event) {
      if (event.key === 'Enter') { event.preventDefault(); submit(); }
    });

    root.replaceChildren(el('div', 'slide', 'Please enter your Prolific ID.'),
                         field, actions(button));
    field.focus();
  }

  function prolificFromUrl() {
    var params = new URLSearchParams(window.location.search);
    return params.get('PROLIFIC_PID') || params.get('prolific_pid') || '';
  }

  // --- keyboard ------------------------------------------------------------

  /* Numbered LEFT TO RIGHT AS DISPLAYED -- 1 Left, 2 Same, 3 Right, 4 Can't
   * tell. The numbers are not printed on the buttons, so the only thing they can
   * agree with is the layout; when Same moved into the middle, `2` had to move
   * with it or it would silently mean the button two places away. */
  var KEYS = { '1': 'left', '2': 'same', '3': 'right', '4': 'cant_tell',
               ArrowLeft: 'left', ArrowRight: 'right' };

  /* Gated on the STAGE, not on the item set being loaded. The order is drawn
   * before the intro slides are shown, so a check for `S.order.length` would let
   * a keypress on an instruction slide record a judgment for item 1 -- and would
   * also swallow every digit typed into the Prolific ID field. */
  document.addEventListener('keydown', function (event) {
    var choice = KEYS[event.key];
    if (choice && S.stage === 'compare' && S.index < S.order.length) {
      event.preventDefault();
      answer(choice);
    }
  });

  // --- start ---------------------------------------------------------------

  /* The session stored for THIS item build, or null. The digest is part of the
   * check so a redeployed item set cannot be answered against a session drawn
   * over the old one. */
  function storedSession() {
    var raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) { return null; }
    var stored = JSON.parse(raw);
    return stored.items_sha256 === S.digest ? stored : null;
  }

  function adopt(stored) {
    S.token = stored.session_token;
    S.order = stored.items;
  }

  /* Draw the order and sides for a rater who has just given their ID. */
  function startSession(prolificPid) {
    return rpc('pref_start_session', {
      p_user_agent: navigator.userAgent,
      p_items_sha256: S.digest,
      p_prolific_pid: prolificPid
    }).then(function (result) {
      var fresh = { session_token: result.session_token, items: result.items,
                    items_sha256: S.digest };
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(fresh));
      adopt(fresh);
    });
  }

  /* Resume at the first UNANSWERED item, so a reload does not re-ask (and
   * overwrite) something already judged. The database computes it; a failure here
   * surfaces rather than falling back to position 0, which would do exactly the
   * overwriting this avoids. */
  function resume() {
    return rpc('pref_progress', { p_token: S.token }).then(function (progress) {
      S.index = Math.min(progress.next_position, S.order.length);
      render();
    });
  }

  Promise.all([asset('config.json'), asset('items.json')]).then(function (loaded) {
    var conf = loaded[0];
    var bundle = loaded[1];
    S.supabaseUrl = conf.supabase_url;
    S.supabaseKey = conf.supabase_anon_key;
    S.digest = bundle.items_sha256;
    bundle.items.forEach(function (item) { S.itemsById[item.item_id] = item; });

    /* A tab that already holds a session goes straight back to rating: the ID is
     * already recorded and the instructions have already been read, so replaying
     * the intro would only stand between a returning rater and the item they were
     * on. A first visit starts at the ID. */
    var stored = storedSession();
    if (!stored) { prolificScreen(); return; }
    adopt(stored);
    return resume();
  }).catch(fail);
}());
