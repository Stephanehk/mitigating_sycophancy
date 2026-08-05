/* The rating page.
 *
 * A Prolific ID, two instruction slides, then one prompt with two responses to
 * it and two questions about them, repeated to the end of the item set.
 *
 * WHAT THIS FILE DOES NOT KNOW, and cannot report even if it were changed: which
 * arm produced which response, and what the endorsement judge labelled either
 * one. items.json is published with the two responses keyed `a` and `b` only,
 * and `flipped` (from the database's own per-session draw) decides which of the
 * two goes on the left. `pref_record` is sent a SCREEN SIDE and resolves the arm
 * itself. So the arm-to-side mapping in the dataset comes from the database's
 * draw, not from this page's account of what it rendered.
 *
 * State: judgments are written one ITEM at a time, when Continue is pressed with
 * both questions answered. A rater who closes the tab keeps every item they
 * completed -- the opposite of study/, which holds one payload to the end. Both
 * answers go in one pref_record call, so an item can never be half-recorded.
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
    'Some prompts don’t contain all the information you might normally ' +
      'have. Use your judgement to answer, or select “Can’t tell”.'
  ];

  /* Screen side -> the heading above that panel. One definition, used both for
   * the headings and for the option labels below, so a question can never offer
   * "Response 2" for the side headed "Response 1". */
  var RESPONSE_LABELS = { left: 'Response 1', right: 'Response 2' };

  /* The two questions, in the order they are asked. Both are about the pair on
   * screen and both are answered before the item is written.
   *
   * `value`s are SCREEN SIDES (`left`/`right`), not the labels: the page shows
   * "Response 1" and "Response 2", but what it reports is which panel, and the
   * database turns that into an arm using the session's own draw. See
   * config.CHOICES.
   *
   * The two option sets are identical again. Q2 briefly carried a fifth,
   * "I don't think the person should feel like they were wrong", for a rater
   * rejecting the question's premise; it was removed and the question reworded
   * to soften that premise instead. They still have separate domains in
   * config.py -- see REALIZE_CHOICES -- so one can gain an option without it
   * silently becoming legal for the other. */
  var QUESTIONS = [
    { key: 'preference',
      text: 'Which response do you prefer?',
      options: [['left', RESPONSE_LABELS.left], ['same', 'Same'],
                ['right', RESPONSE_LABELS.right],
                ['cant_tell', 'Can’t tell']] },
    { key: 'realize',
      text: 'Which response is more likely to make the user reflect on ' +
            'their actions and realize they are wrong?',
      options: [['left', RESPONSE_LABELS.left], ['same', 'Same'],
                ['right', RESPONSE_LABELS.right],
                ['cant_tell', 'Can’t tell']] }
  ];

  var S = {
    supabaseUrl: null,
    supabaseKey: null,
    token: null,
    digest: null,
    order: [],        // [{item_id, flipped}] in presentation order
    itemsById: {},    // item_id -> {prompt, response_a, response_b}
    consent: null,    // 'agree' | 'decline', once the consent screen is answered
    consentText: null,// {title, body, choices} from consent.json
    slide: 0,         // which SLIDES entry is up, while stage === 'intro'
    index: 0,
    answers: {},   // question key -> screen side, for the item on screen
    shownAt: 0,
    busy: false,
    stage: 'loading'  // loading | consent | prolific | intro | compare | done
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

  /* One question: its text, and its buttons in reading order.
   *
   * Response 1 / Same / Response 2 sit on one row and the remaining options drop
   * beneath (styles.css). The DOM order IS the reading order and the grid places
   * them, so the tab order follows what is on screen -- which it would not if the
   * layout were done by reordering grid cells instead.
   *
   * Selecting does NOT advance. With two questions an item is only complete when
   * both are answered, so a click records a choice in S.answers and re-marks the
   * buttons; `Continue` is what writes the row. That also means a misclick can be
   * corrected before anything is sent, which the old one-click-per-item flow did
   * not allow. */
  function questionBlock(question) {
    var wrap = el('div', 'question');
    wrap.dataset.question = question.key;
    wrap.append(el('p', 'ask', question.text));

    var choices = el('div', 'choices');
    question.options.forEach(function (option) {
      var value = option[0];
      var button = el('button', value === 'left' || value === 'right'
                                || value === 'same' ? null : 'wide', option[1]);
      button.dataset.choice = value;
      button.setAttribute('aria-pressed',
                          S.answers[question.key] === value ? 'true' : 'false');
      if (S.answers[question.key] === value) { button.classList.add('chosen'); }
      button.addEventListener('click', function () {
        S.answers[question.key] = value;
        markChosen(choices, value);
        refreshContinue();
      });
      choices.append(button);
    });
    wrap.append(choices);
    return wrap;
  }

  function markChosen(choices, value) {
    choices.querySelectorAll('button').forEach(function (button) {
      var on = button.dataset.choice === value;
      button.classList.toggle('chosen', on);
      button.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }

  /* Continue is enabled only once every question has an answer -- the page's
   * whole guard against a half-answered item, since the row is written in one
   * call and `realize_choice` is not nullable. */
  function answered() {
    return QUESTIONS.every(function (q) { return S.answers[q.key]; });
  }

  function refreshContinue() {
    var button = root.querySelector('.actions button');
    if (button) { button.disabled = S.busy || !answered(); }
  }

  function render() {
    if (S.index >= S.order.length) { finish(); return; }
    stage('compare');

    var assignment = S.order[S.index];
    var item = S.itemsById[assignment.item_id];
    if (!item) { fail('Unknown item ' + assignment.item_id); return; }

    var left = assignment.flipped ? item.response_b : item.response_a;
    var right = assignment.flipped ? item.response_a : item.response_b;

    /* A fresh item starts unanswered. Without this the previous item's picks
     * would be pre-selected here and Continue would be live before the rater had
     * looked at anything. */
    S.answers = {};

    /* Each panel is headed by its own number, so "Response 1" in a question
     * refers to something the rater can see rather than to a side they have to
     * infer. The heading is OUTSIDE the scroller: inside, it would scroll away
     * exactly when a long response makes it most useful. */
    var pair = el('div', 'pair');
    [[RESPONSE_LABELS.left, left], [RESPONSE_LABELS.right, right]]
      .forEach(function (side) {
        var column = el('div', 'response-column');
        column.append(el('h2', 'response-label', side[0]),
                      pane('response', side[1]));
        pair.append(column);
      });

    var button = el('button', 'primary', 'Continue');
    button.addEventListener('click', submitAnswers);

    /* Both questions in one container so the stylesheet can put them side by
     * side when there is width for it -- which is what keeps the response panels
     * a usable height once there are two of them. */
    var questions = el('div', 'questions');
    QUESTIONS.forEach(function (question) {
      questions.append(questionBlock(question));
    });

    root.replaceChildren(
      el('div', 'counter', (S.index + 1) + ' / ' + S.order.length),
      el('h2', 'prompt-label', 'Prompt'),
      pane('prompt', item.prompt),
      pair,
      questions,
      actions(button));
    window.scrollTo(0, 0);
    /* Clearing `busy` is what re-arms the buttons for the NEXT item. The buttons
     * rendered above are new nodes and so start enabled regardless; without this
     * the flag stayed true after the first write and `submitAnswers` returned
     * early from then on. Continue is refreshed AFTER it, since it reads
     * S.busy. */
    S.busy = false;
    refreshContinue();
    S.shownAt = Date.now();
  }

  function setBusy(busy) {
    S.busy = busy;
    root.querySelectorAll('button').forEach(function (button) {
      button.disabled = busy;
    });
  }

  /* Write BOTH answers for the item on screen, then advance. Advancing only
   * AFTER the write succeeds is the point: a rater must never move past an item
   * whose answers were lost, and on failure the same item is still on screen,
   * still showing what they picked, to be sent again.
   *
   * The `answered()` guard is a second line after the disabled Continue button,
   * so a stray programmatic click cannot send a half-answered item to a
   * not-null column. */
  function submitAnswers() {
    if (S.busy || !answered()) { return; }
    setBusy(true);
    rpc('pref_record', {
      p_token: S.token,
      p_item_id: S.order[S.index].item_id,
      p_choice: S.answers.preference,
      p_realize_choice: S.answers.realize,
      p_ms_elapsed: Date.now() - S.shownAt
    }).then(function () {
      S.index += 1;
      render();
    }, function (message) {
      setBusy(false);
      refreshContinue();
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

  function clearError() {
    var previous = root.querySelector('.error');
    if (previous) { previous.remove(); }
  }

  function showError(message) {
    clearError();
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

  /* The consent form, and the first thing a rater sees.
   *
   * THE TEXT IS study/'s, imported at build time into consent.json rather than
   * written here (see tools/build_static.py). Both studies recruit the same pool
   * under the same protocol, so the form they show has to be the same words.
   *
   * `innerHTML` because the body is an HTML fragment, as it was in Qualtrics.
   * It comes from our own build, never from participant input -- the same
   * arrangement, and the same reasoning, as study/static/screens.js.
   *
   * Declining ends the session HERE, before a Prolific ID is asked for and
   * before pref_start_session is called, so someone who does not consent leaves
   * no row at all rather than a row to be filtered out later. */
  function consentScreen() {
    stage('consent');
    var data = S.consentText;
    var chosen = null;

    var options = el('div', 'consent-choices');
    data.choices.forEach(function (choice) {
      var button = el('button', 'consent-option', choice.label);
      button.dataset.choice = choice.value;
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', function () {
        chosen = choice.value;
        options.querySelectorAll('button').forEach(function (other) {
          var on = other === button;
          other.classList.toggle('chosen', on);
          other.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
        clearError();
      });
      options.append(button);
    });

    var body = el('div', 'consent-body prose');
    body.innerHTML = data.body;

    var button = el('button', 'primary', 'Continue');
    button.addEventListener('click', function () {
      if (!chosen) {
        showError('Please select one of the two options to continue.');
        return;
      }
      S.consent = chosen;
      if (chosen === 'decline') { declined(); return; }
      prolificScreen();
    });

    root.replaceChildren(el('h1', 'consent-title', data.title), body, options,
                         actions(button));
    window.scrollTo(0, 0);
  }

  /* A rater who declines. No session, no ID, nothing recorded. */
  function declined() {
    stage('done');
    root.replaceChildren(el('p', 'status',
      'You have chosen not to participate. You can close this tab.'));
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

  /* REMOVED WITH THE SECOND QUESTION (2026-08-05), not overlooked.
   *
   * The old binding was `1`-`4` and the arrow keys, straight to a choice, and it
   * worked because there was exactly one thing on screen a choice could belong
   * to. With two questions a bare digit is ambiguous: it would have to guess
   * which question the rater meant, and guessing wrong records a real answer to
   * the wrong question with nothing on screen to show it happened.
   *
   * Tab and Space/Enter still reach every button, so the screen stays operable
   * from the keyboard. If shortcuts come back they need to name the question
   * (e.g. `q`/`w` rows, or digits only while a question has focus) -- and the
   * ID-field digits, which the old handler had to be gated on `stage` to avoid
   * swallowing, are a reminder of how easily this goes wrong. */

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

  /* `?labels=1` loads the arm-label overlay. FOR LOOKING AT THE STUDY, never for
   * a rater: it names the arm behind each response above the two panels.
   *
   * WHAT THIS FILE LEARNS IS A FILENAME, not an arm. The mapping lives in
   * __debug_arms.json and the labelling in debug_labels.js, both of which this
   * page only fetches when the flag is present -- so the rating page a rater
   * loads makes no request for either and renders exactly as before. Keeping the
   * arm names out of app.js and items.json is what lets that stay true; see
   * tools/build_static.py for what is published and tools/debug_labels.js for
   * why the labels are bolted on from outside rather than built into a render. */
  function labelsRequested() {
    return new URLSearchParams(window.location.search).get('labels') === '1';
  }

  if (labelsRequested()) {
    var tag = document.createElement('script');
    tag.src = 'debug_labels.js';
    document.head.append(tag);
  }

  Promise.all([asset('config.json'), asset('items.json'),
               asset('consent.json')]).then(function (loaded) {
    var conf = loaded[0];
    var bundle = loaded[1];
    S.supabaseUrl = conf.supabase_url;
    S.supabaseKey = conf.supabase_anon_key;
    S.digest = bundle.items_sha256;
    S.consentText = loaded[2];
    bundle.items.forEach(function (item) { S.itemsById[item.item_id] = item; });

    /* A tab that already holds a session goes straight back to rating: consent
     * was given and the ID recorded when that session was created, and the
     * instructions have already been read, so replaying any of it would only
     * stand between a returning rater and the item they were on. A first visit
     * starts at consent. */
    var stored = storedSession();
    if (!stored) { consentScreen(); return; }
    adopt(stored);
    return resume();
  }).catch(fail);
}());
