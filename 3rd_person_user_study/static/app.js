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
    'We will ask you questions about the two responses. For each pair, you ' +
      'will not be allowed to press continue until 45 seconds have elapsed.',
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
      options: [['left', RESPONSE_LABELS.left],
                ['right', RESPONSE_LABELS.right],
                ['cant_tell', 'Can’t tell']] },
    { key: 'realize',
      text: 'Which response is more likely to make the user reflect on ' +
            'their actions and realize they are wrong?',
      options: [['left', RESPONSE_LABELS.left],
                ['right', RESPONSE_LABELS.right],
                ['cant_tell', 'Can’t tell']] }
  ];

  /* Q3, free text. Not in QUESTIONS because it is not a choice: it renders a
   * textarea, it gates on a length rather than on having a value, and it is
   * stored in its own column. Folding it in would mean every helper above
   * branching on which kind of question it had. */
  var REASONING = { key: 'reasoning',
                    text: 'Briefly, explain your reasoning' };

  var S = {
    supabaseUrl: null,
    supabaseKey: null,
    token: null,
    digest: null,
    order: [],        // [{item_id, flipped}] in presentation order
    itemsById: {},    // item_id -> {prompt, response_a, response_b}
    setIndex: null,   // which link's set of pairs this rater was sent to
    completionCode: null,
    consent: null,    // 'agree' | 'decline', once the consent screen is answered
    consentText: null,// {title, body, choices} from consent.json
    slide: 0,         // which SLIDES entry is up, while stage === 'intro'
    index: 0,
    answers: {},   // question key -> screen side, for the item on screen
    reasoning: '', // Q3's free text, for the item on screen
    minChars: null,// Q3's length floor, from config.json
    shownAt: 0,
    minSeconds: null, // per-item floor, from config.json
    tick: null,       // interval re-evaluating the countdown
    busy: false,
    stage: 'loading'  // loading | consent | prolific | intro | compare | submit | done
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
                                ? null : 'wide', option[1]);
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
    return QUESTIONS.every(function (q) { return S.answers[q.key]; })
           && S.reasoning.trim().length >= S.minChars;
  }

  /* The free-text question: a centred prompt and a box, with a live count of
   * how many characters are still needed.
   *
   * The counter is there for the same reason the seconds are on Continue -- a
   * button that stays dead after the rater has answered everything they can see
   * reads as broken unless the page says what is missing. */
  function reasoningBlock() {
    var wrap = el('div', 'question reasoning');
    wrap.dataset.question = REASONING.key;
    wrap.append(el('p', 'ask', REASONING.text));

    var box = el('textarea');
    box.rows = 2;   /* "briefly" -- and it is user-resizable */
    box.value = S.reasoning;
    box.setAttribute('aria-label', REASONING.text);

    var note = el('p', 'reasoning-note');
    function updateNote() {
      var short = S.minChars - S.reasoning.trim().length;
      note.textContent = short > 0 ? short + ' more characters' : '';
    }
    box.addEventListener('input', function () {
      S.reasoning = box.value;
      updateNote();
      refreshContinue();
    });
    updateNote();

    wrap.append(box, note);
    return wrap;
  }

  /* Whole seconds still owed on the current item, or 0. */
  function secondsLeft() {
    var elapsed = Math.floor((Date.now() - S.shownAt) / 1000);
    return Math.max(0, S.minSeconds - elapsed);
  }

  /* Continue reflects BOTH gates, and says which one is holding it.
   *
   * The countdown is on the button on purpose. A button that is disabled for a
   * minute with nothing to explain it reads as broken -- the rater has answered
   * both questions and nothing happens -- and the ones who do not simply give up
   * will email about it. Showing the seconds makes the wait a rule rather than a
   * fault. */
  function refreshContinue() {
    var button = root.querySelector('.actions button');
    if (!button) { return; }
    var left = secondsLeft();
    button.disabled = S.busy || !answered() || left > 0;
    button.textContent = left > 0 ? 'Continue (' + left + 's)' : 'Continue';
  }

  function render() {
    if (S.index >= S.order.length) { submitScreen(); return; }
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
    S.reasoning = '';

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
    questions.append(reasoningBlock());

    root.replaceChildren(
      el('div', 'counter', (S.index + 1) + ' / ' + S.order.length),
      el('h2', 'prompt-label', 'Prompt'),
      pane('prompt', item.prompt),
      pair,
      questions,
      actions(button));
    window.scrollTo(0, 0);
    /* Before refreshContinue, which reads it. */
    S.shownAt = Date.now();
    /* One ticker at a time: render() runs per item, and an interval left behind
     * would keep re-enabling a button on a screen that no longer exists. */
    if (S.tick) { window.clearInterval(S.tick); }
    S.tick = window.setInterval(refreshContinue, 250);
    /* Clearing `busy` is what re-arms the buttons for the NEXT item. The buttons
     * rendered above are new nodes and so start enabled regardless; without this
     * the flag stayed true after the first write and `submitAnswers` returned
     * early from then on. Continue is refreshed AFTER it, since it reads
     * S.busy. */
    S.busy = false;
    refreshContinue();
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
    if (S.busy || !answered() || secondsLeft() > 0) { return; }
    setBusy(true);
    rpc('pref_record', {
      p_token: S.token,
      p_item_id: S.order[S.index].item_id,
      p_choice: S.answers.preference,
      p_realize_choice: S.answers.realize,
      p_reasoning: S.reasoning.trim(),
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

  /* The last screen before the code: everything is answered, nothing has been
   * finalised.
   *
   * EVERY JUDGMENT IS ALREADY IN SUPABASE by this point -- each one is written
   * as it is made, so a rater who closes the tab at pair 7 keeps 7 pairs rather
   * than losing 9. What Submit does is finalise the session (pref_complete) and
   * release the completion code. Buffering all nine in the browser to post them
   * here instead would make one dropped request cost the whole session, and
   * would gain nothing a rater can see. */
  function submitScreen() {
    if (S.tick) { window.clearInterval(S.tick); S.tick = null; }
    stage('submit');
    var button = el('button', 'primary', 'Submit data');
    button.addEventListener('click', function () {
      button.disabled = true;
      clearError();
      rpc('pref_complete', { p_token: S.token }).then(finish, function (message) {
        button.disabled = false;
        showError(message);
      });
    });
    root.replaceChildren(
      el('p', 'slide', 'That is all ' + S.order.length + ' pairs. Press Submit '
         + 'data to record your responses and get your completion code.'),
      actions(button));
    window.scrollTo(0, 0);
  }

  /* The code, and nothing that can fail after it. */
  function finish() {
    stage('done');
    root.replaceChildren(
      el('p', 'slide', 'Your Prolific completion code is ' + S.completionCode),
      el('p', 'status', 'Paste this into Prolific to be paid. '
                        + 'You can close this tab afterwards.'));
    window.scrollTo(0, 0);
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

  /* Which set this link is for, from the URL hash: #set_1 .. #set_8.
   *
   * A HASH and not a path, because this is deployed as a static site: a path
   * like /set_3 needs a rewrite rule to reach index.html, and Amplify's rewrites
   * live in the console rather than in the repo, so a fresh app would 404. The
   * hash is never sent to the server and needs no routing at all.
   *
   * Returns null for a missing or malformed hash. That is a REFUSAL, not a
   * default: falling back to set 1 would silently pile every mis-linked rater
   * onto the same nine items, and falling back to "all of them" would hand one
   * rater 72. */
  function setFromUrl() {
    var match = /^#set_(\d+)$/.exec(window.location.hash || '');
    return match ? parseInt(match[1], 10) : null;
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
      p_prolific_pid: prolificPid,
      p_set_index: S.setIndex
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

  /* THE `?labels=1` OVERLAY IS GONE FROM THE DEPLOYED PAGE (2026-08-05).
   *
   * It injected a script that named the arm above each panel, and the build
   * published the mapping beside the page so it worked on the live site. With
   * the study recruiting, that mapping is the answer key: this page must not be
   * able to fetch it, and the build no longer emits it.
   *
   * tools/local_server.py still serves both files, so the labelled view still
   * works against a local checkout -- which is where looking at it belongs. */

  Promise.all([asset('config.json'), asset('items.json'),
               asset('consent.json')]).then(function (loaded) {
    var conf = loaded[0];
    var bundle = loaded[1];
    S.supabaseUrl = conf.supabase_url;
    S.supabaseKey = conf.supabase_anon_key;
    S.minSeconds = conf.min_item_seconds;
    S.minChars = conf.min_reasoning_chars;
    S.completionCode = conf.completion_code;
    if (typeof S.minSeconds !== 'number' || typeof S.minChars !== 'number'
        || !S.completionCode) {
      return Promise.reject('config.json is missing a pacing value or the code');
    }

    /* No set means the link is wrong, and there is nothing sensible to show
     * past this point -- so say so here rather than let someone consent and
     * enter an ID before failing. */
    S.setIndex = setFromUrl();
    if (S.setIndex === null) {
      fail('This link is incomplete. Please use the exact link from the study '
           + 'listing, which ends in #set_1 through #set_8.');
      return;
    }
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
