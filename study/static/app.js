/* Sycophancy user study — client core: state, flow, transport.
 *
 * THE CLIENT OWNS THE FLOW. It fetches the whole study once, walks the pages
 * locally, accumulates every answer in one object, and posts that object to the
 * server exactly once at the end.
 *
 * This is a reversal of the previous server-driven design, and the reason is
 * latency: every Continue press used to be an HTTP round trip plus two
 * synchronous SQLite writes, which on the NFS-backed database cost ~213ms
 * typically and stalled for 10-14 SECONDS whenever the GPU worker's long poll
 * contended for the write lock. Page turns are now instant because they touch
 * nothing but the DOM.
 *
 * The server is contacted exactly three times per participant:
 *   1. POST /api/session        — token, randomized scenario order, all copy.
 *   2. POST /api/response/start — the two free-text answers that are the
 *                                 model's input; then GET /api/response/status
 *                                 polls until the reply is ready.
 *   3. POST /api/submit         — the whole study, once, at the terminal page.
 *
 * WHAT THIS COSTS, STATED PLAINLY: a participant who abandons mid-study sends
 * nothing, so their partial answers are lost. That is a deliberate choice (see
 * README "Where the data lives"). Terminal pages — complete, screened_out and
 * declined — all submit, so anyone who reaches the end of their path is
 * recorded even if that path was short.
 *
 * WHAT THE SERVER STILL DECIDES: the model arm (drawn at job creation, never
 * sent here) and the one-job-per-session rule. Neither can be influenced from
 * this file, which is the point — otherwise the condition could be re-rolled
 * from devtools.
 *
 * Resilience:
 *   - Session token AND the answer payload live in localStorage, so a refresh
 *     or a closed laptop resumes exactly where the participant was, with their
 *     answers intact.
 *   - The generation screen polls, shows an extra reassurance line if it runs
 *     long, and offers an explicit retry on failure.
 *   - A failed final submit is retryable from the terminal page rather than
 *     silently dropping a completed session.
 *
 * LAYOUT. This file holds state, the stage machine, persistence, the fetch
 * wrapper and the page chrome, and exposes them as `window.APP`. The individual
 * screen renderers live in screens.js, which registers them on `APP.screens`
 * and then calls `APP.start()` — so screens.js must load second (see
 * index.html). The split is only to keep both files readable; there is no
 * framework and no build step.
 */
(function () {
  'use strict';

  var el = window.W.el;
  var TOKEN_KEY = 'syco_study_session_token';
  var DATA_KEY = 'syco_study_payload';

  var root = document.getElementById('root');
  var progressTrack = document.getElementById('progress-track');
  var progressFill = document.getElementById('progress-fill');

  /* Everything the client knows. `content` and `screeningOrder` come from the
   * server once; `data` is what will eventually be posted back. */
  var S = {
    token: null,
    sessionId: null,
    screeningOrder: [],
    content: null,
    stage: 'consent',
    screeningIndex: 0,
    hasJob: false,
    submitted: false,
    shownResponse: null,
    data: null
  };

  var pollTimer = null;

  function freshData() {
    return {
      consent: null,
      prolific_pid: '',
      screening: [],
      selected_scenario: null,
      eligibility: null,
      perspective_text: '',
      additional_thoughts: '',
      answers: {},
      client_timings: {}
    };
  }

  // --- Stage machine (client-side) -----------------------------------------

  /* The linear spine. Built once the content bundle arrives, because the
   * questionnaire pages come from questionnaires.py rather than being hardcoded
   * here — that module stays the single source of truth for the instrument. */
  var SPINE = [];

  function buildSpine() {
    SPINE = ['consent', 'prolific_id', 'prelude', 'screening', 'perspective',
             'response_intro', 'response']
      .concat(S.content.questionnaire_pages.map(function (page) { return page.page; }))
      .concat(['debrief', 'additional_thoughts', 'complete']);
  }

  var TERMINAL = ['complete', 'screened_out', 'declined'];

  function isTerminal(stage) { return TERMINAL.indexOf(stage) !== -1; }

  /* Once a generation job exists, back may not reach any stage before the
   * response page. The participant has seen the model's output, so letting them
   * edit the input behind it would either desynchronize the response from the
   * text that produced it or invite regenerating until they like the answer.
   * The server enforces the same rule by refusing to create a second job; this
   * is the UI half of it. */
  var BACK_LOCK_STAGE = 'response';

  function backTarget() {
    if (S.stage === 'consent' || isTerminal(S.stage)) { return null; }

    var target;
    if (S.stage === 'screening') {
      target = S.screeningIndex > 0
        ? { stage: 'screening', screeningIndex: S.screeningIndex - 1 }
        : { stage: 'prelude' };
    } else if (S.stage === 'perspective') {
      // The screening page that selected the scenario is the one before the
      // cursor, since answering "Very similar" advanced it.
      target = { stage: 'screening', screeningIndex: Math.max(0, S.screeningIndex - 1) };
    } else {
      var index = SPINE.indexOf(S.stage);
      if (index <= 0) { return null; }
      target = { stage: SPINE[index - 1] };
    }

    if (S.hasJob && SPINE.indexOf(target.stage) < SPINE.indexOf(BACK_LOCK_STAGE)) {
      return null;
    }
    return target;
  }

  function goBack() {
    var target = backTarget();
    if (!target) { return; }
    if (target.screeningIndex !== undefined) { S.screeningIndex = target.screeningIndex; }
    if (target.stage === 'screening') {
      // Re-opening screening un-selects the scenario so the re-answer decides
      // it afresh. The ARM is not touched — the server holds it and never
      // redraws it, which is what stops Back from re-randomizing a participant.
      S.data.selected_scenario = null;
      S.data.eligibility = null;
    }
    setStage(target.stage);
  }

  function advance() {
    var index = SPINE.indexOf(S.stage);
    if (index === -1 || index + 1 >= SPINE.length) {
      throw new Error('no successor for stage ' + S.stage);
    }
    setStage(SPINE[index + 1]);
  }

  function setStage(stage) {
    S.stage = stage;
    // First arrival at each stage, for reading-time and dropout analysis.
    if (!S.data.client_timings[stage]) {
      S.data.client_timings[stage] = new Date().toISOString();
    }
    save();
    render();
  }

  // --- Persistence ---------------------------------------------------------

  function save() {
    window.localStorage.setItem(DATA_KEY, JSON.stringify({
      stage: S.stage,
      screeningIndex: S.screeningIndex,
      hasJob: S.hasJob,
      shownResponse: S.shownResponse,
      data: S.data
    }));
  }

  function restore() {
    var raw = window.localStorage.getItem(DATA_KEY);
    if (!raw) { return false; }
    var parsed;
    try { parsed = JSON.parse(raw); } catch (e) { return false; }
    if (!parsed || !parsed.data) { return false; }
    S.stage = parsed.stage || 'consent';
    S.screeningIndex = parsed.screeningIndex || 0;
    S.hasJob = !!parsed.hasJob;
    S.shownResponse = parsed.shownResponse || null;
    S.data = parsed.data;
    return true;
  }

  function clearStorage() {
    window.localStorage.removeItem(TOKEN_KEY);
    window.localStorage.removeItem(DATA_KEY);
  }

  // --- API -----------------------------------------------------------------

  /* One fetch wrapper for the whole app. Only five endpoints use it. */
  function api(path, options) {
    options = options || {};
    var headers = { 'Content-Type': 'application/json' };
    if (S.token) { headers['X-Session-Token'] = S.token; }

    return fetch(path, {
      method: options.method || 'POST',
      headers: headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body)
    }).then(function (response) {
      return response.text().then(function (raw) {
        var parsed = null;
        if (raw) { try { parsed = JSON.parse(raw); } catch (e) { parsed = null; } }
        if (response.ok) { return parsed || {}; }
        if (response.status === 401) { throw { kind: 'unauthorized' }; }
        var detail = parsed && (parsed.error || parsed.detail);
        if (detail && detail.error) { detail = detail.error; }
        if (response.status === 400) {
          throw { kind: 'validation', message: detail || 'Please check your answer.' };
        }
        throw { kind: 'network', message: detail || ('Server error (' + response.status + ')') };
      });
    }, function () {
      throw { kind: 'network', message: 'Could not reach the server. Check your connection.' };
    });
  }

  // --- Chrome --------------------------------------------------------------

  function showError(message) {
    clearError();
    var box = el('div', { className: 'error-box', text: message, attrs: { id: 'error-box' } });
    root.insertBefore(box, root.firstChild);
    box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function clearError() {
    var existing = document.getElementById('error-box');
    if (existing) { existing.remove(); }
  }

  function setProgress(stage) {
    var index = SPINE.indexOf(stage);
    if (index === -1) { progressTrack.hidden = true; return; }
    progressTrack.hidden = false;
    progressFill.style.width = ((index / (SPINE.length - 1)) * 100).toFixed(1) + '%';
  }

  function mount(nodes) {
    window.W.clear(root);
    nodes.forEach(function (node) { if (node) { root.appendChild(node); } });
    window.scrollTo({ top: 0, behavior: 'auto' });
  }

  function primaryButton(label, onClick) {
    var button = el('button', { className: 'btn-primary', text: label });
    button.addEventListener('click', function () { clearError(); onClick(button); });
    return button;
  }

  /* The button row for a page: Back on the left when the flow allows one, then
   * the page's primary button. */
  function navActions(primary) {
    var children = [];
    if (backTarget()) {
      var back = el('button', { className: 'btn-secondary', text: 'Back' });
      back.addEventListener('click', function () { clearError(); goBack(); });
      children.push(back);
    }
    children.push(primary);
    return el('div', { className: 'actions' }, children);
  }

  function stopPolling() {
    if (pollTimer) { window.clearInterval(pollTimer); pollTimer = null; }
  }

  function setPollTimer(timer) { pollTimer = timer; }

  function pageFor(stage) {
    var pages = S.content.questionnaire_pages;
    for (var i = 0; i < pages.length; i++) {
      if (pages[i].page === stage) { return pages[i]; }
    }
    return null;
  }

  // --- Router --------------------------------------------------------------

  function render() {
    stopPolling();
    clearError();
    setProgress(S.stage);

    var screens = APP.screens;
    if (isTerminal(S.stage)) { return screens.terminal(); }
    if (screens[S.stage]) { return screens[S.stage](); }

    var page = pageFor(S.stage);
    if (page) { return screens.questionnaire(page); }

    showError('Unexpected study state. Please contact the researchers.');
  }

  function prolificFromUrl() {
    var params = new URLSearchParams(window.location.search);
    return params.get('PROLIFIC_PID') || params.get('prolific_pid') || '';
  }

  function restart() {
    clearStorage();
    window.location.reload();
  }

  function start() {
    var stored = window.localStorage.getItem(TOKEN_KEY);
    api('/api/session', {
      body: { session_token: stored, prolific_pid: prolificFromUrl() }
    }).then(function (body) {
      S.token = body.session_token;
      S.sessionId = body.session_id;
      S.screeningOrder = body.screening_order;
      S.content = body.content;
      window.localStorage.setItem(TOKEN_KEY, S.token);
      buildSpine();

      var resumed = stored && stored === body.session_token && restore();
      if (!resumed) {
        S.data = freshData();
        S.stage = 'consent';
        S.screeningIndex = 0;
        S.hasJob = false;
        S.shownResponse = null;
      }
      // The server is authoritative on whether a generation exists, which is
      // what arms the one-way door after a localStorage wipe.
      S.hasJob = S.hasJob || body.has_job;
      setStage(S.stage);
    }).catch(function (error) {
      window.W.clear(root);
      root.appendChild(el('div', { className: 'terminal' }, [
        el('h1', { text: 'We could not start the study.' }),
        el('p', { className: 'muted',
          text: error.message || 'Please refresh the page to try again.' }),
        el('div', { className: 'actions' }, [
          primaryButton('Reload', function () { window.location.reload(); })
        ])
      ]));
    });
  }

  /* What screens.js is allowed to touch. Kept explicit rather than exporting
   * the whole closure, so the surface between the two files is readable. */
  var APP = {
    S: S,
    screens: {},
    api: api,
    advance: advance,
    setStage: setStage,
    save: save,
    clearStorage: clearStorage,
    restart: restart,
    render: render,
    mount: mount,
    navActions: navActions,
    primaryButton: primaryButton,
    showError: showError,
    clearError: clearError,
    stopPolling: stopPolling,
    setPollTimer: setPollTimer,
    prolificFromUrl: prolificFromUrl,
    start: start
  };

  window.APP = APP;
})();
