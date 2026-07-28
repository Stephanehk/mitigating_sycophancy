/* Sycophancy user study — the individual page renderers.
 *
 * Split out of app.js purely for readability; there is no framework here. Each
 * function draws one stage and wires its Continue button. app.js owns the state
 * and the stage machine and exposes what these need as `window.APP`; this file
 * registers its renderers on `APP.screens` and then starts the study, so it
 * must load AFTER app.js (see index.html).
 *
 * Two rules inherited from widgets.js hold throughout:
 *   1. Study-authored copy is trusted HTML (the .qsf text carries its own
 *      <strong>/<br>) and is set with `html`.
 *   2. Anything a participant typed is set with `text`, never `html`.
 *
 * Every renderer here is local-only — no HTTP — EXCEPT `response_intro` (which
 * enqueues the generation), `response` (which polls it) and `terminal` (which
 * submits the study). Those three call Supabase RPCs directly; there is no
 * application server. That is the whole network surface of the participant
 * experience.
 */
(function () {
  'use strict';

  var APP = window.APP;
  var el = window.W.el;
  var S = APP.S;

  // --- Consent, ID, prelude ------------------------------------------------

  APP.screens.consent = function () {
    var data = S.content.consent;
    var pick = { name: 'consent', value: S.data.consent };
    APP.mount([
      el('h1', { text: data.title }),
      el('div', { className: 'scroll-prose prose', html: data.body }),
      el('div', { attrs: { style: 'height:22px' } }),
      window.W.choiceList(data.choices, pick, function () { APP.clearError(); }),
      APP.navActions(APP.primaryButton('Continue', function () {
        if (!pick.value) {
          APP.showError('Please select one of the two options to continue.');
          return;
        }
        S.data.consent = pick.value;
        if (pick.value === 'decline') {
          S.data.eligibility = 'declined';
          APP.setStage('declined');
          return;
        }
        APP.advance();
      }))
    ]);
  };

  APP.screens.prolific_id = function () {
    var field = el('input', { attrs: { type: 'text', placeholder: 'Your Prolific ID' } });
    field.value = S.data.prolific_pid || APP.prolificFromUrl() || '';
    APP.mount([
      el('h1', { html: S.content.prolific.question }),
      el('p', { className: 'help-text', text: S.content.prolific.help }),
      field,
      APP.navActions(APP.primaryButton('Continue', function () {
        // REQUIRED. The source survey used Qualtrics' soft "request response"
        // and accepted a blank, which meant a submission could arrive with no
        // way to match it to a Prolific participant — unpayable and unusable.
        var value = field.value.trim();
        if (!value) {
          APP.showError(S.content.prolific.required_error);
          field.focus();
          return;
        }
        S.data.prolific_pid = value.slice(0, 200);
        APP.advance();
      }))
    ]);
  };

  APP.screens.prelude = function () {
    APP.mount([
      el('div', { className: 'prose', html: S.content.prelude }),
      APP.navActions(APP.primaryButton('Continue', function () { APP.advance(); }))
    ]);
  };

  // --- Screening -----------------------------------------------------------

  function findScreening(key) {
    for (var i = 0; i < S.data.screening.length; i++) {
      if (S.data.screening[i].scenario_key === key) { return S.data.screening[i]; }
    }
    return null;
  }

  function recordScreening(key, similarity, description) {
    var existing = findScreening(key);
    if (existing) {
      existing.similarity = similarity;
      existing.description = description;
      return;
    }
    S.data.screening.push({
      scenario_key: key, similarity: similarity, description: description
    });
  }

  APP.screens.screening = function () {
    var conf = S.content.screening;
    var key = S.screeningOrder[S.screeningIndex];
    var scenario = conf.scenarios[key];
    var existing = findScreening(key);

    var pick = { name: 'similarity', value: existing ? existing.similarity : null };
    var describeWrap = el('div', { className: 'hidden' });
    var describe = window.W.textArea({
      minChars: conf.min_describe_chars,
      maxChars: S.content.max_text_chars,
      value: existing ? (existing.description || '') : '',
      placeholder: 'Briefly describe what happened…'
    });
    describeWrap.appendChild(el('div', { className: 'question-text',
      text: conf.describe_question }));
    describeWrap.appendChild(describe.node);
    if (pick.value && conf.describe_required_for.indexOf(pick.value) !== -1) {
      describeWrap.classList.remove('hidden');
    }

    var picker = window.W.choiceList(conf.choices, pick, function (value) {
      APP.clearError();
      describeWrap.classList.toggle('hidden',
        conf.describe_required_for.indexOf(value) === -1);
    });

    APP.mount([
      el('p', { className: 'faint', text: 'Situation ' + (S.screeningIndex + 1) +
        ' of ' + S.screeningOrder.length }),
      el('div', { className: 'scenario' }, [
        el('div', { className: 'scenario-title', text: scenario.title }),
        el('p', { className: 'scenario-vignette', text: scenario.vignette })
      ]),
      el('div', { className: 'question-text', html: scenario.question }),
      picker,
      el('div', { attrs: { style: 'height:22px' } }),
      describeWrap,
      APP.navActions(APP.primaryButton('Continue', function () {
        if (!pick.value) {
          APP.showError('Please choose one of the options before continuing.');
          return;
        }
        var needsDescription = conf.describe_required_for.indexOf(pick.value) !== -1;
        var text = describe.field.value.trim();
        if (needsDescription && text.length < conf.min_describe_chars) {
          APP.showError('Please write a bit more — at least ' +
                        conf.min_describe_chars + ' characters.');
          return;
        }
        recordScreening(key, pick.value, needsDescription ? text : null);

        if (conf.selects_scenario.indexOf(pick.value) !== -1) {
          S.data.selected_scenario = key;
          S.data.eligibility = 'eligible';
          S.screeningIndex += 1;
          APP.setStage('perspective');
          return;
        }
        if (S.screeningIndex + 1 >= S.screeningOrder.length) {
          S.data.eligibility = 'screened_out';
          APP.setStage('screened_out');
          return;
        }
        S.screeningIndex += 1;
        APP.setStage('screening');
      }))
    ]);
  };

  APP.screens.perspective = function () {
    var conf = S.content.perspective;
    var scenario = S.content.screening.scenarios[S.data.selected_scenario];
    var area = window.W.textArea({
      minChars: conf.min_chars,
      maxChars: S.content.max_text_chars,
      value: S.data.perspective_text || '',
      placeholder: 'Describe what happened from your point of view…'
    });
    APP.mount([
      el('p', { className: 'muted', text: conf.intro }),
      el('div', { className: 'scenario' }, [
        el('div', { className: 'scenario-title', text: scenario.title }),
        el('p', { className: 'scenario-vignette', text: scenario.vignette })
      ]),
      el('div', { className: 'question-text', html: conf.question }),
      area.node,
      el('div', { className: 'reminder', html: conf.reminder }),
      APP.navActions(APP.primaryButton('Continue', function () {
        var text = area.field.value.trim();
        if (text.length < conf.min_chars) {
          APP.showError('Please write a bit more — at least ' + conf.min_chars +
                        ' characters.');
          return;
        }
        S.data.perspective_text = text;
        APP.advance();
      }))
    ]);
  };

  // --- The generation ------------------------------------------------------

  APP.screens.response_intro = function () {
    APP.mount([
      el('h1', { html: S.content.response_intro.text }),
      APP.navActions(APP.primaryButton('Continue', function () { APP.advance(); }))
    ]);
  };

  /* The review page: the participant sees EXACTLY what will be sent to the
   * model and may edit it before committing.
   *
   * The two halves are joined here rather than in the database, because the
   * whole point is that what gets sent is what they approved — composing it
   * server-side afterwards would silently discard their edits. start_response
   * now takes the finished text for the same reason.
   *
   * This is also the one-way door: pressing Confirm creates the job, and from
   * then on the input cannot be changed. Back is still available UP TO here. */
  APP.screens.review_input = function () {
    var conf = S.content.review;
    var selected = findScreening(S.data.selected_scenario);
    var composed = S.modelInput
      || [(selected && selected.description) || '', S.data.perspective_text || '']
           .filter(Boolean).join('\n\n');

    var area = window.W.textArea({
      minChars: conf.min_chars,
      maxChars: S.content.max_text_chars,
      value: composed,
      placeholder: 'What the AI chatbot will be told…'
    });

    APP.mount([
      el('h1', { text: conf.headline }),
      el('p', { className: 'help-text', text: conf.help }),
      area.node,
      APP.navActions(APP.primaryButton(conf.confirm, function (button) {
        var text = area.field.value.trim();
        if (text.length < conf.min_chars) {
          APP.showError('Please write a bit more — at least ' + conf.min_chars +
                        ' characters.');
          return;
        }
        // Remembered so Back-then-forward shows their edit rather than
        // recomposing from the original two answers.
        S.modelInput = text;
        button.disabled = true;
        APP.rpc('start_response', {
          p_token: S.token,
          p_scenario_key: S.data.selected_scenario,
          p_user_text: text
        }).then(function (job) {
          S.hasJob = true;
          APP.setStage('response');
          renderResponse(job);
        }).catch(function (error) {
          button.disabled = false;
          if (error.kind === 'unauthorized') { return APP.restart(); }
          APP.showError(error.message || 'Something went wrong. Please try again.');
        });
      }))
    ]);
  };

  APP.screens.response = function () {
    // Reached by navigation (Back from a DV page, or a refresh) rather than by
    // the start call, so ask the server where the job stands.
    APP.stopPolling();
    if (S.shownResponse) { return renderResponseReady({ text: S.shownResponse }); }
    APP.rpc('response_status', { p_token: S.token }).then(function (job) {
      renderResponse(job);
    }).catch(function (error) {
      if (error.kind === 'unauthorized') { return APP.restart(); }
      APP.showError(error.message || 'Could not reach the server.');
    });
  };

  function renderResponse(response) {
    APP.stopPolling();
    if (response.status === 'done') {
      S.shownResponse = response.text;
      APP.save();
      return renderResponseReady(response);
    }
    if (response.status === 'failed') { return renderResponseFailed(response); }
    return renderResponseWaiting(response);
  }

  function renderResponseWaiting(response) {
    var copy = S.content.response;
    var slowNotice = el('div', { className: 'loading-slow hidden', text: copy.loading_slow });
    APP.mount([
      el('div', { className: 'loading' }, [
        el('div', { className: 'spinner' }),
        el('div', { className: 'loading-headline', text: copy.loading_headline }),
        el('div', { className: 'loading-sub', text: copy.loading_subtext }),
        slowNotice
      ])
    ]);

    var slowTimer = window.setTimeout(function () {
      slowNotice.classList.remove('hidden');
    }, response.slow_notice_ms);

    APP.setPollTimer(window.setInterval(function () {
      APP.rpc('response_status', { p_token: S.token }).then(function (job) {
        if (job.status === 'pending' || job.status === 'running') { return; }
        window.clearTimeout(slowTimer);
        APP.stopPolling();
        renderResponse(job);
      }).catch(function (error) {
        // A transient poll failure is not fatal — keep polling. Only a lost
        // session forces a restart.
        if (error.kind === 'unauthorized') {
          window.clearTimeout(slowTimer);
          APP.stopPolling();
          APP.restart();
        }
      });
    }, response.poll_interval_ms));
  }

  function renderResponseReady(response) {
    var copy = S.content.response;
    APP.mount([
      el('div', { className: 'ai-response-label', text: 'AI chatbot' }),
      window.W.aiResponse(response.text),
      el('div', { className: 'reminder', html: copy.reminder }),
      el('div', { className: 'actions' }, [
        APP.primaryButton('Continue', function () { APP.advance(); })
      ])
    ]);
  }

  function renderResponseFailed(response) {
    var copy = S.content.response;
    var children = [el('h1', { text: copy.failure_headline })];
    if (response.can_retry) {
      children.push(el('p', { className: 'muted', text: copy.failure_subtext }));
      children.push(el('div', { className: 'actions' }, [
        APP.primaryButton(copy.failure_retry, function (button) {
          button.disabled = true;
          APP.rpc('retry_response', { p_token: S.token }).then(function (job) {
            renderResponse(job);
          }).catch(function (error) {
            button.disabled = false;
            APP.showError(error.message ||
              'Could not retry. Please wait a moment and try again.');
          });
        })
      ]));
    } else {
      children.push(el('p', { className: 'muted', text: copy.failure_exhausted }));
    }
    APP.mount([el('div', { className: 'terminal' }, children)]);
  }

  // --- Questionnaire, debrief, closing text --------------------------------

  APP.screens.questionnaire = function (page) {
    // The widget builders read and write S.data.answers directly, so a Back
    // navigation re-renders the page already filled in — no server round trip
    // to restore it, because it never left the browser.
    var blocks = [];
    var markers = [];
    page.questions.forEach(function (spec) {
      var built = window.W.question(spec, S.data.answers);
      blocks.push(built.node);
      markers.push(built.markMissing);
    });

    var header = [];
    // The DV pages keep the response visible, since every stem refers to it.
    if (page.page.indexOf('dv_') === 0) {
      header.push(el('div', { className: 'ai-response-label', text: 'AI chatbot' }));
      header.push(window.W.aiResponse(S.shownResponse || ''));
      header.push(el('div', { attrs: { style: 'height:28px' } }));
    }

    APP.mount(header.concat(blocks).concat([
      APP.navActions(APP.primaryButton('Continue', function () {
        var missing = markers.reduce(function (acc, mark) { return acc.concat(mark()); }, []);
        if (missing.length) {
          APP.showError('Please answer every question before continuing.');
          var flagged = document.querySelector('.flagged');
          if (flagged) { flagged.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
          return;
        }
        APP.save();
        APP.advance();
      }))
    ]));
  };

  APP.screens.debrief = function () {
    APP.mount([
      el('div', { className: 'prose', html: S.content.debrief }),
      APP.navActions(APP.primaryButton('Continue', function () { APP.advance(); }))
    ]);
  };

  APP.screens.additional_thoughts = function () {
    var conf = S.content.additional_thoughts;
    var area = window.W.textArea({
      minChars: 0,
      maxChars: S.content.max_text_chars,
      value: S.data.additional_thoughts || '',
      placeholder: 'Optional'
    });
    APP.mount([
      el('h1', { text: conf.headline }),
      el('div', { className: 'question-text', html: conf.question }),
      area.node,
      APP.navActions(APP.primaryButton('Finish', function () {
        S.data.additional_thoughts = area.field.value.trim();
        APP.setStage('complete');
      }))
    ]);
  };

  // --- Terminal: the one and only submit -----------------------------------

  /* This is where the entire study is posted — the first and only time the
   * server sees the participant's answers.
   *
   * All three terminal stages submit, not just `complete`: someone screened out
   * on their fourth scenario, or who declined consent, reached the end of their
   * path and is real data (the screening funnel is a reportable result). Only
   * participants who ABANDON send nothing. */
  APP.screens.terminal = function () {
    var conf = S.content.terminal[S.stage];
    var body = el('div', { className: 'terminal' }, []);
    APP.mount([body]);

    function showSaving() {
      window.W.clear(body);
      body.appendChild(el('p', { text: conf.message }));
      body.appendChild(el('div', { className: 'spinner' }));
      body.appendChild(el('p', { className: 'muted', text: 'Saving your responses…' }));
    }

    /* The completion code, shown only AFTER the submit succeeds.
     *
     * Nothing redirects and nothing links to Prolific. A redirect that was slow,
     * blocked or mistimed could strand someone who had finished the study with
     * no evidence they had; a code on screen cannot fail that way, and the
     * participant controls when they leave.
     *
     * Order matters: the code appears only once the data is safely stored. A
     * participant who has the code has, by construction, been recorded.
     *
     * A DECLINER GETS NO CODE. The bundle simply omits `code` for that terminal
     * state — someone who did not consent has not taken part, so there is
     * nothing to pay for. They are told to return the submission instead, which
     * is why the absence is handled here rather than rendering an empty box. */
    function finish() {
      APP.clearStorage();
      window.W.clear(body);
      body.appendChild(el('p', { text: conf.message }));
      if (conf.code) {
        body.appendChild(el('p', { className: 'code-label', text: S.content.code_label }));
        body.appendChild(el('div', { className: 'completion-code', text: conf.code }));
        body.appendChild(el('p', { className: 'muted', text: S.content.code_instruction }));
      } else {
        body.appendChild(el('p', { className: 'muted',
          text: S.content.no_code_instruction }));
      }
    }

    function attempt() {
      showSaving();
      APP.rpc('submit_study', { p_token: S.token, p_payload: S.data }).then(function () {
        S.submitted = true;
        finish();
      }).catch(function (error) {
        if (error.kind === 'unauthorized') {
          // The session is gone server-side; there is nothing left to submit to
          // and nothing the participant can do. Give them the code anyway —
          // withholding it would cost them their payment for our failure.
          finish();
          return;
        }
        window.W.clear(body);
        body.appendChild(el('h1', { text: 'We could not save your responses.' }));
        body.appendChild(el('p', { className: 'muted',
          text: error.message || 'Please try again — your answers are still here.' }));
        body.appendChild(el('div', { className: 'actions' }, [
          APP.primaryButton('Try again', function (button) {
            button.disabled = true;
            attempt();
          })
        ]));
      });
    }

    attempt();
  };

  APP.start();
})();
