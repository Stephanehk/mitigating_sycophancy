/* Reusable DOM builders for the sycophancy user study.
 *
 * Exposes `window.W`. Two rules hold throughout:
 *
 *  1. Study-authored copy (from content.py / questionnaires.py) is trusted
 *     HTML and set with innerHTML, because the source survey's text carries
 *     its own <strong>/<br> markup.
 *  2. Anything a participant typed is set with textContent, never innerHTML.
 *     `el()` takes a `text` option for exactly this, and no participant string
 *     is ever passed through `html`.
 *
 * The Likert matrix renders twice — a table for wide viewports and stacked
 * cards for narrow ones — with both views bound to the same `state` object, so
 * a participant who rotates their phone mid-page keeps their answers.
 */
(function () {
  'use strict';

  function el(tag, options, children) {
    var node = document.createElement(tag);
    options = options || {};
    if (options.className) { node.className = options.className; }
    if (options.text !== undefined && options.text !== null) { node.textContent = options.text; }
    if (options.html !== undefined && options.html !== null) { node.innerHTML = options.html; }
    if (options.attrs) {
      Object.keys(options.attrs).forEach(function (key) {
        node.setAttribute(key, options.attrs[key]);
      });
    }
    if (options.on) {
      Object.keys(options.on).forEach(function (event) {
        node.addEventListener(event, options.on[event]);
      });
    }
    (children || []).forEach(function (child) {
      if (child === null || child === undefined) { return; }
      node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    });
    return node;
  }

  function clear(node) {
    while (node.firstChild) { node.removeChild(node.firstChild); }
  }

  /* A vertical list of mutually exclusive options.
   * `onPick` receives the chosen value. `state.value` holds the selection. */
  function choiceList(options, state, onPick) {
    var wrap = el('div', { className: 'choices', attrs: { role: 'radiogroup' } });
    options.forEach(function (option) {
      var input = el('input', {
        attrs: { type: 'radio', name: state.name, value: String(option.value) }
      });
      input.checked = state.value === option.value;
      var row = el('label', {
        className: 'choice' + (state.value === option.value ? ' selected' : '')
      }, [input, el('span', { text: option.label })]);
      row.addEventListener('click', function (event) {
        if (event.target !== input) { input.checked = true; }
        state.value = option.value;
        Array.prototype.forEach.call(wrap.children, function (child) {
          child.classList.remove('selected');
        });
        row.classList.add('selected');
        if (onPick) { onPick(option.value); }
      });
      wrap.appendChild(row);
    });
    return wrap;
  }

  /* A horizontal 7-point row with labelled endpoints (QSF "MC / SAHR"). */
  function scaleRow(question, state) {
    var low = '';
    var high = '';
    question.options.forEach(function (option) {
      if (option.scale_point === 1 && option.endpoint) { low = option.endpoint; }
      if (option.scale_point === question.options.length && option.endpoint) {
        high = option.endpoint;
      }
    });

    var row = el('div', { className: 'scale-options' });
    question.options.forEach(function (option) {
      var input = el('input', {
        attrs: { type: 'radio', name: question.id, value: String(option.value) }
      });
      input.checked = state[question.id] === option.value;
      var point = el('label', {
        className: 'scale-point' + (state[question.id] === option.value ? ' selected' : '')
      }, [input, el('span', { text: option.label })]);
      point.addEventListener('click', function () {
        state[question.id] = option.value;
        Array.prototype.forEach.call(row.children, function (child) {
          child.classList.remove('selected');
        });
        point.classList.add('selected');
      });
      row.appendChild(point);
    });

    return el('div', { className: 'scale-row' }, [
      el('div', { className: 'scale-endpoints' }, [
        el('span', { text: low }), el('span', { text: high })
      ]),
      row
    ]);
  }

  function optionHeader(option) {
    /* Column header text: the label, plus its numeric sublabel when the QSF
     * gave one (e.g. "Strongly disagree" over "1"). */
    return option.sublabel ? option.label + '\n' + option.sublabel : option.label;
  }

  /* A Likert grid. Returns {node, markMissing} — `markMissing` highlights the
   * rows the participant left blank when they try to advance. */
  function matrix(question, state) {
    var rowNodes = {};
    var cardNodes = {};

    var headRow = el('tr', {}, [el('th', { className: 'item-col', text: '' })]);
    question.options.forEach(function (option) {
      headRow.appendChild(el('th', { text: optionHeader(option) }));
    });

    var body = el('tbody');
    question.items.forEach(function (item) {
      var tr = el('tr', { className: 'unanswered' }, [
        el('td', { className: 'item-label', text: item.label })
      ]);
      question.options.forEach(function (option) {
        var input = el('input', {
          attrs: { type: 'radio', name: item.id, value: String(option.value) }
        });
        input.checked = state[item.id] === option.value;
        input.addEventListener('change', function () {
          state[item.id] = option.value;
          tr.classList.remove('unanswered', 'flagged');
          syncCard(item, option.value);
        });
        tr.appendChild(el('td', {}, [input]));
      });
      if (state[item.id] !== undefined) { tr.classList.remove('unanswered'); }
      rowNodes[item.id] = tr;
      body.appendChild(tr);
    });

    var table = el('table', { className: 'matrix' }, [el('thead', {}, [headRow]), body]);

    /* Narrow-viewport mirror of the same state. */
    var cards = el('div', { className: 'matrix-cards' });
    question.items.forEach(function (item) {
      var optionNodes = [];
      var list = el('div', { className: 'matrix-card-options' });
      question.options.forEach(function (option) {
        var input = el('input', {
          attrs: { type: 'radio', name: item.id + '__m', value: String(option.value) }
        });
        input.checked = state[item.id] === option.value;
        var label = el('label', {
          className: 'matrix-card-option' + (state[item.id] === option.value ? ' selected' : '')
        }, [input, el('span', { text: option.sublabel
          ? option.label + ' (' + option.sublabel + ')' : option.label })]);
        label.addEventListener('click', function () {
          state[item.id] = option.value;
          optionNodes.forEach(function (node) { node.classList.remove('selected'); });
          label.classList.add('selected');
          card.classList.remove('flagged');
          syncRow(item, option.value);
        });
        optionNodes.push(label);
        list.appendChild(label);
      });
      var card = el('div', { className: 'matrix-card' }, [
        el('div', { className: 'matrix-card-label', text: item.label }), list
      ]);
      cardNodes[item.id] = { card: card, options: optionNodes };
      cards.appendChild(card);
    });

    function syncCard(item, value) {
      var entry = cardNodes[item.id];
      if (!entry) { return; }
      entry.card.classList.remove('flagged');
      entry.options.forEach(function (node, index) {
        var matches = question.options[index].value === value;
        node.classList.toggle('selected', matches);
        node.querySelector('input').checked = matches;
      });
    }

    function syncRow(item, value) {
      var tr = rowNodes[item.id];
      if (!tr) { return; }
      tr.classList.remove('unanswered', 'flagged');
      var inputs = tr.querySelectorAll('input');
      question.options.forEach(function (option, index) {
        inputs[index].checked = option.value === value;
      });
    }

    function markMissing() {
      var missing = [];
      question.items.forEach(function (item) {
        if (state[item.id] === undefined) {
          missing.push(item.id);
          rowNodes[item.id].classList.add('flagged');
          cardNodes[item.id].card.classList.add('flagged');
        }
      });
      return missing;
    }

    return { node: el('div', {}, [table, cards]), markMissing: markMissing };
  }

  /* A checkbox list, with support for the QSF's "Other" text entry and the
   * exclusive "None of the Above" option. */
  function multiSelect(question, state) {
    state[question.id] = state[question.id] || [];
    var wrap = el('div', { className: 'choices' });
    var rows = [];
    var otherInput = null;

    question.options.forEach(function (option) {
      var input = el('input', { attrs: { type: 'checkbox', value: String(option.value) } });
      input.checked = state[question.id].indexOf(option.value) !== -1;
      var row = el('label', {
        className: 'choice' + (input.checked ? ' selected' : '')
      }, [input, el('span', { text: option.label })]);

      input.addEventListener('change', function () {
        var selected = state[question.id];
        if (input.checked) {
          if (option.exclusive) {
            selected = [option.value];
          } else {
            var exclusiveValues = question.options
              .filter(function (o) { return o.exclusive; })
              .map(function (o) { return o.value; });
            selected = selected.filter(function (v) { return exclusiveValues.indexOf(v) === -1; });
            selected = selected.concat([option.value]);
          }
        } else {
          selected = selected.filter(function (v) { return v !== option.value; });
        }
        state[question.id] = selected;
        rows.forEach(function (entry) {
          var on = selected.indexOf(entry.value) !== -1;
          entry.input.checked = on;
          entry.row.classList.toggle('selected', on);
        });
        if (otherInput) {
          var otherOn = selected.indexOf(otherInput.value) !== -1;
          otherInput.node.classList.toggle('hidden', !otherOn);
        }
      });

      rows.push({ value: option.value, input: input, row: row });
      wrap.appendChild(row);

      if (option.text_entry) {
        var field = el('input', {
          className: 'hidden',
          attrs: { type: 'text', placeholder: 'Please specify' }
        });
        field.value = state[question.id + '__other_text'] || '';
        field.addEventListener('input', function () {
          state[question.id + '__other_text'] = field.value;
        });
        if (input.checked) { field.classList.remove('hidden'); }
        otherInput = { node: field, value: option.value };
        wrap.appendChild(field);
      }
    });

    return wrap;
  }

  /* Render one question of any shape. Returns {node, markMissing}.
   *
   * `markMissing` reports the question ids left blank and highlights them, so
   * the participant is shown WHICH question is missing rather than a bare
   * error.
   *
   * THIS IS THE ONLY THING ENFORCING REQUIREDNESS. The server no longer
   * re-rejects an incomplete page — it cannot, because it does not see answers
   * until the study is over, and refusing a finished session over one blank
   * cell would lose the participant entirely. It records the gap in
   * `missing_keys` instead (flow.validate_questionnaire_answers), which is why
   * a non-zero `n_missing_answers` in the export means a bug in this function.
   *
   * Optional questions (spec.required false) never block, which is why the
   * check is gated on it — every question in questionnaires.py is currently
   * required, but the gate keeps the client honest if that changes. */
  function question(spec, state) {
    var header = el('div', { className: 'question-text', html: spec.text });

    if (spec.type === 'matrix') {
      var built = matrix(spec, state);
      return {
        node: el('div', { className: 'question-block' }, [header, built.node]),
        markMissing: spec.required ? built.markMissing : function () { return []; }
      };
    }

    var body;
    var isBlank;
    if (spec.type === 'multi') {
      body = multiSelect(spec, state);
      // A multi-select is answered when at least one box is ticked.
      isBlank = function () {
        var selected = state[spec.id];
        return !selected || selected.length === 0;
      };
    } else if (spec.type === 'vertical') {
      body = choiceList(spec.options, { name: spec.id, value: state[spec.id] },
        function (value) { state[spec.id] = value; block.classList.remove('flagged'); });
      isBlank = function () { return state[spec.id] === undefined; };
    } else {
      body = scaleRow(spec, state);
      isBlank = function () { return state[spec.id] === undefined; };
    }

    var block = el('div', { className: 'question-block' }, [header, body]);
    // Clearing the flag on any interaction keeps the highlight from lingering
    // after the participant fixes the question. The vertical list clears its own
    // above, because choiceList owns that click.
    block.addEventListener('click', function () {
      if (!isBlank()) { block.classList.remove('flagged'); }
    });

    return {
      node: block,
      markMissing: function () {
        if (!spec.required || !isBlank()) { return []; }
        block.classList.add('flagged');
        return [spec.id];
      }
    };
  }

  /* A textarea with a live character counter that turns red below `minChars`. */
  function textArea(options) {
    var area = el('textarea', {
      attrs: { placeholder: options.placeholder || 'Type your answer here…',
               maxlength: String(options.maxChars || 4000) }
    });
    area.value = options.value || '';
    var counter = el('div', { className: 'char-count' });

    function update() {
      var length = area.value.trim().length;
      var min = options.minChars || 0;
      counter.textContent = min > 0 && length < min
        ? length + ' / ' + min + ' characters minimum'
        : length + ' characters';
      counter.classList.toggle('short', min > 0 && length < min);
      if (options.onInput) { options.onInput(area.value); }
    }

    area.addEventListener('input', update);
    update();
    return { node: el('div', {}, [area, counter]), field: area };
  }

  window.W = {
    el: el,
    clear: clear,
    choiceList: choiceList,
    matrix: matrix,
    question: question,
    textArea: textArea
  };
})();
