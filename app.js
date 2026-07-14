// app.js
// Wires up the UI: builds n-gram models from the story text, runs the
// sampling experiment for each selected model, and (optionally) runs
// a real small language model fully in-browser via transformers.js for
// a bidirectional-context comparison. Results are rendered as a single
// horizontally-scrollable comparison table.

const els = {
  prompt: document.getElementById('prompt'),
  runs: document.getElementById('runs'),
  runBtn: document.getElementById('run-btn'),
  status: document.getElementById('status'),
  results: document.getElementById('results'),
  picker: document.getElementById('model-picker'),
};

// The background corpus is always blended in at the same fixed strength —
// this isn't a user-facing knob, just an internal smoothing constant.
const BG_WEIGHT = 1;

const MODEL_LABELS = {
  unigram: 'Unigram',
  bigram: 'Bigram',
  trigram: 'Trigram',
  fourgram: '4-gram',
};
const MODEL_ORDER = { unigram: 1, bigram: 2, trigram: 3, fourgram: 4 };

function selectedModels() {
  return [...els.picker.querySelectorAll('input[type=checkbox]:checked')].map(i => i.value);
}

function splitOnBlank(text) {
  const idx = text.indexOf('___');
  if (idx === -1) return null;
  return { before: text.slice(0, idx), after: text.slice(idx + 3) };
}

function leftContextTokens(before) {
  const sents = toSentences(tokenize(before));
  return sents.length ? sents[sents.length - 1] : [];
}

// The spinner lives in its own full-page overlay, completely separate from
// the #results container. Earlier this reused #results, which meant every
// showSpinner() call wiped out the results table mid-run (losing rows and
// breaking multi-model runs) and only ever covered part of the page.
let overlayEl = null;
let overlayMsgEl = null;

function ensureOverlay() {
  if (overlayEl) return overlayEl;
  overlayEl = document.createElement('div');
  overlayEl.className = 'page-spinner-overlay';
  overlayEl.id = 'page-spinner-overlay';
  overlayEl.innerHTML = `
    <div class="page-spinner-box">
      <span class="spinner"></span>
      <span id="spinner-msg"></span>
    </div>
  `;
  document.body.appendChild(overlayEl);
  overlayMsgEl = overlayEl.querySelector('#spinner-msg');
  return overlayEl;
}

function showSpinner(message) {
  ensureOverlay();
  overlayMsgEl.textContent = message;
  overlayEl.classList.add('visible');
}

function updateSpinnerMessage(message) {
  if (overlayMsgEl) overlayMsgEl.textContent = message;
}

function hideSpinner() {
  if (overlayEl) overlayEl.classList.remove('visible');
}

let tableEl = null;
let tbodyEl = null;

function ensureResultsTable(preview) {
  if (tableEl) return tbodyEl;
  els.results.innerHTML = '';

  const heading = document.createElement('h2');
  heading.className = 'results-heading';
  heading.textContent = 'Results';
  els.results.appendChild(heading);

  const previewEl = document.createElement('div');
  previewEl.className = 'sentence-preview';
  previewEl.innerHTML = preview;
  els.results.appendChild(previewEl);

  const scroll = document.createElement('div');
  scroll.className = 'table-scroll';
  const table = document.createElement('table');
  table.className = 'results-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>Model</th>
        <th>#1</th><th>#2</th><th>#3</th><th>#4</th><th>#5</th>
        <th>#6</th><th>#7</th><th>#8</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  scroll.appendChild(table);
  els.results.appendChild(scroll);

  tableEl = table;
  tbodyEl = table.querySelector('tbody');
  return tbodyEl;
}

function predCell(word, pct) {
  const td = document.createElement('td');
  td.className = 'pred-cell';
  td.innerHTML = `${word} <span class="pct">${pct}%</span>`;
  return td;
}

function addNgramRow(label, freqPairs, totalRuns) {
  const tbody = ensureResultsTable(currentPreview);
  const top = freqPairs.slice(0, 8);
  const tr = document.createElement('tr');
  const modelTd = document.createElement('td');
  modelTd.className = 'model-cell';
  modelTd.innerHTML = `${label}<span class="model-tag">n-gram · ${totalRuns} samples, left-context only</span>`;
  tr.appendChild(modelTd);
  for (const [word, count] of top) {
    tr.appendChild(predCell(word, ((count / totalRuns) * 100).toFixed(1)));
  }
  for (let i = top.length; i < 8; i++) tr.appendChild(document.createElement('td'));
  tbody.appendChild(tr);
}

function addLLMRow(predictions) {
  const tbody = ensureResultsTable(currentPreview);
  const top = predictions.slice(0, 8);
  const tr = document.createElement('tr');
  const modelTd = document.createElement('td');
  modelTd.className = 'model-cell';
  modelTd.innerHTML = `DistilBERT<span class="model-tag">masked LLM · full left + right context</span>`;
  tr.appendChild(modelTd);
  for (const p of top) {
    tr.appendChild(predCell(p.token_str.trim(), (p.score * 100).toFixed(1)));
  }
  for (let i = top.length; i < 8; i++) tr.appendChild(document.createElement('td'));
  tbody.appendChild(tr);
}

let llmPipelinePromise = null;
async function getLLMPipeline(setStatus) {
  if (!llmPipelinePromise) {
    setStatus('Loading DistilBERT into your browser (first time only)…');
    llmPipelinePromise = import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2')
      .then(({ pipeline }) => pipeline('fill-mask', 'Xenova/distilbert-base-uncased'));
  }
  return llmPipelinePromise;
}

let currentPreview = '';

async function run() {
  const promptText = els.prompt.value.trim();
  const runs = Math.max(10, Math.min(5000, parseInt(els.runs.value, 10) || 1000));
  const models = selectedModels();

  const split = splitOnBlank(promptText);
  if (!split) {
    els.status.textContent = 'Add "___" (three underscores) somewhere in the prompt to mark the blank.';
    return;
  }
  if (models.length === 0) {
    els.status.textContent = 'Pick at least one model to run.';
    return;
  }

  els.runBtn.disabled = true;
  tableEl = null;
  tbodyEl = null;

  const ctxTokens = leftContextTokens(split.before);
  const previewMarker = '<b>___</b>';
  currentPreview = `${split.before.trim()} ${previewMarker} ${split.after.trim()}`.trim();

  showSpinner('Warming up…');

  const ngramModels = models.filter(m => m in MODEL_ORDER);
  for (const key of ngramModels) {
    updateSpinnerMessage(`Training ${MODEL_LABELS[key]} on the story text…`);
    await new Promise(r => setTimeout(r, 0)); // let status paint
    const model = buildModel(MODEL_ORDER[key], promptText.replace('___', ''), BACKGROUND_CORPUS, BG_WEIGHT);

    updateSpinnerMessage(`Sampling ${runs} guesses from the ${MODEL_LABELS[key]}…`);
    await new Promise(r => setTimeout(r, 0));
    const freqPairs = runSamplingExperiment(model, ctxTokens, runs);

    hideSpinner();
    addNgramRow(MODEL_LABELS[key], freqPairs, runs);
    if (ngramModels.indexOf(key) < ngramModels.length - 1 || models.includes('llm')) {
      showSpinner('Preparing next model…');
    }
  }

  if (models.includes('llm')) {
    showSpinner('Loading DistilBERT…');
    try {
      const pipe = await getLLMPipeline(msg => updateSpinnerMessage(msg));
      updateSpinnerMessage('Running DistilBERT on the full passage…');
      const maskedSentence = `${split.before}[MASK]${split.after}`.replace(/\s+/g, ' ').trim();
      const predictions = await pipe(maskedSentence, { topk: 10 });
      hideSpinner();
      addLLMRow(predictions);
    } catch (err) {
      console.error(err);
      hideSpinner();
      ensureResultsTable(currentPreview);
      const note = document.createElement('p');
      note.className = 'error-note';
      note.textContent = `Couldn't load the in-browser LLM (${err.message || 'network or WebAssembly issue'}). This needs internet access to fetch model weights the first time.`;
      els.results.appendChild(note);
    }
  }

  hideSpinner();
  els.status.textContent = `Done — ran ${ngramModels.length} n-gram model(s)${models.includes('llm') ? ' + LLM' : ''}.`;
  els.runBtn.disabled = false;
}

els.runBtn.addEventListener('click', run);
