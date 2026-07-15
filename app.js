// app.js
// Wires up the UI: builds n-gram models from the story text, runs the
// sampling experiment for each selected model, and (optionally) runs
// a real small language model fully in-browser via transformers.js for
// a bidirectional-context comparison.
//
// Blanks are marked with <blank> (case-insensitive). A passage can
// contain more than one <blank> — each one gets its own results
// section below, and every model predicts each blank independently
// from its own surrounding real words (other blanks in the passage
// are never treated as context, since their value is unknown).

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
// How many extra times the user's own story sentences are counted relative
// to the background corpus. The background corpus is now several thousand
// sentences (up from a few hundred), so without this the story's own
// specific bigrams/trigrams would get statistically swamped and the model
// would just answer from generic background patterns instead of the story.
const STORY_WEIGHT = 3;

const MODEL_LABELS = {
  unigram: 'Unigram',
  bigram: 'Bigram',
  trigram: 'Trigram',
  fourgram: '4-gram',
};
const MODEL_ORDER = { unigram: 1, bigram: 2, trigram: 3, fourgram: 4 };

// Recognizes <blank> (any casing, optional whitespace inside the tag) as
// the canonical marker. A run of 3+ underscores is still accepted too, so
// passages written for the old marker keep working.
const BLANK_RE = /<\s*blank\s*>|_{3,}/gi;

const CIRCLED = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩','⑪','⑫','⑬','⑭','⑮','⑯','⑰','⑱','⑲','⑳'];
function blankMarker(n) { return CIRCLED[n - 1] || `[${n}]`; }

function selectedModels() {
  return [...els.picker.querySelectorAll('input[type=checkbox]:checked')].map(i => i.value);
}

// Splits the passage on every blank marker into text.length+1 "chunks".
// chunks[i] is the real text that sits immediately before blank i (and
// chunks[0] is everything before the first blank); chunks[chunks.length-1]
// is everything after the last blank. Returns null if there's no blank.
function splitOnBlanks(text) {
  const matches = [...text.matchAll(BLANK_RE)];
  if (!matches.length) return null;
  const chunks = [];
  let last = 0;
  for (const m of matches) {
    chunks.push(text.slice(last, m.index));
    last = m.index + m[0].length;
  }
  chunks.push(text.slice(last));
  return chunks;
}

function leftContextTokens(chunkBefore) {
  const sents = toSentences(tokenize(chunkBefore));
  return sents.length ? sents[sents.length - 1] : [];
}

function rightContextTokens(chunkAfter) {
  const sents = toSentences(tokenize(chunkAfter));
  return sents.length ? sents[0] : [];
}

// Short "...last few words <blank> first few words..." caption for a
// single blank's results section, so it's obvious which hole a table
// belongs to without rereading the whole passage.
function localSnippet(chunks, blankIdx) {
  const beforeWords = chunks[blankIdx].trim().split(/\s+/).filter(Boolean);
  const afterWords = chunks[blankIdx + 1].trim().split(/\s+/).filter(Boolean);
  const before = beforeWords.slice(-8).join(' ');
  const after = afterWords.slice(0, 8).join(' ');
  const leadIn = beforeWords.length > 8 ? '…' : '';
  const leadOut = afterWords.length > 8 ? '…' : '';
  return `${leadIn}${before} <b>___</b> ${after}${leadOut}`.trim();
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

// One table (and its own tbody) per blank, keyed by blank index.
let blankSections = new Map();
let resultsHeaderBuilt = false;

function ensureResultsHeader(preview) {
  if (resultsHeaderBuilt) return;
  els.results.innerHTML = '';
  const heading = document.createElement('h2');
  heading.className = 'results-heading';
  heading.textContent = 'Results';
  els.results.appendChild(heading);

  const previewEl = document.createElement('div');
  previewEl.className = 'sentence-preview';
  previewEl.innerHTML = preview;
  els.results.appendChild(previewEl);

  resultsHeaderBuilt = true;
}

function ensureBlankSection(blankIdx, blankCount, chunks) {
  if (blankSections.has(blankIdx)) return blankSections.get(blankIdx).tbody;

  const section = document.createElement('div');
  section.className = 'blank-section';

  if (blankCount > 1) {
    const sub = document.createElement('h3');
    sub.className = 'blank-heading';
    sub.innerHTML = `Blank ${blankMarker(blankIdx + 1)}`;
    section.appendChild(sub);

    const caption = document.createElement('p');
    caption.className = 'blank-caption';
    caption.innerHTML = localSnippet(chunks, blankIdx);
    section.appendChild(caption);
  }

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
  section.appendChild(scroll);
  els.results.appendChild(section);

  const tbody = table.querySelector('tbody');
  blankSections.set(blankIdx, { table, tbody });
  return tbody;
}

function predCell(word, pct) {
  const td = document.createElement('td');
  td.className = 'pred-cell';
  td.innerHTML = `${word} <span class="pct">${pct}%</span>`;
  return td;
}

function addNgramRow(blankIdx, blankCount, chunks, label, freqPairs, totalRuns) {
  const tbody = ensureBlankSection(blankIdx, blankCount, chunks);
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

function addLLMRow(blankIdx, blankCount, chunks, predictions) {
  const tbody = ensureBlankSection(blankIdx, blankCount, chunks);
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

// Shared renderer for any model that already produces [word, probability]
// pairs (0-1 range) — used by the embeddings and RNN models.
function addVectorRow(blankIdx, blankCount, chunks, label, tag, pairs) {
  const tbody = ensureBlankSection(blankIdx, blankCount, chunks);
  const top = pairs.slice(0, 8);
  const tr = document.createElement('tr');
  const modelTd = document.createElement('td');
  modelTd.className = 'model-cell';
  modelTd.innerHTML = `${label}<span class="model-tag">${tag}</span>`;
  tr.appendChild(modelTd);
  for (const [word, p] of top) {
    tr.appendChild(predCell(word, (p * 100).toFixed(1)));
  }
  for (let i = top.length; i < 8; i++) tr.appendChild(document.createElement('td'));
  tbody.appendChild(tr);
}

function addErrorNote(message) {
  const note = document.createElement('p');
  note.className = 'error-note';
  note.textContent = message;
  els.results.appendChild(note);
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

async function run() {
  const promptText = els.prompt.value.trim();
  const runs = Math.max(10, Math.min(5000, parseInt(els.runs.value, 10) || 1000));
  const models = selectedModels();

  const chunks = splitOnBlanks(promptText);
  if (!chunks) {
    els.status.textContent = 'Mark at least one word to guess with <blank> somewhere in the prompt.';
    return;
  }
  if (models.length === 0) {
    els.status.textContent = 'Pick at least one model to run.';
    return;
  }

  const blankCount = chunks.length - 1;

  els.runBtn.disabled = true;
  blankSections = new Map();
  resultsHeaderBuilt = false;

  // Whole-passage preview with every blank replaced by a numbered marker,
  // so it's clear at a glance which table below corresponds to which hole.
  const previewHtml = chunks
    .map(c => c.trim())
    .reduce((acc, chunk, i) => {
      if (i === 0) return chunk;
      return `${acc} <b class="blank-marker">${blankMarker(i)}</b> ${chunk}`;
    }, '')
    .trim();
  ensureResultsHeader(previewHtml);

  showSpinner('Warming up…');

  const ranLabels = [];
  // Training text for every model: the passage with blanks simply removed
  // (a blank is a hole, not a word — it should never itself become part of
  // any model's vocabulary or training counts).
  const storyOnly = chunks.join(' ');
  const stillToRun = () =>
    models.includes('embeddings') || models.includes('rnn') || models.includes('llm');

  const blankIdxs = Array.from({ length: blankCount }, (_, i) => i);

  const ngramModels = models.filter(m => m in MODEL_ORDER);
  for (const key of ngramModels) {
    updateSpinnerMessage(`Training ${MODEL_LABELS[key]} on the story text…`);
    await new Promise(r => setTimeout(r, 0)); // let status paint
    const model = buildModel(MODEL_ORDER[key], storyOnly, BACKGROUND_CORPUS, BG_WEIGHT, STORY_WEIGHT);

    for (const b of blankIdxs) {
      updateSpinnerMessage(
        blankCount > 1
          ? `Sampling ${runs} guesses from the ${MODEL_LABELS[key]} for blank ${b + 1} of ${blankCount}…`
          : `Sampling ${runs} guesses from the ${MODEL_LABELS[key]}…`
      );
      await new Promise(r => setTimeout(r, 0));
      const ctxTokens = leftContextTokens(chunks[b]);
      const freqPairs = runSamplingExperiment(model, ctxTokens, runs);
      addNgramRow(b, blankCount, chunks, MODEL_LABELS[key], freqPairs, runs);
    }

    hideSpinner();
    ranLabels.push(MODEL_LABELS[key]);
    if (ngramModels.indexOf(key) < ngramModels.length - 1 || stillToRun()) {
      showSpinner('Preparing next model…');
    }
  }

  if (models.includes('embeddings')) {
    showSpinner('Building word vectors from the story…');
    await new Promise(r => setTimeout(r, 0));
    try {
      const emb = buildEmbeddings(storyOnly, BACKGROUND_CORPUS);
      for (const b of blankIdxs) {
        updateSpinnerMessage(
          blankCount > 1
            ? `Averaging context vectors for blank ${b + 1} of ${blankCount}…`
            : 'Averaging context vectors and ranking neighbors…'
        );
        await new Promise(r => setTimeout(r, 0));
        const leftTokens = leftContextTokens(chunks[b]);
        const rightTokens = rightContextTokens(chunks[b + 1]);
        const preds = embeddingPredict(emb, leftTokens, rightTokens, 8);
        addVectorRow(b, blankCount, chunks, 'Embeddings', 'PPMI + SVD · vector similarity, left + right context', preds);
      }
      hideSpinner();
      ranLabels.push('Embeddings');
    } catch (err) {
      console.error(err);
      hideSpinner();
      addErrorNote(`Couldn't build the embedding model (${err.message || 'error'}).`);
    }
    if (models.includes('rnn') || models.includes('llm')) showSpinner('Preparing next model…');
  }

  if (models.includes('rnn')) {
    showSpinner('Training a tiny RNN on the story text…');
    await new Promise(r => setTimeout(r, 0));
    try {
      const rnnModel = trainRNN(storyOnly, BACKGROUND_CORPUS);
      for (const b of blankIdxs) {
        updateSpinnerMessage(
          blankCount > 1
            ? `Running the RNN forward for blank ${b + 1} of ${blankCount}…`
            : 'Running the RNN forward over the left context…'
        );
        await new Promise(r => setTimeout(r, 0));
        const ctxTokens = leftContextTokens(chunks[b]);
        const preds = rnnPredict(rnnModel, ctxTokens, 8);
        addVectorRow(b, blankCount, chunks, 'RNN', 'live-trained · running hidden state, left context only', preds);
      }
      hideSpinner();
      ranLabels.push('RNN');
    } catch (err) {
      console.error(err);
      hideSpinner();
      addErrorNote(`Couldn't train the RNN (${err.message || 'error'}).`);
    }
    if (models.includes('llm')) showSpinner('Preparing next model…');
  }

  if (models.includes('llm')) {
    showSpinner('Loading DistilBERT…');
    try {
      const pipe = await getLLMPipeline(msg => updateSpinnerMessage(msg));
      updateSpinnerMessage('Running DistilBERT on the full passage…');
      // Replace every blank with a single [MASK] token in one pass, rather
      // than looping per blank — a masked LM predicts all mask positions
      // from one shared bidirectional forward pass over the *whole*
      // passage, so this is both more correct (every blank actually gets
      // the real text of every other blank position as context, not a
      // leftover stray marker) and faster than calling the pipeline once
      // per blank.
      const maskedSentence = chunks.join('[MASK]').replace(/\s+/g, ' ').trim();
      const rawPredictions = await pipe(maskedSentence, { topk: 10 });
      // transformers.js returns a flat array of predictions when there is
      // exactly one [MASK], and an array of arrays (one per mask, in the
      // order the masks appear) when there is more than one.
      const perBlank = blankCount === 1 && !Array.isArray(rawPredictions[0])
        ? [rawPredictions]
        : rawPredictions;
      for (const b of blankIdxs) {
        const predictions = perBlank[b] || [];
        addLLMRow(b, blankCount, chunks, predictions);
      }
      hideSpinner();
      ranLabels.push('DistilBERT');
    } catch (err) {
      console.error(err);
      hideSpinner();
      addErrorNote(`Couldn't load the in-browser LLM (${err.message || 'network or WebAssembly issue'}). This needs internet access to fetch model weights the first time.`);
    }
  }

  hideSpinner();
  els.status.textContent = ranLabels.length
    ? `Done — ran ${ranLabels.join(', ')} on ${blankCount} blank${blankCount === 1 ? '' : 's'}.`
    : `Nothing ran — check the browser console for errors.`;
  els.runBtn.disabled = false;
}

els.runBtn.addEventListener('click', run);
