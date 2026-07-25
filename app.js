// app.js
// Wires up the UI: for each blank, queries the REAL, live infini-gram API
// (see infinigram.js) for exact n-gram statistics over the 2.6T-token
// Dolma corpus — no local training, no local corpus, no story blending —
// and, optionally, runs a real small language model fully in-browser via
// transformers.js for a bidirectional-context comparison.
//
// Blanks are marked with <blank> (case-insensitive). A passage can
// contain more than one <blank> — each one gets its own results section,
// and every model predicts each blank independently from its own
// surrounding real words (other blanks are never treated as context,
// since their value is unknown).

const els = {
  prompt: document.getElementById('prompt'),
  runs: document.getElementById('runs'),
  runBtn: document.getElementById('run-btn'),
  status: document.getElementById('status'),
  results: document.getElementById('results'),
  picker: document.getElementById('model-picker'),
};

const MODEL_LABELS = { unigram: 'Unigram', bigram: 'Bigram', trigram: 'Trigram', fourgram: '4-gram' };
const MODEL_ORDER = { unigram: 1, bigram: 2, trigram: 3, fourgram: 4 };
const ORDER_LABELS = { 1: 'Unigram', 2: 'Bigram', 3: 'Trigram', 4: '4-gram' };

// Recognizes <blank> (any casing, optional whitespace inside the tag) as
// the canonical marker. A run of 3+ underscores is still accepted too.
const BLANK_RE = /<\s*blank\s*>|_{3,}/gi;

const CIRCLED = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩','⑪','⑫','⑬','⑭','⑮','⑯','⑰','⑱','⑲','⑳'];
function blankMarker(n) { return CIRCLED[n - 1] || `[${n}]`; }

function selectedModels() {
  return [...els.picker.querySelectorAll('input[type=checkbox]:checked')].map(i => i.value);
}

// Splits the passage on every blank marker into text.length+1 "chunks".
// chunks[i] is the real text that sits immediately before blank i.
// Returns null if there's no blank.
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

// The exact words (real case, real punctuation attached — e.g. "banana,")
// immediately before a blank, restricted to the current sentence only (so
// context never reaches back across a period into an unrelated earlier
// sentence). Deliberately NOT lowercased/stripped: infini-gram runs a real
// tokenizer server-side over whatever literal text you send it, so real
// punctuation (a comma before a blank in a list, for instance) is signal.
function rawLeftContextWords(chunkBefore) {
  const lastEnd = Math.max(
    chunkBefore.lastIndexOf('.'),
    chunkBefore.lastIndexOf('!'),
    chunkBefore.lastIndexOf('?')
  );
  const segment = lastEnd >= 0 ? chunkBefore.slice(lastEnd + 1) : chunkBefore;
  return segment.trim().split(/\s+/).filter(Boolean);
}

// Queries the live model for one blank at one requested order. Returns
// synthesized integer "counts" out of the exact real probabilities (top
// slice only) so this plugs into the existing count/totalRuns percentage
// renderer, without pretending anything was actually sampled — the
// underlying numbers are exact corpus statistics, not draws.
async function ngramPredictLive(order, chunks, blankIdx, displayScale) {
  const rawWords = rawLeftContextWords(chunks[blankIdx]);
  const { pairs, usedOrder, promptCnt, approx, backedOff } =
    await infiniNgramWithBackoff(rawWords, order);
  const freqPairs = pairs.slice(0, 32).map(([w, p]) => [w, Math.max(0, Math.round(p * displayScale))]);
  return { freqPairs, usedOrder, promptCnt, approx, backedOff };
}

// Short "...last few words <blank> first few words..." caption for a
// single blank's results section.
function localSnippet(chunks, blankIdx) {
  const beforeWords = chunks[blankIdx].trim().split(/\s+/).filter(Boolean);
  const afterWords = chunks[blankIdx + 1].trim().split(/\s+/).filter(Boolean);
  const before = beforeWords.slice(-8).join(' ');
  const after = afterWords.slice(0, 8).join(' ');
  const leadIn = beforeWords.length > 8 ? '…' : '';
  const leadOut = afterWords.length > 8 ? '…' : '';
  return `${leadIn}${before} <b>___</b> ${after}${leadOut}`.trim();
}

// The spinner lives in its own full-page overlay, separate from #results,
// so it never wipes out results mid-run.
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

function addNgramRow(blankIdx, blankCount, chunks, label, freqPairs, totalRuns, tag, backedOff) {
  const tbody = ensureBlankSection(blankIdx, blankCount, chunks);
  const top = freqPairs.slice(0, 8);
  const tr = document.createElement('tr');
  if (backedOff) tr.className = 'fallback-row';
  const modelTd = document.createElement('td');
  modelTd.className = 'model-cell';
  modelTd.innerHTML = `${label}<span class="model-tag">${tag}</span>`;
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
  const displayScale = Math.max(10, Math.min(5000, parseInt(els.runs.value, 10) || 1000));
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
  const blankIdxs = Array.from({ length: blankCount }, (_, i) => i);
  const ngramModels = models.filter(m => m in MODEL_ORDER);

  for (const key of ngramModels) {
    const order = MODEL_ORDER[key];

    for (const b of blankIdxs) {
      updateSpinnerMessage(
        blankCount > 1
          ? `Querying live ${MODEL_LABELS[key]} (infini-gram), blank ${b + 1} of ${blankCount}…`
          : `Querying live ${MODEL_LABELS[key]} (infini-gram)…`
      );
      await new Promise(r => setTimeout(r, 0));
      try {
        const { freqPairs, usedOrder, promptCnt, approx, backedOff } =
          await ngramPredictLive(order, chunks, b, displayScale);
        const seenNote = `context seen ${promptCnt.toLocaleString()}×${approx ? ' (approximate)' : ''}`;
        const tag = backedOff
          ? `n-gram · live exact counts, Dolma v1.7 (2.6T tokens) via infini-gram · no match at ${MODEL_LABELS[key]} order — backed off to ${ORDER_LABELS[usedOrder]} · ${seenNote}`
          : `n-gram · live exact counts, Dolma v1.7 (2.6T tokens) via infini-gram · ${seenNote}`;
        addNgramRow(b, blankCount, chunks, MODEL_LABELS[key], freqPairs, displayScale, tag, backedOff);
      } catch (err) {
        console.error(err);
        addErrorNote(`Couldn't reach the live n-gram model for ${MODEL_LABELS[key]}, blank ${b + 1} (${err.message || 'network error'}).`);
      }
    }
    ranLabels.push(MODEL_LABELS[key]);
  }

  if (models.includes('llm')) {
    showSpinner('Loading DistilBERT…');
    try {
      const pipe = await getLLMPipeline(msg => updateSpinnerMessage(msg));
      updateSpinnerMessage('Running DistilBERT on the full passage…');
      const maskedSentence = chunks.join('[MASK]').replace(/\s+/g, ' ').trim();
      const rawPredictions = await pipe(maskedSentence, { topk: 10 });
      const perBlank = blankCount === 1 && !Array.isArray(rawPredictions[0])
        ? [rawPredictions]
        : rawPredictions;
      for (const b of blankIdxs) {
        const predictions = perBlank[b] || [];
        addLLMRow(b, blankCount, chunks, predictions);
      }
      ranLabels.push('DistilBERT');
    } catch (err) {
      console.error(err);
      addErrorNote(`Couldn't load the in-browser LLM (${err.message || 'network or WebAssembly issue'}). This needs internet access to fetch model weights the first time.`);
    }
  }

  hideSpinner();
  els.status.textContent = ranLabels.length
    ? `Done — ran ${ranLabels.join(', ')} on ${blankCount} blank${blankCount === 1 ? '' : 's'} (n-grams: live from infini-gram/Dolma, no local training).`
    : `Nothing ran — check the browser console for errors.`;
  els.runBtn.disabled = false;
}

els.runBtn.addEventListener('click', run);