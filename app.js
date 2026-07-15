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
  corpusSource: document.getElementById('corpus-source'),
};

// How many extra times the user's own story sentences are counted when
// training the LOCAL part of each n-gram/embeddings/RNN model. The n-gram
// models' background signal no longer comes from locally-trained text at
// all (see ngramPredict / infinigram.js) — it comes from live infini-gram
// queries against a real trillion-token corpus — but embeddings and RNN
// still blend the story against a locally-trained background corpus, so
// this weight still matters for those.
const STORY_WEIGHT = 3;

// How much to trust the local, story-only n-gram counts vs. the live
// infini-gram background distribution for a given context: alpha rises
// toward 1 as the exact context has been seen more times in the story
// itself, so a well-attested story-specific pattern still wins, exactly
// like the old STORY_WEIGHT-vs-background-corpus blend did — just with
// infini-gram standing in for the background corpus now. UNIGRAM_LOCAL_K
// is larger than NGRAM_LOCAL_K because "this word appeared in the story"
// is much weaker evidence on its own than "this exact 2-3 word context
// appeared".
const NGRAM_LOCAL_K = 2;
const UNIGRAM_LOCAL_K = 40;

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

// Raw (original-case, unsplit-into-sentences) words immediately before the
// blank, for querying infini-gram — the real Dolma corpus is case-sensitive
// and works on real text, not the lowercased/sentence-boundary-stripped
// tokens the local n-gram/embeddings/RNN trainers use.
function rawWordsBeforeBlank(chunkBefore) {
  return chunkBefore.replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
}

// Predicts one blank for one n-gram order by blending two sources:
//   1. A model trained ONLY on the user's own story (local, exact match)
//   2. A live infini-gram query against a real trillion-token corpus
// weighted by how many times the exact context has actually been seen in
// the story — the same "story pattern can win out" idea the old
// STORY_WEIGHT-vs-background-corpus design used, just with a real corpus
// standing in for the background instead of a few thousand local sentences.
// Returns [word, sampleCount] pairs so it can feed the existing table
// renderer unchanged.
async function ngramPredict(model, order, chunks, blankIdx, runs) {
  const ctxTokens = leftContextTokens(chunks[blankIdx]);
  const rawWords = rawWordsBeforeBlank(chunks[blankIdx]);

  let infiniPairs = [];
  try {
    const result = await infiniNgramWithBackoff(rawWords, order);
    infiniPairs = result.pairs;
  } catch (err) {
    console.error('infini-gram query failed:', err);
  }

  const localPairs = model.distribution(ctxTokens);
  const ctx = ctxTokens.slice(ctxTokens.length - (order - 1));
  // model was trained on the story repeated STORY_WEIGHT times (so the local
  // KN distribution/backoff has enough data to smooth well), which means raw
  // counts pulled from it are inflated by exactly that factor. Undo the
  // inflation here so `localCount` reflects how many times this context
  // genuinely occurred in what the user actually wrote — otherwise alpha
  // below over-trusts the local story (e.g. a phrase seen once for real
  // reads as 3 occurrences) and the blend barely uses live infini-gram data
  // at all, even for the unigram row.
  const rawLocalCount = order === 1
    ? (model._unigramTotal || 0)
    : (model.contextTotal[order].get(model._key(ctx)) || 0);
  const localCount = rawLocalCount / STORY_WEIGHT;
  const K = order === 1 ? UNIGRAM_LOCAL_K : NGRAM_LOCAL_K;
  const alpha = localCount / (localCount + K);

  const blended = new Map();
  const addMass = (pairs, weight) => {
    if (weight <= 0) return;
    for (const [w, p] of pairs) blended.set(w, (blended.get(w) || 0) + weight * p);
  };
  if (infiniPairs.length > 0) {
    addMass(infiniPairs, 1 - alpha);
    addMass(localPairs, alpha);
  } else {
    // infini-gram had nothing at all (or the request failed) — fall back
    // to pure local story statistics rather than dropping the row.
    addMass(localPairs, 1);
  }
  let total = 0;
  for (const p of blended.values()) total += p;
  const merged = [...blended.entries()]
    .map(([w, p]) => [w, total > 0 ? p / total : 0])
    .sort((a, b) => b[1] - a[1]);

  // Sample `runs` draws from the blended distribution so the results table
  // keeps its existing "N samples" presentation.
  const cum = [];
  let running = 0;
  for (const [w, p] of merged) { running += p; cum.push([w, running]); }
  const counts = new Map();
  const cumTotal = cum.length ? cum[cum.length - 1][1] : 0;
  for (let i = 0; i < runs; i++) {
    if (!cum.length) break;
    const r = Math.random() * cumTotal;
    let picked = cum[cum.length - 1][0];
    for (const [w, c] of cum) { if (r <= c) { picked = w; break; } }
    counts.set(picked, (counts.get(picked) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
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
  modelTd.innerHTML = `${label}<span class="model-tag">n-gram · story blended with a live infini-gram query over ${INFINIGRAM_LABEL} · ${totalRuns} samples, left-context only</span>`;
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
  // Training text for every model: the passage with blanks removed. Chunks
  // are joined with a sentence-ending period, NOT a bare space — joining
  // with just a space would make the text immediately before a blank and
  // the text immediately after it look like directly adjacent words to the
  // n-gram/embeddings/RNN trainers (e.g. "...a juicy" + "that is..." would
  // train the model on "juicy" being followed by "that", since that's
  // literally what the blank-deleted sentence looks like). Since this is
  // the user's own story, repeated via STORY_WEIGHT, that single artificial
  // adjacency was strong enough to dominate every prediction. A period
  // forces the tokenizer's sentence splitter to treat the two sides as
  // separate sentences during training, so a blank's own left/right text
  // is never fed back in as a false answer to itself. Left/right context
  // for prediction is unaffected — that's computed straight from the
  // individual chunks below, never from storyOnly.
  const storyOnly = chunks.map(c => c.trim()).filter(Boolean).join('. ');
  const stillToRun = () =>
    models.includes('embeddings') || models.includes('rnn') || models.includes('llm');

  // Resolve the background text ONCE for the whole run: either the small
  // built-in corpus, or a live sample of REAL documents pulled from the
  // infini-gram index (see infinigram.js's fetchInfiniGramCorpusText).
  // Every embeddings/RNN run below shares this same background text.
  // The n-gram models no longer need this at all — their background signal
  // comes live from infini-gram's next-token API instead (see ngramPredict)
  // — so this fetch is only relevant when embeddings and/or RNN are
  // selected.
  let backgroundText = (typeof BACKGROUND_CORPUS !== 'undefined') ? BACKGROUND_CORPUS : '';
  const corpusChoice = els.corpusSource ? els.corpusSource.value : 'builtin';
  const needsBackground = models.some(m => m === 'embeddings' || m === 'rnn');
  if (corpusChoice === 'infinigram' && needsBackground) {
    updateSpinnerMessage(`Fetching real documents from infini-gram (${INFINIGRAM_LABEL})…`);
    await new Promise(r => setTimeout(r, 0));
    try {
      backgroundText = await fetchInfiniGramCorpusText();
    } catch (err) {
      console.error(err);
      addErrorNote(`Couldn't fetch a live infini-gram sample (${err.message || 'network issue'}) — used the built-in corpus instead.`);
    }
  }

  const blankIdxs = Array.from({ length: blankCount }, (_, i) => i);

  const ngramModels = models.filter(m => m in MODEL_ORDER);
  for (const key of ngramModels) {
    updateSpinnerMessage(`Training ${MODEL_LABELS[key]} on the story text…`);
    await new Promise(r => setTimeout(r, 0)); // let status paint
    // No background text passed in at all — background weight 0 means this
    // model only ever sees the user's own story. Its background signal now
    // comes from live infini-gram queries in ngramPredict below.
    const model = buildModel(MODEL_ORDER[key], storyOnly, '', 0, STORY_WEIGHT);

    for (const b of blankIdxs) {
      updateSpinnerMessage(
        blankCount > 1
          ? `Querying infini-gram (${INFINIGRAM_LABEL}) for the ${MODEL_LABELS[key]}, blank ${b + 1} of ${blankCount}…`
          : `Querying infini-gram (${INFINIGRAM_LABEL}) for the ${MODEL_LABELS[key]}…`
      );
      await new Promise(r => setTimeout(r, 0));
      let freqPairs;
      try {
        freqPairs = await ngramPredict(model, MODEL_ORDER[key], chunks, b, runs);
      } catch (err) {
        console.error(err);
        freqPairs = runSamplingExperiment(model, leftContextTokens(chunks[b]), runs);
        addErrorNote(`Couldn't reach infini-gram for the ${MODEL_LABELS[key]} (${err.message || 'network issue'}) — showing story-only statistics instead.`);
      }
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
      const emb = buildEmbeddings(storyOnly, backgroundText);
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
      const rnnModel = trainRNN(storyOnly, backgroundText);
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
  const corpusNote = (corpusChoice === 'infinigram' && needsBackground)
    ? ` (background: ${backgroundText === BACKGROUND_CORPUS ? 'built-in, fallback' : INFINIGRAM_LABEL + ' live sample'})`
    : '';
  els.status.textContent = ranLabels.length
    ? `Done — ran ${ranLabels.join(', ')} on ${blankCount} blank${blankCount === 1 ? '' : 's'}${corpusNote}.`
    : `Nothing ran — check the browser console for errors.`;
  els.runBtn.disabled = false;
}

els.runBtn.addEventListener('click', run);