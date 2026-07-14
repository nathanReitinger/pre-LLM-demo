// app.js
// Wires up the UI: builds n-gram models from the story text, runs the
// 1000-sample experiment for each selected model, and (optionally) runs
// a real small language model fully in-browser via transformers.js for
// a bidirectional-context comparison.

const els = {
  context: document.getElementById('context'),
  target: document.getElementById('target'),
  runs: document.getElementById('runs'),
  bgWeight: document.getElementById('bg-weight'),
  runBtn: document.getElementById('run-btn'),
  status: document.getElementById('status'),
  results: document.getElementById('results'),
  tapePanel: document.getElementById('tape-panel'),
  tapeStrip: document.getElementById('tape-strip'),
  picker: document.getElementById('model-picker'),
};

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

function splitOnBlank(sentence) {
  const idx = sentence.indexOf('___');
  if (idx === -1) return null;
  return { before: sentence.slice(0, idx), after: sentence.slice(idx + 3) };
}

function leftContextTokens(fullContextText, before) {
  const combined = `${fullContextText} ${before}`;
  const sents = toSentences(tokenize(combined));
  return sents.length ? sents[sents.length - 1] : [];
}

async function animateTape(words) {
  els.tapePanel.hidden = false;
  els.tapeStrip.innerHTML = '';
  const sample = words.slice(0, 60);
  for (const w of sample) {
    const span = document.createElement('span');
    span.textContent = w;
    if (Math.random() < 0.15) span.classList.add('hit');
    els.tapeStrip.appendChild(span);
  }
  els.tapeStrip.scrollLeft = 0;
  // brief animated scroll for flavor, then settle
  await new Promise(r => setTimeout(r, 550));
}

function renderNgramResult(label, contextPreview, freqPairs, totalRuns) {
  const card = document.createElement('div');
  card.className = 'result-card';
  const top = freqPairs.slice(0, 10);
  const max = top.length ? top[0][1] : 1;
  card.innerHTML = `
    <h3>${label} <span class="model-tag">n-gram</span></h3>
    <p class="sub">${totalRuns} samples · left-context only · ${freqPairs.length} distinct words produced</p>
    <div class="sentence-preview">${contextPreview}</div>
    <div class="bars"></div>
  `;
  const bars = card.querySelector('.bars');
  for (const [word, count] of top) {
    const pct = ((count / totalRuns) * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-word">${word}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(count / max) * 100}%"></div></div>
      <div class="bar-pct">${pct}%</div>
    `;
    bars.appendChild(row);
  }
  els.results.appendChild(card);
}

function renderLLMResult(contextPreview, predictions) {
  const card = document.createElement('div');
  card.className = 'result-card llm';
  const max = predictions.length ? predictions[0].score : 1;
  card.innerHTML = `
    <h3>DistilBERT <span class="model-tag llm">masked LLM</span></h3>
    <p class="sub">single forward pass · full left + right context · top ${predictions.length} predictions</p>
    <div class="sentence-preview">${contextPreview}</div>
    <div class="bars"></div>
  `;
  const bars = card.querySelector('.bars');
  for (const p of predictions) {
    const pct = (p.score * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-word">${p.token_str.trim()}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(p.score / max) * 100}%"></div></div>
      <div class="bar-pct">${pct}%</div>
    `;
    bars.appendChild(row);
  }
  els.results.appendChild(card);
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
  const contextText = els.context.value.trim();
  const targetText = els.target.value.trim();
  const runs = Math.max(10, Math.min(5000, parseInt(els.runs.value, 10) || 1000));
  const bgWeight = parseInt(els.bgWeight.value, 10);
  const models = selectedModels();

  const split = splitOnBlank(targetText);
  if (!split) {
    els.status.textContent = 'Add "___" (three underscores) to the target sentence to mark the blank.';
    return;
  }
  if (models.length === 0) {
    els.status.textContent = 'Pick at least one model to run.';
    return;
  }

  els.results.innerHTML = '';
  els.runBtn.disabled = true;

  const ctxTokens = leftContextTokens(contextText, split.before);
  const previewBefore = `${contextText} ${split.before}`.trim();
  const previewMarker = '<b>___</b>';
  const previewAfter = split.after.trim();
  const preview = `${previewBefore} ${previewMarker} ${previewAfter}`;

  const ngramModels = models.filter(m => m in MODEL_ORDER);
  for (const key of ngramModels) {
    const order = MODEL_ORDER[key];
    els.status.textContent = `Training ${MODEL_LABELS[key]} on the story text…`;
    await new Promise(r => setTimeout(r, 0)); // let status paint
    const model = buildModel(order, contextText + ' ' + targetText.replace('___', ''), BACKGROUND_CORPUS, bgWeight);

    els.status.textContent = `Sampling ${runs} guesses from the ${MODEL_LABELS[key]}…`;
    await new Promise(r => setTimeout(r, 0));
    const freqPairs = runSamplingExperiment(model, ctxTokens, runs);
    await animateTape(freqPairs.flatMap(([w, c]) => Array(Math.min(c, 3)).fill(w)));
    renderNgramResult(MODEL_LABELS[key], preview, freqPairs, runs);
  }

  els.tapePanel.hidden = true;

  if (models.includes('llm')) {
    try {
      const pipe = await getLLMPipeline(msg => (els.status.textContent = msg));
      els.status.textContent = 'Running DistilBERT on the full passage…';
      const maskedSentence = `${contextText} ${split.before}[MASK]${split.after}`.replace(/\s+/g, ' ').trim();
      const predictions = await pipe(maskedSentence, { topk: 10 });
      renderLLMResult(preview, predictions);
    } catch (err) {
      console.error(err);
      const card = document.createElement('div');
      card.className = 'result-card llm';
      card.innerHTML = `<h3>DistilBERT <span class="model-tag llm">masked LLM</span></h3>
        <p class="sub">Couldn't load the in-browser model (${err.message || 'network or WebAssembly issue'}).
        This needs internet access to fetch model weights the first time.</p>`;
      els.results.appendChild(card);
    }
  }

  els.status.textContent = `Done — ran ${ngramModels.length} n-gram model(s)${models.includes('llm') ? ' + LLM' : ''}.`;
  els.runBtn.disabled = false;
}

els.runBtn.addEventListener('click', run);
