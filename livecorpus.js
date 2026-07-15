// livecorpus.js
// Pulls a real, random slice of an actual large corpus live from Hugging
// Face's public `datasets-server` API (https://datasets-server.huggingface.co)
// — no server of ours, no API key, just a plain fetch() from the browser at
// run time. This is what actually lets the n-gram / embeddings / RNN models
// train on real data instead of just the small hand-written corpus.js.
//
// Google Books Ngrams is deliberately NOT included here: it's published as
// raw n-gram frequency counts (how many times "the cat" occurred), not
// sentences, so there's no text to tokenize into training sentences the way
// there is for the other three.

const LIVE_SOURCES = {
  wikitext103: {
    label: 'Wikitext-103',
    dataset: 'Salesforce/wikitext',
    config: 'wikitext-103-raw-v1',
    split: 'train',
    field: 'text',
    clean: cleanWikitextLines,
  },
  openwebtext: {
    label: 'OpenWebText',
    dataset: 'Skylion007/openwebtext',
    config: 'plain_text',
    split: 'train',
    field: 'text',
    clean: (t) => t,
  },
  wikipedia: {
    label: 'Wikipedia',
    dataset: 'wikimedia/wikipedia',
    config: '20231101.en',
    split: 'train',
    field: 'text',
    clean: (t) => t,
  },
};

// Wikitext-103's raw rows are one line per paragraph, including Wikipedia's
// " = Section Header = " lines and blank spacer lines — strip both so the
// tokenizer only ever sees actual prose sentences.
function cleanWikitextLines(text) {
  return text
    .split('\n')
    .filter((line) => {
      const t = line.trim();
      if (!t) return false;
      if (/^=+.*=+$/.test(t)) return false;
      return true;
    })
    .join(' ');
}

const HF_ROWS_ENDPOINT = 'https://datasets-server.huggingface.co/rows';

async function fetchHFRows(cfg, offset, length) {
  const url = `${HF_ROWS_ENDPOINT}?dataset=${encodeURIComponent(cfg.dataset)}&config=${encodeURIComponent(cfg.config)}&split=${encodeURIComponent(cfg.split)}&offset=${offset}&length=${length}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Hugging Face datasets-server returned ${res.status}`);
  return res.json();
}

// Returns a blob of real plain text pulled from a random slice of the
// named public dataset — ready to be tokenized into sentences exactly the
// way the built-in BACKGROUND_CORPUS text is (see corpus.js / app.js).
async function fetchLiveCorpusText(sourceKey, opts = {}) {
  const cfg = LIVE_SOURCES[sourceKey];
  if (!cfg) throw new Error(`Unknown corpus source: ${sourceKey}`);
  const numRows = opts.numRows || 40;
  const maxChars = opts.maxChars || 250000;

  // Probe how many rows the split has so the slice we grab isn't always
  // the same first few rows of the dataset.
  const probe = await fetchHFRows(cfg, 0, 1);
  const total = probe.num_rows_total || numRows;
  const offset = total > numRows ? Math.floor(Math.random() * (total - numRows)) : 0;

  const data = await fetchHFRows(cfg, offset, numRows);
  const rows = data.rows || [];
  let text = rows
    .map((r) => (r.row && r.row[cfg.field]) || '')
    .filter(Boolean)
    .join('\n\n');
  if (cfg.clean) text = cfg.clean(text);
  if (!text.trim()) throw new Error('sample came back empty');
  if (text.length > maxChars) text = text.slice(0, maxChars);
  return text;
}

if (typeof module !== 'undefined') {
  module.exports = { LIVE_SOURCES, fetchLiveCorpusText };
}
