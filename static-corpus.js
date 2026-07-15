// static-corpus.js
// The ONE background-text source for the embeddings/RNN models: a real text
// corpus (~1GB) split into small chunk files committed directly into this
// repo under static-corpus/ (each safely under GitHub's 100MB file limit,
// no Git LFS). GitHub Pages serves these as plain static files, so a
// same-origin fetch() here is as reliable as loading style.css — no
// third-party API, no rate limits, no gating.
//
// We only need a few hundred KB of real prose per run, not the whole
// corpus, so this fetches ONE random chunk file and takes a random slice
// out of it rather than downloading everything.

const STATIC_CORPUS_DIR = 'static-corpus';
const STATIC_CORPUS_MANIFEST = `${STATIC_CORPUS_DIR}/manifest.json`;
// Fallback list used only if manifest.json doesn't exist — matches the
// naming produced by `split` + the rename loop in the setup instructions
// (chunk-0001.txt, chunk-0002.txt, ...). Adjust FALLBACK_CHUNK_COUNT if you
// have a different number of chunk files.
const FALLBACK_CHUNK_COUNT = 12;

let chunkListPromise = null;

async function getChunkList() {
  if (chunkListPromise) return chunkListPromise;

  chunkListPromise = (async () => {
    try {
      const res = await fetch(STATIC_CORPUS_MANIFEST);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.chunks) && data.chunks.length) return data.chunks;
      }
    } catch (e) {
      // no manifest.json — fall through to the naming-convention guess below
    }
    const names = [];
    for (let i = 1; i <= FALLBACK_CHUNK_COUNT; i++) {
      names.push(`chunk-${String(i).padStart(4, '0')}.txt`);
    }
    return names;
  })();

  return chunkListPromise;
}

// Returns a blob of real plain text, ready to be tokenized into training
// sentences exactly the way the built-in BACKGROUND_CORPUS text is used
// elsewhere (see embeddings.js / rnn.js).
async function fetchStaticCorpusText(opts = {}) {
  const maxChars = opts.maxChars || 400000;

  const chunks = await getChunkList();
  if (!chunks.length) throw new Error('no static corpus chunks configured');

  // Try random chunks until one actually loads (in case the fallback guess
  // includes a filename that doesn't exist).
  const pool = [...chunks];
  let text = '';
  while (pool.length && !text) {
    const i = Math.floor(Math.random() * pool.length);
    const name = pool.splice(i, 1)[0];
    try {
      const res = await fetch(`${STATIC_CORPUS_DIR}/${name}`);
      if (!res.ok) continue;
      const full = await res.text();
      if (!full.trim()) continue;
      // Random slice so repeated runs against the same chunk still vary,
      // and so we never have to hold the whole ~95MB chunk in memory for
      // longer than the one fetch.
      if (full.length > maxChars) {
        const start = Math.floor(Math.random() * Math.max(1, full.length - maxChars));
        text = full.slice(start, start + maxChars);
      } else {
        text = full;
      }
    } catch (e) {
      continue; // try another chunk
    }
  }

  if (!text) throw new Error('could not load any static corpus chunk');
  return text;
}

if (typeof module !== 'undefined') {
  module.exports = { fetchStaticCorpusText };
}
