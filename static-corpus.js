// static-corpus.js
// Loads a real, large text corpus that lives as many small chunk files
// committed directly into this repo under static-corpus/ (see
// fetch-static-corpus.js for how they were generated) — no Git LFS, no
// third-party API at runtime. GitHub Pages serves these as plain static
// files, so a same-origin fetch() is as reliable as loading style.css.
//
// We don't want to download the whole 200+MB corpus on every run — we only
// need a few hundred KB of real prose per experiment — so this fetches the
// manifest once, then grabs a small random subset of chunk files per call.

const STATIC_CORPUS_DIR = 'static-corpus';
const STATIC_CORPUS_MANIFEST = `${STATIC_CORPUS_DIR}/manifest.json`;

let manifestPromise = null;
function getManifest() {
  if (!manifestPromise) {
    manifestPromise = fetch(STATIC_CORPUS_MANIFEST).then((r) => {
      if (!r.ok) throw new Error(`manifest fetch failed: ${r.status}`);
      return r.json();
    });
  }
  return manifestPromise;
}

// Returns a blob of real plain text, ready to be tokenized into training
// sentences exactly the way BACKGROUND_CORPUS / fetchInfiniGramCorpusText's
// output is used elsewhere (see embeddings.js / rnn.js / app.js).
async function fetchStaticCorpusText(opts = {}) {
  const numChunks = opts.numChunks || 3;
  const maxChars = opts.maxChars || 400000;

  const manifest = await getManifest();
  const chunks = manifest.chunks || [];
  if (!chunks.length) throw new Error('static corpus manifest is empty');

  // Sample without replacement so a single call never fetches the same
  // chunk twice, but different calls (different runs) land on different
  // chunks over time.
  const pool = [...chunks];
  const picked = [];
  for (let i = 0; i < Math.min(numChunks, pool.length); i++) {
    const j = Math.floor(Math.random() * pool.length);
    picked.push(pool.splice(j, 1)[0]);
  }

  const texts = await Promise.all(
    picked.map(async (name) => {
      const res = await fetch(`${STATIC_CORPUS_DIR}/${name}`);
      if (!res.ok) throw new Error(`chunk fetch failed: ${name} (${res.status})`);
      return res.text();
    })
  );

  let text = texts.join('\n\n');
  if (!text.trim()) throw new Error('static corpus sample came back empty');

  // Take a random slice rather than always the start of the concatenated
  // text, so repeated runs against the same chunk(s) still see different
  // material.
  if (text.length > maxChars) {
    const start = Math.floor(Math.random() * Math.max(1, text.length - maxChars));
    text = text.slice(start, start + maxChars);
  }
  return text;
}

if (typeof module !== 'undefined') {
  module.exports = { fetchStaticCorpusText };
}
