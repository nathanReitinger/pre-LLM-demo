// static-ngram-model.js
// Loads and queries the PRETRAINED n-gram model produced by train.py
// (ngram-model/manifest.json + vocab.json + o{2,3,4}-<shard>.json) — a
// model built once, offline, over the entire (or --max-sentences-capped)
// static corpus, instead of the tiny live-in-browser model ngram.js trains
// from whatever text happens to be loaded for one run.
//
// This is what makes the n-gram rows good BEFORE the user has written much
// of a story: ngram.js's live model only ever has the story (repeated a
// few times) plus one ~750KB random slice of the corpus to learn from, so
// most contexts are unseen and it backs off to near-uniform guessing. This
// module instead queries real counts gathered over the whole trained
// corpus, fetched lazily as small per-context shard files, so there's no
// multi-second in-browser training step and no dependence on which slice
// of the corpus got randomly sampled for this particular run.
//
// Data format (must match train.py exactly):
//   ngram-model/manifest.json — { orders, vocabSize, numShards, discounts,
//                                 topK, minCount, ... }
//   ngram-model/vocab.json    — { words[], unigramCount[], contUnigram[],
//                                 contUnigramTotal, totalTokens }
//   ngram-model/o{order}-{shardIdx}.json — array of
//     [ctxKey, total, distinct, [[wordId, count], ...]]  (top-K continuations only)
//
// shardFor() below MUST match shard_for() in train.py bit-for-bit (same
// 32-bit wraparound hash), or a context that was written into shard N by
// the trainer would be looked for in the wrong shard file at query time
// and silently look unseen.

const STATIC_MODEL_DIR = 'ngram-model';
const STATIC_MODEL_MANIFEST = `${STATIC_MODEL_DIR}/manifest.json`;
const STATIC_MODEL_VOCAB = `${STATIC_MODEL_DIR}/vocab.json`;
const FALLBACK_DISCOUNT = 0.75;
const UNIGRAM_SMOOTH = 0.001; // must mirror ngram.js's own additive smoothing constant

// 32-bit wraparound multiply-and-add hash, identical to shard_for() in
// train.py: h = 7; for each context id: h = (h*131 + id) mod 2^32.
// `>>> 0` reproduces Python's `& 0xFFFFFFFF` for any integer that fits
// exactly in a JS double (true here: h stays < 2^32, id < 65536, so
// h*131+id stays well under 2^53 and is never rounded).
function shardFor(contextIds, numShards) {
  let h = 7;
  for (const cid of contextIds) {
    h = (h * 131 + cid) >>> 0;
  }
  return h % numShards;
}

let metaPromise = null;

// Fetches manifest.json + vocab.json once and caches the result (including
// failures — if the static model isn't present, every caller finds out
// via the same rejected promise instead of re-fetching a 404 repeatedly).
function loadStaticModelMeta() {
  if (metaPromise) return metaPromise;
  metaPromise = (async () => {
    const [manifestRes, vocabRes] = await Promise.all([
      fetch(STATIC_MODEL_MANIFEST),
      fetch(STATIC_MODEL_VOCAB),
    ]);
    if (!manifestRes.ok || !vocabRes.ok) {
      throw new Error('static n-gram model not found (ngram-model/manifest.json or vocab.json missing — run train.py and commit ngram-model/ to enable it)');
    }
    const manifest = await manifestRes.json();
    const vocab = await vocabRes.json();
    if (!Array.isArray(vocab.words) || !vocab.words.length) {
      throw new Error('ngram-model/vocab.json is malformed (no words[])');
    }
    const wordToId = new Map(vocab.words.map((w, i) => [w, i]));
    return { manifest, vocab, wordToId };
  })();
  return metaPromise;
}

// True if the static model loads successfully. Callers use this to decide
// whether to use the pretrained model or fall back to ngram.js's
// live-in-browser training.
async function staticModelAvailable() {
  try {
    await loadStaticModelMeta();
    return true;
  } catch (e) {
    return false;
  }
}

const shardCache = new Map(); // "order-shardIdx" -> Promise<Map<ctxKey, entry>>

function loadShard(order, shardIdx) {
  const key = `${order}-${shardIdx}`;
  if (shardCache.has(key)) return shardCache.get(key);
  const p = (async () => {
    const res = await fetch(`${STATIC_MODEL_DIR}/o${order}-${shardIdx}.json`);
    if (!res.ok) throw new Error(`missing shard o${order}-${shardIdx}.json`);
    const rows = await res.json();
    const map = new Map();
    for (const row of rows) {
      const [ctxKey, total, distinct, words] = row;
      map.set(ctxKey, { total, distinct, wordMap: new Map(words) });
    }
    return map;
  })();
  shardCache.set(key, p);
  return p;
}

// Looks up the stored (total, distinct, top-K word->count map) entry for
// one exact context at one order, fetching (and caching) only the single
// shard file that context hashes into. Returns null if the context was
// never seen at this order, or was seen but pruned below --min-count-*.
async function getContextEntry(order, meta, ctxIds) {
  const numShards = meta.manifest.numShards[String(order)];
  if (!numShards) return null;
  const shardIdx = shardFor(ctxIds, numShards);
  const shard = await loadShard(order, shardIdx);
  return shard.get(ctxIds.join(',')) || null;
}

class StaticNgramModel {
  constructor(order, meta) {
    this.order = order; // 2, 3, or 4 (unigram has no shards — see StaticUnigramModel below)
    this.meta = meta;
  }

  _pUnigram(wordId) {
    const { vocab } = this.meta;
    const V = vocab.words.length;
    const contTotal = vocab.contUnigramTotal || 0;
    if (contTotal > 0) {
      const cc = vocab.contUnigram[wordId] || 0;
      return (cc + UNIGRAM_SMOOTH) / (contTotal + UNIGRAM_SMOOTH * V);
    }
    const total = vocab.totalTokens || 0;
    const c = vocab.unigramCount[wordId] || 0;
    return (c + UNIGRAM_SMOOTH) / (total + UNIGRAM_SMOOTH * V);
  }

  // Classic (non-generalized) interpolated Kneser-Ney: raw joint
  // counts/discounts at every stored order, true continuation counts only
  // at the unigram base case — this is what the stored shards actually
  // support, since train.py only wrote top-K RAW context/word counts per
  // order, not continuation counts for every intermediate order the way
  // ngram.js's live model derives them. `entries` maps order -> the one
  // context entry needed at that order (or null), prefetched once by
  // distribution() below so this stays fully synchronous.
  _pAt(order, entries, wordId) {
    if (order <= 1) return this._pUnigram(wordId);
    const entry = entries[order];
    const lowerP = this._pAt(order - 1, entries, wordId);
    if (!entry || entry.total === 0) return lowerP;
    const D = this.meta.manifest.discounts[String(order)] ?? FALLBACK_DISCOUNT;
    const c = entry.wordMap.get(wordId) || 0;
    const discounted = Math.max(c - D, 0) / entry.total;
    const lambda = (D * entry.distinct) / entry.total;
    return discounted + lambda * lowerP;
  }

  // Full probability distribution over the vocabulary given a left
  // context, normalized to sum to 1 — same contract as NGramModel's
  // distribution() in ngram.js so the two are interchangeable callers.
  async distribution(ctxTokens) {
    const { vocab, wordToId } = this.meta;
    const order = Math.min(this.order, ctxTokens.length + 1);
    const ctxIds = ctxTokens.map(t => (wordToId.has(t) ? wordToId.get(t) : null));

    // Prefetch exactly the shard entries this query needs — at most one
    // fetch per order (2..order), each for the single shard the exact
    // context hashes into. Missing/never-seen contexts resolve to null and
    // just fall through to backoff below.
    const entries = {};
    for (let o = order; o >= 2; o--) {
      const need = o - 1;
      const tail = ctxIds.slice(ctxIds.length - need);
      if (tail.length < need || tail.includes(null)) { entries[o] = null; continue; }
      entries[o] = await getContextEntry(o, this.meta, tail);
    }

    const V = vocab.words.length;
    const scored = new Array(V);
    for (let wid = 0; wid < V; wid++) {
      scored[wid] = [vocab.words[wid], Math.max(this._pAt(order, entries, wid), 0)];
    }
    const total = scored.reduce((s, [, p]) => s + p, 0) || 1;
    for (let i = 0; i < scored.length; i++) scored[i][1] /= total;
    return scored;
  }

  async cdf(ctxTokens) {
    const dist = (await this.distribution(ctxTokens)).sort((a, b) => b[1] - a[1]);
    const cum = [];
    let running = 0;
    for (const [w, p] of dist) { running += p; cum.push([w, running]); }
    return cum;
  }

  async topK(ctxTokens, k = 8) {
    const dist = await this.distribution(ctxTokens);
    return dist.sort((a, b) => b[1] - a[1]).slice(0, k);
  }
}

function sampleFromCdf(cum, rng = Math.random) {
  if (!cum.length) return null;
  const r = rng() * cum[cum.length - 1][1];
  for (const [w, c] of cum) if (r <= c) return w; // cum sorted desc by prob, common words hit fast
  return cum[cum.length - 1][0];
}

// Same shape as ngram.js's runSamplingExperiment (word -> sample count
// pairs, descending), but async since the underlying distribution requires
// a network fetch for the relevant shard(s) the first time it's used.
async function staticRunSamplingExperiment(model, ctxTokens, runs = 1000) {
  const cum = await model.cdf(ctxTokens);
  const counts = new Map();
  for (let i = 0; i < runs; i++) {
    const w = sampleFromCdf(cum);
    if (w === null) break;
    counts.set(w, (counts.get(w) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

// order: 1 (unigram) through 4. Unigram has no shards to fetch — it's
// served straight out of vocab.json — so it uses the same class with a
// trivial order-1 path (StaticNgramModel._pAt already handles order<=1).
async function loadStaticNgramModel(order) {
  const meta = await loadStaticModelMeta();
  const maxOrder = Math.max(...meta.manifest.orders, 1);
  return new StaticNgramModel(Math.min(order, maxOrder), meta);
}

if (typeof module !== 'undefined') {
  module.exports = {
    shardFor,
    loadStaticModelMeta,
    staticModelAvailable,
    loadStaticNgramModel,
    staticRunSamplingExperiment,
    StaticNgramModel,
    sampleFromCdf,
  };
}
