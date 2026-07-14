// ngram.js
// A small, dependency-free n-gram language modeling engine.
// Supports unigram, bigram, trigram and 4-gram models with
// interpolated Kneser-Ney smoothing — the technique that actually made
// trigram/4-gram models work well in the late 1990s-2000s (Chen &
// Goodman 1998 showed it beats every other classical smoothing method
// on held-out perplexity). Trained live in the browser on whatever text
// the user supplies, blended with a background corpus for coverage.

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[’']/g, "'")
    .match(/[a-z']+|[.!?,;]/g) || [];
}

// Wrap a token stream with sentence-boundary markers so contexts don't
// leak across sentences.
function toSentences(tokens) {
  const sentences = [];
  let cur = [];
  for (const t of tokens) {
    if (/^[.!?]$/.test(t)) {
      if (cur.length) sentences.push(cur);
      cur = [];
    } else if (/^[,;]$/.test(t)) {
      continue; // drop commas/semicolons, keep words flowing
    } else {
      cur.push(t);
    }
  }
  if (cur.length) sentences.push(cur);
  return sentences;
}

const DISCOUNT = 0.75; // standard fixed absolute-discount value (Chen & Goodman)

class NGramModel {
  constructor(n) {
    this.n = n; // 1 = unigram, 2 = bigram, 3 = trigram, 4 = four-gram
    // ngramCounts[order]: Map("w1 SEP ... wO" -> count), full o-gram (context+word)
    this.ngramCounts = {};
    // contextTotal[order]: Map(context(order-1) -> total count of ngrams starting with it)
    this.contextTotal = {};
    // distinctFollows[order]: Map(context(order-1) -> # distinct words following it) == N1+(context•)
    this.distinctFollows = {};
    // contCount[order]: Map(order-gram string -> # distinct single words that precede it as
    // an (order+1)-gram) — the "continuation count" used when this order is the lower order
    // being backed off to. contCount[1] = classic KN unigram continuation count.
    this.contCount = {};
    for (let k = 1; k <= n; k++) {
      this.ngramCounts[k] = new Map();
      this.contextTotal[k] = new Map();
      this.distinctFollows[k] = new Map();
      this.contCount[k] = new Map();
    }
    this.vocab = new Set();
  }

  _key(tokens) { return tokens.join(' \u0001 '); }

  train(sentencesOfTokens) {
    for (const sent of sentencesOfTokens) {
      const padded = Array(this.n - 1).fill('<s>').concat(sent).concat(['</s>']);
      for (const w of sent) this.vocab.add(w);
      for (let order = 1; order <= this.n; order++) {
        for (let i = order - 1; i < padded.length; i++) {
          const full = padded.slice(i - (order - 1), i + 1);
          const key = this._key(full);
          this.ngramCounts[order].set(key, (this.ngramCounts[order].get(key) || 0) + 1);
        }
      }
    }
    this._finalize();
  }

  // Derive context totals, distinct-follow counts, and continuation counts
  // from the raw n-gram tallies. Called once after all training text is in.
  _finalize() {
    for (let order = 1; order <= this.n; order++) {
      const ctxTotal = new Map();
      const distinctFollows = new Map();
      for (const [key, count] of this.ngramCounts[order]) {
        const parts = key.split(' \u0001 ');
        const ctxKey = this._key(parts.slice(0, -1));
        ctxTotal.set(ctxKey, (ctxTotal.get(ctxKey) || 0) + count);
        distinctFollows.set(ctxKey, (distinctFollows.get(ctxKey) || 0) + 1);
      }
      this.contextTotal[order] = ctxTotal;
      this.distinctFollows[order] = distinctFollows;
    }
    // Continuation counts: for order k, contCount[k] counts, for every k-gram
    // (as a *suffix*), how many distinct single words precede it as a (k+1)-gram.
    for (let order = 1; order < this.n; order++) {
      const cc = new Map();
      for (const key of this.ngramCounts[order + 1].keys()) {
        const parts = key.split(' \u0001 ');
        const suffix = this._key(parts.slice(1)); // drop the leftmost word
        cc.set(suffix, (cc.get(suffix) || 0) + 1);
      }
      this.contCount[order] = cc;
    }
    this._contTotalUnigram = this.contCount[1] ?
      [...this.contCount[1].values()].reduce((a, b) => a + b, 0) : 0;
  }

  // P_KN for a single word given a context, at a specific order, using
  // interpolated (not backoff) Kneser-Ney: every order contributes,
  // weighted by how well-attested that context is, so a well-attested
  // trigram context can dominate while a one-off trigram barely nudges
  // the bigram estimate.
  _pKN(order, ctxTokens, word) {
    if (order <= 1) {
      // Base case: continuation probability, not raw frequency — a word
      // that appears after many *different* words ranks higher than a
      // word that appears often but always after the same neighbor.
      const total = this._contTotalUnigram || 0;
      if (total === 0) return 1 / (this.vocab.size || 1);
      const cc = this.contCount[1] ? (this.contCount[1].get(word) || 0) : 0;
      // tiny additive smoothing so unseen vocab words aren't exactly zero
      return (cc + 0.001) / (total + 0.001 * (this.vocab.size || 1));
    }
    const ctx = ctxTokens.slice(ctxTokens.length - (order - 1));
    const ctxKey = this._key(ctx);
    const total = this.contextTotal[order].get(ctxKey) || 0;
    const lowerP = this._pKN(order - 1, ctxTokens, word);
    if (total === 0) return lowerP; // never saw this context; defer entirely to lower order
    const fullKey = this._key(ctx.concat([word]));
    const c = this.ngramCounts[order].get(fullKey) || 0;
    const discounted = Math.max(c - DISCOUNT, 0) / total;
    const nFollows = this.distinctFollows[order].get(ctxKey) || 0;
    const lambda = (DISCOUNT * nFollows) / total; // weight handed back to the lower order
    return discounted + lambda * lowerP;
  }

  // Full distribution over the vocabulary for the given left context.
  distribution(contextTokens) {
    const order = Math.min(this.n, contextTokens.length + 1);
    const entries = [...this.vocab].map(w => [w, this._pKN(order, contextTokens, w)]);
    const total = entries.reduce((s, [, p]) => s + Math.max(p, 0), 0) || 1;
    return entries.map(([w, p]) => [w, Math.max(p, 0) / total]);
  }

  // Build a cached cumulative distribution once, so drawing many samples
  // for the same context is O(1) per sample instead of re-running the KN
  // recursion over the whole vocab on every single draw.
  cdf(contextTokens) {
    const dist = this.distribution(contextTokens).sort((a, b) => b[1] - a[1]);
    const cum = [];
    let running = 0;
    for (const [w, p] of dist) { running += p; cum.push([w, running]); }
    return cum;
  }

  sampleFromCdf(cum, rng = Math.random) {
    const r = rng() * cum[cum.length - 1][1];
    for (const [w, c] of cum) if (r <= c) return w; // cum is sorted desc by prob, so common words hit fast
    return cum[cum.length - 1][0];
  }

  topK(contextTokens, k = 8) {
    return this.distribution(contextTokens).sort((a, b) => b[1] - a[1]).slice(0, k);
  }
}

function buildModel(order, storyText, backgroundText, backgroundWeight) {
  const model = new NGramModel(order);
  const storySentences = toSentences(tokenize(storyText));
  const allSentences = storySentences.slice();
  if (backgroundWeight > 0) {
    const bgSentences = toSentences(tokenize(backgroundText));
    const reps = Math.max(1, Math.round(backgroundWeight));
    for (let i = 0; i < reps; i++) allSentences.push(...bgSentences);
  }
  model.train(allSentences);
  return model;
}

// Run `runs` samples of the model given left context and tally frequencies.
// The distribution is computed ONCE, then draws come from a cached CDF —
// faster, and what makes Kneser-Ney (pricier per-call than plain linear
// interpolation) practical at 1000+ samples.
function runSamplingExperiment(model, contextTokens, runs = 1000) {
  const cum = model.cdf(contextTokens);
  const counts = new Map();
  for (let i = 0; i < runs; i++) {
    const w = model.sampleFromCdf(cum);
    counts.set(w, (counts.get(w) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

if (typeof module !== 'undefined') {
  module.exports = { tokenize, toSentences, NGramModel, buildModel, runSamplingExperiment };
}