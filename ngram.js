// ngram.js
// A small, dependency-free n-gram language modeling engine.
// Supports unigram, bigram, trigram and 4-gram models with linear
// (Jelinek-Mercer) interpolation smoothing — the technique real n-gram
// LMs of the 1980s-1990s used — trained live in the browser on whatever
// text the user supplies, blended with a large background corpus so
// bigram/trigram contexts actually have enough data to say something.

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

class NGramModel {
  constructor(n) {
    this.n = n; // 1 = unigram, 2 = bigram, 3 = trigram, 4 = four-gram
    // counts[order] maps "context key" -> Map(word -> count)
    this.counts = {};
    for (let k = 1; k <= n; k++) this.counts[k] = new Map();
    this.vocab = new Set();
  }

  _key(ctxTokens) {
    return ctxTokens.join(' \u0001 ');
  }

  train(sentencesOfTokens) {
    for (const sent of sentencesOfTokens) {
      const padded = Array(this.n - 1).fill('<s>').concat(sent).concat(['</s>']);
      for (const w of sent) this.vocab.add(w);
      for (let order = 1; order <= this.n; order++) {
        for (let i = order - 1; i < padded.length; i++) {
          const ctx = padded.slice(i - (order - 1), i);
          const word = padded[i];
          const key = this._key(ctx);
          if (!this.counts[order].has(key)) this.counts[order].set(key, new Map());
          const m = this.counts[order].get(key);
          m.set(word, (m.get(word) || 0) + 1);
        }
      }
    }
  }

  // Linear (Jelinek-Mercer style) interpolation across all orders.
  //
  // This is the historically accurate smoothing technique for this era of
  // language modeling (Jelinek & Mercer, ~1980; standard in n-gram LMs
  // through the 1990s) — as opposed to "stupid backoff," which is a
  // 2007 Google-scale shortcut that only makes sense when you have
  // billions of words and can't afford anything smarter. Stupid backoff
  // also has a specific failure mode that makes small-corpus bigram/
  // trigram demos misleading: the instant the exact higher-order context
  // is unseen, it discards that evidence completely and drops straight to
  // a lower order — so on a small corpus, "bigram" and "trigram" almost
  // always just collapse into "unigram wearing a costume."
  //
  // Interpolation instead *blends* every order's evidence at once, weighted
  // by how much data actually backs each order up:
  //
  //   P_k(w | ctx) = lambda(ctx) * P_ML_k(w | ctx) + (1 - lambda(ctx)) * P_{k-1}(w | ctx[1:])
  //
  // where lambda(ctx) = count(ctx) / (count(ctx) + K) grows toward 1 as
  // the model has seen that specific context more often, and toward 0
  // (deferring almost entirely to the lower order) when it's rare or
  // unseen. This means a well-attested trigram context can dominate,
  // while a one-off trigram context barely perturbs the bigram estimate —
  // which is exactly the qualitative difference that made trigram models
  // a genuine improvement over bigram models in the literature, and
  // exactly what stupid backoff throws away on small data.
  distribution(contextTokens) {
    const V = this.vocab.size || 1;
    const unigramMap = this.counts[1].get('') || new Map();
    const totalUnigram = [...unigramMap.values()].reduce((a, b) => a + b, 0);

    // Base case: Laplace(add-one)-smoothed unigram distribution over the
    // full vocabulary, so every known word has *some* nonzero probability.
    let probs = new Map();
    const unigramDenom = totalUnigram + V;
    for (const w of this.vocab) {
      probs.set(w, ((unigramMap.get(w) || 0) + 1) / unigramDenom);
    }

    // K controls how many times a context needs to be seen before the
    // model starts trusting it over the lower order — small enough that a
    // handful of real occurrences (which a corpus this size now provides)
    // meaningfully shifts the prediction, large enough that a single
    // freak occurrence doesn't dominate.
    const K = 2;

    for (let order = 2; order <= this.n; order++) {
      if (contextTokens.length < order - 1) break; // not enough left context yet
      const ctx = contextTokens.slice(contextTokens.length - (order - 1));
      const key = this._key(ctx);
      const m = this.counts[order].get(key);
      const ctxTotal = m ? [...m.values()].reduce((a, b) => a + b, 0) : 0;
      if (ctxTotal === 0) continue; // no evidence at this order; keep lower-order estimate

      const lambda = ctxTotal / (ctxTotal + K);
      const next = new Map();
      for (const w of this.vocab) {
        const pml = (m.get(w) || 0) / ctxTotal;
        next.set(w, lambda * pml + (1 - lambda) * probs.get(w));
      }
      probs = next;
    }

    // Exclude sentence-boundary markers from what's shown/sampled, then
    // renormalize so the displayed percentages sum to 100%.
    const entries = [...probs.entries()].filter(([w]) => w !== '<s>' && w !== '</s>');
    const total = entries.reduce((s, [, p]) => s + p, 0) || 1;
    return entries.map(([w, p]) => [w, p / total]);
  }

  // Sample one word given left context, using the (possibly backed-off) distribution.
  sample(contextTokens, rng = Math.random) {
    const dist = this.distribution(contextTokens);
    const total = dist.reduce((s, [, p]) => s + p, 0);
    let r = rng() * total;
    for (const [w, p] of dist) {
      r -= p;
      if (r <= 0) return w;
    }
    return dist[dist.length - 1][0];
  }

  topK(contextTokens, k = 8) {
    const dist = this.distribution(contextTokens);
    const total = dist.reduce((s, [, p]) => s + p, 0);
    return dist
      .map(([w, p]) => [w, p / total])
      .sort((a, b) => b[1] - a[1])
      .slice(0, k);
  }
}

function buildModel(order, storyText, backgroundText, backgroundWeight) {
  const model = new NGramModel(order);
  const storySentences = toSentences(tokenize(storyText));
  model.train(storySentences);
  if (backgroundWeight > 0) {
    const bgSentences = toSentences(tokenize(backgroundText));
    // Repeat background training pass to control its relative influence
    // without having to touch raw counts directly.
    const reps = Math.max(1, Math.round(backgroundWeight));
    for (let i = 0; i < reps; i++) model.train(bgSentences);
  }
  return model;
}

// Run `runs` samples of the model given left context and tally frequencies.
function runSamplingExperiment(model, contextTokens, runs = 1000) {
  const counts = new Map();
  for (let i = 0; i < runs; i++) {
    const w = model.sample(contextTokens);
    counts.set(w, (counts.get(w) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

if (typeof module !== 'undefined') {
  module.exports = { tokenize, toSentences, NGramModel, buildModel, runSamplingExperiment };
}
