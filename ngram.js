// ngram.js
// A small, dependency-free n-gram language modeling engine.
// Supports unigram, bigram, trigram and 4-gram models with simple
// stupid-backoff smoothing, trained live in the browser on whatever
// text the user supplies (blended with a background corpus).

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

  // Stupid backoff: try highest order context first; if unseen, fall back
  // to a lower order, discounting scores by `alpha` each step down.
  distribution(contextTokens) {
    const alpha = 0.4;
    let order = this.n;
    let ctx = contextTokens.slice(Math.max(0, contextTokens.length - (order - 1)));
    let scale = 1.0;

    while (order >= 1) {
      const key = this._key(ctx);
      const m = this.counts[order].get(key);
      if (m && m.size > 0) {
        // Exclude sentence-boundary tokens from displayed predictions
        const entries = [...m.entries()].filter(([w]) => w !== '<s>' && w !== '</s>');
        if (entries.length > 0) {
          const total = entries.reduce((s, [, c]) => s + c, 0);
          return entries.map(([w, c]) => [w, (c / total) * scale]);
        }
      }
      order -= 1;
      ctx = ctx.slice(1);
      scale *= alpha;
    }
    // total fallback: uniform over vocab
    const words = [...this.vocab];
    return words.map(w => [w, 1 / words.length]);
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
