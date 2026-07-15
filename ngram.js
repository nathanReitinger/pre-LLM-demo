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

const DEFAULT_DISCOUNT = 0.75; // fallback when an order has too little data to estimate its own

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
    // an (order+1)-gram) — the "continuation count" N1+(• order-gram), used whenever this
    // order is serving as a *lower* (backed-off-to) order rather than the top query order.
    // contCount[1] = classic KN unigram continuation count; contCount[2..n-1] generalize the
    // same idea to bigrams/trigrams so a trigram model backing off to bigram also gets the
    // "how many distinct contexts does this bigram continue" signal, not just raw bigram counts.
    this.contCount = {};
    // contextContTotal[order] / distinctFollowsCont[order]: the continuation-count analogues of
    // contextTotal / distinctFollows above, i.e. summed/counted over contCount instead of raw
    // counts. Needed so a backed-off-to order can normalize and interpolate correctly.
    this.contextContTotal = {};
    this.distinctFollowsCont = {};
    // discount[order]: per-order absolute discount, estimated from the corpus via the classic
    // Kneser-Ney formula D = n1 / (n1 + 2*n2), rather than a single hardcoded constant — a
    // sparse order (e.g. 4-gram on a short story) gets a smaller, safer discount than a
    // well-attested order.
    this.discount = {};
    for (let k = 1; k <= n; k++) {
      this.ngramCounts[k] = new Map();
      this.contextTotal[k] = new Map();
      this.distinctFollows[k] = new Map();
      this.contCount[k] = new Map();
      this.contextContTotal[k] = new Map();
      this.distinctFollowsCont[k] = new Map();
      this.discount[k] = DEFAULT_DISCOUNT;
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
    // (as a *suffix*), how many distinct single words precede it as a (k+1)-gram —
    // i.e. N1+(• k-gram). Computed for every order below the top (1..n-1) so that
    // ANY order, not just the unigram, can use true continuation probability when
    // it's serving as a backed-off-to lower order.
    for (let order = 1; order < this.n; order++) {
      const cc = new Map();
      for (const key of this.ngramCounts[order + 1].keys()) {
        const parts = key.split(' \u0001 ');
        const suffix = this._key(parts.slice(1)); // drop the leftmost word
        cc.set(suffix, (cc.get(suffix) || 0) + 1);
      }
      this.contCount[order] = cc;

      // Roll contCount up by context, exactly the way contextTotal/distinctFollows
      // roll up raw counts by context — this is what makes generalized KN work at
      // every order: contextContTotal[order] = N1+(• context •), the normalizer for
      // continuation-based probabilities at this order.
      const ctxContTotal = new Map();
      const followsCont = new Map();
      for (const [key, cnt] of cc) {
        const parts = key.split(' \u0001 ');
        const ctxKey = this._key(parts.slice(0, -1));
        ctxContTotal.set(ctxKey, (ctxContTotal.get(ctxKey) || 0) + cnt);
        followsCont.set(ctxKey, (followsCont.get(ctxKey) || 0) + 1);
      }
      this.contextContTotal[order] = ctxContTotal;
      this.distinctFollowsCont[order] = followsCont;
    }
    this._contTotalUnigram = this.contCount[1] ?
      [...this.contCount[1].values()].reduce((a, b) => a + b, 0) : 0;

    // Plain relative-frequency total, independent of continuation counts.
    // Continuation counts (above) only exist when this.n >= 2, because they're
    // derived from bigrams. A standalone unigram model (this.n === 1) has no
    // bigrams to derive them from, so it needs a real fallback — otherwise it
    // has nothing to rank words by and ends up guessing uniformly at random.
    this._unigramTotal = 0;
    for (const c of this.ngramCounts[1].values()) this._unigramTotal += c;

    // Data-driven absolute discount per order: the original Kneser-Ney estimate
    // D = n1 / (n1 + 2*n2), where n1/n2 are the number of order-grams seen exactly
    // once/twice. A sparse order (e.g. 4-gram on a short story, where almost every
    // count is 1) gets pulled toward a smaller, safer discount instead of always
    // subtracting a flat 0.75 regardless of how much data actually supports it.
    for (let order = 1; order <= this.n; order++) {
      let n1 = 0, n2 = 0;
      for (const c of this.ngramCounts[order].values()) {
        if (c === 1) n1++; else if (c === 2) n2++;
      }
      if (n1 + 2 * n2 > 0) {
        const d = n1 / (n1 + 2 * n2);
        // keep it in a sane range; a degenerate corpus can otherwise push D to 0 or >1
        this.discount[order] = Math.min(0.9, Math.max(0.1, d));
      } else {
        this.discount[order] = DEFAULT_DISCOUNT;
      }
    }
  }

  // P_KN for a single word given a context, at a specific order, using
  // *generalized* interpolated Kneser-Ney: every order contributes, weighted by
  // how well-attested that context is, so a well-attested trigram context can
  // dominate while a one-off trigram barely nudges the bigram estimate.
  //
  // `isTop` marks the order the caller actually asked for (e.g. "trigram given
  // this 2-word context") as opposed to an order reached purely by backing off.
  // Only the top order uses raw joint counts — every order it backs off to uses
  // *continuation* counts (N1+(• x)) instead of raw counts, which is the actual
  // definition of Kneser-Ney at every level, not just the final unigram fallback.
  // A word that follows many different contexts should rank higher there than a
  // word that occurs often but always glued to one specific neighbor, and that's
  // true whether the lower order is a unigram, bigram, or trigram.
  _pKN(order, ctxTokens, word, isTop = true) {
    if (order <= 1) {
      // Base case: continuation probability, not raw frequency. Needs bigrams to
      // derive from, so only available when this.n >= 2.
      const contTotal = this._contTotalUnigram || 0;
      if (contTotal > 0) {
        const cc = this.contCount[1] ? (this.contCount[1].get(word) || 0) : 0;
        // tiny additive smoothing so unseen vocab words aren't exactly zero
        return (cc + 0.001) / (contTotal + 0.001 * (this.vocab.size || 1));
      }
      // Standalone unigram model (this.n === 1): no bigrams exist to derive
      // continuation counts from, so fall back to plain relative frequency —
      // the actual definition of a unigram model — instead of guessing
      // uniformly at random over the vocabulary.
      const total = this._unigramTotal || 0;
      if (total === 0) return 1 / (this.vocab.size || 1);
      const c = this.ngramCounts[1].get(word) || 0;
      return (c + 0.001) / (total + 0.001 * (this.vocab.size || 1));
    }

    const ctx = ctxTokens.slice(ctxTokens.length - (order - 1));
    const ctxKey = this._key(ctx);
    const fullKey = this._key(ctx.concat([word]));
    const lowerP = this._pKN(order - 1, ctxTokens, word, false); // recursive = never top
    const D = this.discount[order] ?? DEFAULT_DISCOUNT;

    if (isTop) {
      // Top query order: use real joint counts, same as before.
      const total = this.contextTotal[order].get(ctxKey) || 0;
      if (total === 0) return lowerP; // never saw this context; defer entirely to lower order
      const c = this.ngramCounts[order].get(fullKey) || 0;
      const discounted = Math.max(c - D, 0) / total;
      const nFollows = this.distinctFollows[order].get(ctxKey) || 0;
      const lambda = (D * nFollows) / total;
      return discounted + lambda * lowerP;
    }

    // Backed-off-to order: use continuation counts instead of raw counts, both
    // in the numerator (N1+(• context word)) and the denominator (N1+(• context •)).
    const contTotal = this.contextContTotal[order].get(ctxKey) || 0;
    if (contTotal === 0) return lowerP; // this context never continues anything at this order
    const cc = this.contCount[order].get(fullKey) || 0;
    const discounted = Math.max(cc - D, 0) / contTotal;
    const nFollowsCont = this.distinctFollowsCont[order].get(ctxKey) || 0;
    const lambda = (D * nFollowsCont) / contTotal;
    return discounted + lambda * lowerP;
  }

  // Full distribution over the vocabulary for the given left context.
  distribution(contextTokens) {
    const order = Math.min(this.n, contextTokens.length + 1);
    const entries = [...this.vocab].map(w => [w, this._pKN(order, contextTokens, w, true)]);
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