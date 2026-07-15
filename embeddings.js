// embeddings.js
// Distributed word representations (the 2003-2013 neural-LM / word2vec era
// idea) computed live in the browser: build a word x word co-occurrence
// matrix, convert it to PPMI (Positive Pointwise Mutual Information — the
// count-based cousin of what word2vec learns implicitly, per Levy & Goldberg
// 2014's "word2vec as implicit matrix factorization"), then compress it to a
// small dense vector per word with truncated SVD via randomized power
// iteration. No external ML library needed — just linear algebra.
//
// Unlike the n-gram models (which only ever look left), this model fills
// the blank using vectors built from BOTH left and right neighbors of the
// blank, averaged into a single "context vector," then finds the vocabulary
// words whose own vector is most similar (cosine) to that context vector.
// That's the real conceptual leap this era introduced: words as points in a
// continuous space where "nearby in meaning" is a geometric fact, not a
// frequency table lookup.

const STOPWORDS = new Set([
  'the','a','an','and','or','but','of','to','in','on','at','for','with',
  'as','is','was','were','be','been','being','it','its','this','that',
  'these','those','i','you','he','she','they','we','who','which','what',
  'when','where','why','how','not','no','so','if','than','then','there',
  'here','their','his','her','my','your','our','have','has','had','do',
  'does','did','will','would','could','should','can','may','might','from',
  'by','up','out','about','into','over','after','before','between'
]);

function buildEmbeddings(storyText, backgroundText, opts = {}) {
  const dim = opts.dim || 24;
  const window = opts.window || 4;
  const maxVocab = opts.maxVocab || 900;

  const storySentences = toSentences(tokenize(storyText));
  const bgSentences = toSentences(tokenize(backgroundText || ''));
  const allSentences = storySentences.concat(bgSentences);

  // Frequency-limit the vocabulary so the co-occurrence matrix stays small
  // enough to factor in well under a second.
  const freq = new Map();
  for (const sent of allSentences) for (const w of sent) freq.set(w, (freq.get(w) || 0) + 1);
  const storyWords = new Set(storySentences.flat());
  // Always keep every word that appears in the user's own story, even if rare,
  // then fill the remaining budget with the most frequent background words.
  const ranked = [...freq.entries()].sort((a, b) => b[1] - a[1]).map(([w]) => w);
  const vocabSet = new Set(storyWords);
  for (const w of ranked) {
    if (vocabSet.size >= maxVocab) break;
    vocabSet.add(w);
  }
  const vocab = [...vocabSet];
  const idx = new Map(vocab.map((w, i) => [w, i]));
  const V = vocab.length;

  // Sparse co-occurrence counts within a symmetric window, distance-weighted
  // (closer words count more) — the standard GloVe-style windowing.
  const cooc = new Map(); // "i,j" -> weight
  const totalPerWord = new Float64Array(V);
  const bump = (i, j, w) => {
    const key = i < j ? `${i},${j}` : `${j},${i}`;
    cooc.set(key, (cooc.get(key) || 0) + w);
  };
  for (const sent of allSentences) {
    const ids = sent.map(w => idx.get(w)).filter(v => v !== undefined);
    for (let i = 0; i < ids.length; i++) {
      for (let d = 1; d <= window; d++) {
        if (i + d >= ids.length) break;
        const w = 1 / d;
        bump(ids[i], ids[i + d], w);
        totalPerWord[ids[i]] += w;
        totalPerWord[ids[i + d]] += w;
      }
    }
  }
  const totalMass = [...cooc.values()].reduce((a, b) => a + b, 0) * 2 || 1;

  // PPMI: log( P(i,j) / (P(i)P(j)) ), floored at 0.
  const entries = []; // {i, j, val}
  for (const [key, w] of cooc) {
    const [i, j] = key.split(',').map(Number);
    const pij = w / totalMass;
    const pi = totalPerWord[i] / totalMass;
    const pj = totalPerWord[j] / totalMass;
    const pmi = Math.log((pij + 1e-12) / (pi * pj + 1e-12));
    const ppmi = Math.max(pmi, 0);
    if (ppmi > 0) entries.push([i, j, ppmi]);
  }

  // Dense matrix (V is capped at maxVocab so this stays small, e.g. <=900^2).
  const M = new Float64Array(V * V);
  for (const [i, j, v] of entries) { M[i * V + j] = v; M[j * V + i] = v; }

  // Truncated SVD of the symmetric PPMI matrix via randomized simultaneous
  // power iteration (subspace iteration) — fast, dependency-free, and exact
  // enough for a demo: repeatedly multiply a random V x dim block by M and
  // re-orthonormalize (Gram-Schmidt). With enough iterations this basis
  // converges to (approximately) the top-|dim| eigenvectors of M, ordered by
  // eigenvalue magnitude.
  let vectors = randomMatrix(V, dim);
  orthonormalize(vectors, V, dim);
  const iters = 10;
  let Mv = null;
  for (let it = 0; it < iters; it++) {
    Mv = matMulSymmetric(M, vectors, V, dim);
    vectors = Mv;
    orthonormalize(vectors, V, dim);
  }

  // Proper truncated-SVD scaling: embedding = eigenvectors * sqrt(eigenvalue),
  // the standard LSA/word2vec-as-matrix-factorization construction (Levy &
  // Goldberg 2014), rather than an ad hoc per-word magnitude fudge. Each
  // orthonormal column's eigenvalue is estimated via its Rayleigh quotient
  // v^T M v (valid since ||v|| = 1 after orthonormalization); dimensions that
  // capture more corpus variance then contribute proportionally more to cosine
  // similarity, and near-zero/negative directions (numerical noise, since PPMI
  // matrices aren't guaranteed positive-semidefinite) are scaled to ~0 instead
  // of counting equally alongside the real signal.
  Mv = matMulSymmetric(M, vectors, V, dim);
  const eigVals = new Float64Array(dim);
  for (let d = 0; d < dim; d++) {
    let num = 0;
    for (let i = 0; i < V; i++) num += vectors[i * dim + d] * Mv[i * dim + d];
    eigVals[d] = num;
  }
  for (let d = 0; d < dim; d++) {
    const scale = Math.sqrt(Math.max(eigVals[d], 0));
    for (let i = 0; i < V; i++) vectors[i * dim + d] *= scale;
  }

  return { vocab, idx, vectors, dim, V };
}

function randomMatrix(rows, cols) {
  const m = new Float64Array(rows * cols);
  for (let i = 0; i < m.length; i++) m[i] = Math.random() * 2 - 1;
  return m;
}

// Gram-Schmidt orthonormalization of the `cols` column vectors of an
// rows x cols matrix stored row-major.
function orthonormalize(m, rows, cols) {
  const col = (c) => { const v = new Float64Array(rows); for (let r = 0; r < rows; r++) v[r] = m[r * cols + c]; return v; };
  const setCol = (c, v) => { for (let r = 0; r < rows; r++) m[r * cols + c] = v[r]; };
  const dot = (a, b) => { let s = 0; for (let r = 0; r < rows; r++) s += a[r] * b[r]; return s; };
  const cols_ = [];
  for (let c = 0; c < cols; c++) {
    let v = col(c);
    for (const u of cols_) {
      const proj = dot(v, u);
      for (let r = 0; r < rows; r++) v[r] -= proj * u[r];
    }
    const norm = Math.sqrt(dot(v, v)) || 1e-9;
    for (let r = 0; r < rows; r++) v[r] /= norm;
    cols_.push(v);
    setCol(c, v);
  }
}

// (V x V) * (V x dim) -> (V x dim), M symmetric and dense.
function matMulSymmetric(M, X, V, dim) {
  const out = new Float64Array(V * dim);
  for (let i = 0; i < V; i++) {
    const rowOff = i * V;
    for (let k = 0; k < V; k++) {
      const mv = M[rowOff + k];
      if (mv === 0) continue;
      const xOff = k * dim;
      const oOff = i * dim;
      for (let d = 0; d < dim; d++) out[oOff + d] += mv * X[xOff + d];
    }
  }
  return out;
}

function cosine(vectors, dim, i, vec) {
  let dot = 0, na = 0, nb = 0;
  const off = i * dim;
  for (let d = 0; d < dim; d++) {
    dot += vectors[off + d] * vec[d];
    na += vectors[off + d] * vectors[off + d];
    nb += vec[d] * vec[d];
  }
  return dot / ((Math.sqrt(na) * Math.sqrt(nb)) || 1e-9);
}

// Predict the blank by averaging the vectors of nearby context words on
// BOTH sides (this is the point of the demo: no more left-only window),
// then ranking the whole vocabulary by cosine similarity to that average.
function embeddingPredict(emb, leftTokens, rightTokens, k = 8) {
  const { vocab, idx, vectors, dim } = emb;
  const ctxWords = leftTokens.slice(-6).concat(rightTokens.slice(0, 6))
    .filter(w => idx.has(w) && !STOPWORDS.has(w));
  if (ctxWords.length === 0) {
    // fall back to any content words available at all
    leftTokens.concat(rightTokens).forEach(w => { if (idx.has(w)) ctxWords.push(w); });
  }
  const avg = new Float64Array(dim);
  let n = 0;
  for (const w of ctxWords) {
    const off = idx.get(w) * dim;
    for (let d = 0; d < dim; d++) avg[d] += vectors[off + d];
    n++;
  }
  if (n === 0) return vocab.slice(0, k).map(w => [w, 1 / k]);
  for (let d = 0; d < dim; d++) avg[d] /= n;

  const ctxSet = new Set(ctxWords);
  const sims = vocab.map((w, i) => [w, cosine(vectors, dim, i, avg)])
    .filter(([w]) => !ctxSet.has(w) && !STOPWORDS.has(w) && w !== '<s>' && w !== '</s>');
  sims.sort((a, b) => b[1] - a[1]);
  const top = sims.slice(0, k);
  // Turn similarities (roughly -1..1) into a display-friendly pseudo-distribution
  // via softmax, purely for the percentage column — cosine similarity itself
  // isn't a probability.
  const maxS = top.length ? top[0][1] : 1;
  const exps = top.map(([w, s]) => [w, Math.exp((s - maxS) * 6)]);
  const total = exps.reduce((s, [, e]) => s + e, 0) || 1;
  return exps.map(([w, e]) => [w, e / total]);
}

if (typeof module !== 'undefined') {
  module.exports = { buildEmbeddings, embeddingPredict };
}