// rnn.js
// A small word-level recurrent neural network language model, trained live
// in the browser with plain backprop-through-time (BPTT) — no external ML
// library, just arrays. This is the Mikolov (2010) RNNLM idea: instead of a
// fixed n-1 word window, a hidden state vector is updated one word at a time
// and carries (in principle) information from arbitrarily far back in the
// sentence. It still only looks left-to-right, same as the n-grams — the
// difference here isn't context direction, it's *how* context is represented
// (a compressed running summary vector instead of exact discrete counts).
//
// This is deliberately small (tiny hidden layer, few epochs, capped vocab)
// so it trains in about a second on a short story. Real RNNLMs of this era
// were trained for hours on millions of words — expect this demo model to
// be noticeably rougher than the n-grams on such a tiny amount of text,
// which is itself historically honest: RNNs of this generation were
// genuinely hard to train well, exactly why LSTMs (gated memory) took over
// a few years later.

function buildVocabForRNN(storySentences, bgSentences, maxVocab) {
  const freq = new Map();
  const bump = (w) => freq.set(w, (freq.get(w) || 0) + 1);
  for (const s of storySentences) for (const w of s) bump(w);
  for (const s of bgSentences) for (const w of s) bump(w);
  const storyWords = new Set(storySentences.flat());
  const ranked = [...freq.entries()].sort((a, b) => b[1] - a[1]).map(([w]) => w);
  const vocabSet = new Set(['<unk>', ...storyWords]);
  for (const w of ranked) {
    if (vocabSet.size >= maxVocab) break;
    vocabSet.add(w);
  }
  const vocab = [...vocabSet];
  const idx = new Map(vocab.map((w, i) => [w, i]));
  return { vocab, idx };
}

function zeros(n) { return new Float64Array(n); }
function randInit(n, scale) { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * scale; return a; }
// Xavier/Glorot-style init: scale by 1/sqrt(fan_in) instead of a flat constant,
// so layers of different widths (embedding V->H vs recurrent H->H vs output H->V)
// each start with activations/gradients in a sane range instead of one width
// starting too hot and another too cold.
function xavierInit(rows, cols, fanIn) { return randInit(rows * cols, Math.sqrt(1 / fanIn)); }
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function trainRNN(storyText, backgroundText, opts = {}) {
  const H = opts.hidden || 28;
  const maxVocab = opts.maxVocab || 550;
  const epochs = opts.epochs || 7;
  const lr = opts.lr || 0.35; // Adagrad divides the effective step per-parameter, so this can run hotter than plain SGD's 0.15
  const maxTrainTokens = opts.maxTrainTokens || 4200;

  const storySentences = toSentences(tokenize(storyText));
  const bgSentencesAll = toSentences(tokenize(backgroundText || ''));

  // Cap total training volume so this stays fast; keep the user's own story
  // sentences (repeated a few times so they actually influence the weights)
  // plus as much background as fits the budget.
  const { vocab, idx } = buildVocabForRNN(storySentences, bgSentencesAll, maxVocab);
  const V = vocab.length;
  const UNK = idx.get('<unk>');

  let trainSet = [];
  for (let r = 0; r < 3; r++) trainSet.push(...storySentences); // weight the story up
  let tokenBudget = maxTrainTokens - trainSet.reduce((s, sent) => s + sent.length, 0);
  for (const s of bgSentencesAll) {
    if (tokenBudget <= 0) break;
    trainSet.push(s);
    tokenBudget -= s.length;
  }

  const toId = (w) => (idx.has(w) ? idx.get(w) : UNK);

  // Parameters: input embedding (V x H), recurrent (H x H), output (H x V).
  // Xavier-scaled so each matrix starts appropriately for its own fan-in
  // rather than sharing one arbitrary constant across very different widths.
  const Wxh = xavierInit(V, H, V);
  const Whh = xavierInit(H, H, H);
  const Why = xavierInit(H, V, H);
  const bh = zeros(H);
  const by = zeros(V);

  // Adagrad accumulators: per-parameter adaptive learning rate. Words seen
  // rarely (most of the vocab, on a short story) get proportionally bigger
  // effective updates when they do appear, instead of the same tiny flat
  // learning rate that plain SGD gives every parameter regardless of how
  // often it's actually touched — this matters a lot when training for only
  // a handful of epochs on a few thousand tokens.
  const cacheWxh = zeros(V * H), cacheWhh = zeros(H * H), cacheWhy = zeros(H * V);
  const cacheBh = zeros(H), cacheBy = zeros(V);
  const EPS = 1e-8;

  function softmax(logits) {
    let max = -Infinity; for (const v of logits) if (v > max) max = v;
    let sum = 0; const out = new Float64Array(logits.length);
    for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - max); sum += out[i]; }
    for (let i = 0; i < logits.length; i++) out[i] /= (sum || 1);
    return out;
  }

  function clip(x, c) { return x > c ? c : x < -c ? -c : x; }

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffle(trainSet); // avoid always seeing the (tripled) story sentences in the same clustered order
    for (const sent of trainSet) {
      if (sent.length < 2) continue;
      const ids = sent.map(toId);
      const T = ids.length - 1; // predict ids[1..] from ids[0..T-1]
      if (T <= 0) continue;

      const hs = [zeros(H)];
      const ys = [];
      // forward
      for (let t = 0; t < T; t++) {
        const xId = ids[t];
        const hPrev = hs[t];
        const h = new Float64Array(H);
        for (let j = 0; j < H; j++) {
          let sum = Wxh[xId * H + j] + bh[j];
          for (let k = 0; k < H; k++) sum += Whh[j * H + k] * hPrev[k];
          h[j] = Math.tanh(sum);
        }
        hs.push(h);
        const logits = new Float64Array(V);
        for (let v = 0; v < V; v++) {
          let sum = by[v];
          for (let j = 0; j < H; j++) sum += Why[j * V + v] * h[j];
          logits[v] = sum;
        }
        ys.push(softmax(logits));
      }

      // backward (BPTT)
      const dWxh = zeros(V * H), dWhh = zeros(H * H), dWhy = zeros(H * V), dbh = zeros(H), dby = zeros(V);
      let dhNext = zeros(H);
      for (let t = T - 1; t >= 0; t--) {
        const target = ids[t + 1];
        const y = ys[t];
        const dy = new Float64Array(V);
        for (let v = 0; v < V; v++) dy[v] = y[v] - (v === target ? 1 : 0);

        const h = hs[t + 1];
        const hPrev = hs[t];
        for (let v = 0; v < V; v++) {
          dby[v] += dy[v];
          for (let j = 0; j < H; j++) dWhy[j * V + v] += dy[v] * h[j];
        }
        const dh = new Float64Array(H);
        for (let j = 0; j < H; j++) {
          let sum = dhNext[j];
          for (let v = 0; v < V; v++) sum += dy[v] * Why[j * V + v];
          dh[j] = sum;
        }
        const dhRaw = new Float64Array(H);
        for (let j = 0; j < H; j++) dhRaw[j] = (1 - h[j] * h[j]) * dh[j];

        const xId = ids[t];
        for (let j = 0; j < H; j++) {
          dbh[j] += dhRaw[j];
          dWxh[xId * H + j] += dhRaw[j];
          for (let k = 0; k < H; k++) dWhh[j * H + k] += dhRaw[j] * hPrev[k];
        }
        const dhPrev = zeros(H);
        for (let k = 0; k < H; k++) {
          let sum = 0;
          for (let j = 0; j < H; j++) sum += dhRaw[j] * Whh[j * H + k];
          dhPrev[k] = sum;
        }
        dhNext = dhPrev;
      }

      // clipped Adagrad update: clip raw gradients first (same safety net as
      // before against exploding BPTT gradients on the recurrent weights),
      // then take an adaptive step sized by each parameter's own update
      // history instead of one flat learning rate for the whole network.
      const C = 5;
      for (let i = 0; i < Wxh.length; i++) {
        const g = clip(dWxh[i], C);
        cacheWxh[i] += g * g;
        Wxh[i] -= lr * g / (Math.sqrt(cacheWxh[i]) + EPS);
      }
      for (let i = 0; i < Whh.length; i++) {
        const g = clip(dWhh[i], C);
        cacheWhh[i] += g * g;
        Whh[i] -= lr * g / (Math.sqrt(cacheWhh[i]) + EPS);
      }
      for (let i = 0; i < Why.length; i++) {
        const g = clip(dWhy[i], C);
        cacheWhy[i] += g * g;
        Why[i] -= lr * g / (Math.sqrt(cacheWhy[i]) + EPS);
      }
      for (let i = 0; i < H; i++) {
        const g = clip(dbh[i], C);
        cacheBh[i] += g * g;
        bh[i] -= lr * g / (Math.sqrt(cacheBh[i]) + EPS);
      }
      for (let i = 0; i < V; i++) {
        const g = clip(dby[i], C);
        cacheBy[i] += g * g;
        by[i] -= lr * g / (Math.sqrt(cacheBy[i]) + EPS);
      }
    }
  }

  return { vocab, idx, Wxh, Whh, Why, bh, by, H, V, toId };
}

// Run the trained RNN forward over the left-context tokens only (it is,
// like the n-grams, a purely left-to-right model) and return the softmax
// distribution over the next word.
function rnnPredict(model, leftTokens, k = 8) {
  const { Wxh, Whh, Why, bh, by, H, V, vocab, toId } = model;
  let h = zeros(H);
  const ids = leftTokens.map(toId);
  for (const xId of ids) {
    const hNew = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let sum = Wxh[xId * H + j] + bh[j];
      for (let kk = 0; kk < H; kk++) sum += Whh[j * H + kk] * h[kk];
      hNew[j] = Math.tanh(sum);
    }
    h = hNew;
  }
  const logits = new Float64Array(V);
  for (let v = 0; v < V; v++) {
    let sum = by[v];
    for (let j = 0; j < H; j++) sum += Why[j * V + v] * h[j];
    logits[v] = sum;
  }
  let max = -Infinity; for (const v of logits) if (v > max) max = v;
  let sum = 0; const probs = new Float64Array(V);
  for (let i = 0; i < V; i++) { probs[i] = Math.exp(logits[i] - max); sum += probs[i]; }
  const unkId = model.idx.get('<unk>');
  const pairs = [];
  for (let i = 0; i < V; i++) {
    if (i === unkId) continue;
    pairs.push([vocab[i], probs[i] / (sum || 1)]);
  }
  pairs.sort((a, b) => b[1] - a[1]);
  return pairs.slice(0, k);
}

if (typeof module !== 'undefined') {
  module.exports = { trainRNN, rnnPredict };
}