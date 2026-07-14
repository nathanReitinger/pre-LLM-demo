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

function trainRNN(storyText, backgroundText, opts = {}) {
  const H = opts.hidden || 28;
  const maxVocab = opts.maxVocab || 550;
  const epochs = opts.epochs || 6;
  const lr = opts.lr || 0.15;
  const maxTrainTokens = opts.maxTrainTokens || 5000;

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
  const Wxh = randInit(V * H, 0.1);
  const Whh = randInit(H * H, 0.1 / Math.sqrt(H));
  const Why = randInit(H * V, 0.1);
  const bh = zeros(H);
  const by = zeros(V);

  function softmax(logits) {
    let max = -Infinity; for (const v of logits) if (v > max) max = v;
    let sum = 0; const out = new Float64Array(logits.length);
    for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - max); sum += out[i]; }
    for (let i = 0; i < logits.length; i++) out[i] /= (sum || 1);
    return out;
  }

  function clip(x, c) { return x > c ? c : x < -c ? -c : x; }

  for (let epoch = 0; epoch < epochs; epoch++) {
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

      // clipped SGD update
      const C = 5;
      for (let i = 0; i < Wxh.length; i++) Wxh[i] -= lr * clip(dWxh[i], C);
      for (let i = 0; i < Whh.length; i++) Whh[i] -= lr * clip(dWhh[i], C);
      for (let i = 0; i < Why.length; i++) Why[i] -= lr * clip(dWhy[i], C);
      for (let i = 0; i < H; i++) bh[i] -= lr * clip(dbh[i], C);
      for (let i = 0; i < V; i++) by[i] -= lr * clip(dby[i], C);
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