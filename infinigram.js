// infinigram.js
// A REAL pretrained n-gram corpus, queried live, in place of the small
// hand-written / lightly-sampled background text the n-gram models used to
// train on locally.
//
// infini-gram (Liu, Min, Zettlemoyer, Choi & Hajishirzi, 2024, "Infini-gram:
// Scaling Unbounded n-gram Language Models to a Trillion Tokens") is a free,
// public, no-key API hosted by the University of Washington. It holds a
// suffix-array index over real trillion-token pretraining corpora and can
// return the *exact* next-token distribution following any prompt in
// milliseconds — an actual n-gram language model, just computed on the fly
// against real data instead of counted locally in-browser.
// Docs: https://infini-gram.readthedocs.io/en/latest/api.html
//
// This is what the unigram/bigram/trigram/4-gram checkboxes now use for
// their *background* signal. The user's own story is still trained locally
// (see ngram.js / app.js's ngramPredict) so a story-specific pattern can
// still win out — infini-gram just replaces the small local/live-sampled
// corpus as the thing that local model backs off to when the story itself
// doesn't say enough.

const INFINIGRAM_ENDPOINT = 'https://api.infini-gram.io/';
// Dolma v1.7: ~2.6 trillion tokens of web pages, books, Wikipedia, Reddit,
// and code — a broad, general-purpose pretraining corpus, tokenized
// server-side with the Llama-2 tokenizer.
const INFINIGRAM_INDEX = 'v4_dolma-v1_7_llama';
const INFINIGRAM_LABEL = 'Dolma-v1.7, 2.6T tokens (via infini-gram)';

// SentencePiece's word-start marker (U+2581 "▁"). A token that begins with
// it is the first piece of a new, space-preceded word; a token without it
// is a mid-word continuation piece (e.g. "ing", "tion") that isn't a
// standalone guessable word for this demo's table, so those are dropped.
const WORD_START = '\u2581';

async function infiniQuery(payload) {
  const res = await fetch(INFINIGRAM_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ index: INFINIGRAM_INDEX, ...payload }),
  });
  if (!res.ok) throw new Error(`infini-gram API returned ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// Real next-token distribution for an EXACT (order-1)-word context, queried
// against the full Dolma index. contextText === '' queries the plain
// whole-corpus unigram distribution. Returns word-level probabilities
// (merged/renormalized over whole-word tokens only).
async function infiniNgramDistribution(contextText) {
  const data = await infiniQuery({ query_type: 'ntd', query: contextText, max_support: 1000 });
  const byId = data.result_by_token_id || {};
  const merged = new Map(); // word -> prob
  let keptMass = 0;
  for (const key in byId) {
    const { token, prob } = byId[key];
    if (!token || !token.startsWith(WORD_START)) continue; // mid-word piece
    const word = token.slice(WORD_START.length).toLowerCase();
    if (!/^[a-z']+$/.test(word)) continue; // punctuation/number/special token
    merged.set(word, (merged.get(word) || 0) + prob);
    keptMass += prob;
  }
  const pairs = [...merged.entries()];
  if (keptMass > 0) for (let i = 0; i < pairs.length; i++) pairs[i][1] /= keptMass;
  pairs.sort((a, b) => b[1] - a[1]);
  return { pairs, promptCnt: data.prompt_cnt || 0, approx: !!data.approx };
}

// Classic n-gram backoff, just aimed at the real corpus instead of a local
// one: try the full (order-1)-word context first; if Dolma has never seen
// that exact string (prompt_cnt/empty distribution), drop the oldest word
// and try again, all the way down to the plain unigram distribution (which
// is always non-empty). This is the same idea as the local Kneser-Ney
// backoff elsewhere in this app, just applied to the background corpus.
async function infiniNgramWithBackoff(rawWords, order) {
  for (let n = order - 1; n >= 0; n--) {
    const contextText = n === 0 ? '' : rawWords.slice(rawWords.length - n).join(' ');
    const { pairs, promptCnt, approx } = await infiniNgramDistribution(contextText);
    if (pairs.length > 0) return { pairs, usedOrder: n + 1, promptCnt, approx, backedOff: n < order - 1 };
  }
  return { pairs: [], usedOrder: 0, promptCnt: 0, approx: false, backedOff: true };
}

if (typeof module !== 'undefined') {
  module.exports = { infiniNgramDistribution, infiniNgramWithBackoff, INFINIGRAM_INDEX, INFINIGRAM_LABEL };
}
