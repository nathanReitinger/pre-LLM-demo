// infinigram.js
// A REAL pretrained n-gram corpus, queried live, used two ways in this app:
//   1. Next-word statistics for the unigram/bigram/trigram/4-gram rows
//      (infiniNgramDistribution / infiniNgramWithBackoff).
//   2. Real sampled documents to train the embeddings/RNN models' background
//      signal (fetchInfiniGramCorpusText) — this replaced the old
//      livecorpus.js, which pulled random rows from Hugging Face's
//      datasets-server API against three specific datasets. That approach
//      got increasingly unreliable: HF has moved several popular datasets
//      (e.g. Salesforce/wikitext, the Wikitext-103 source) behind gating,
//      so an anonymous browser fetch just fails with a generic network
//      error ("Load failed") — no amount of retrying fixes that. Rather
//      than juggle three flaky dataset-specific endpoints, everything now
//      goes through this one already-proven-reliable API instead.
//
// infini-gram (Liu, Min, Zettlemoyer, Choi & Hajishirzi, 2024, "Infini-gram:
// Scaling Unbounded n-gram Language Models to a Trillion Tokens") is a free,
// public, no-key API hosted by the University of Washington. It holds a
// suffix-array index over real trillion-token pretraining corpora and can
// return the *exact* next-token distribution following any prompt, or the
// *actual real documents* that contain a given phrase, in milliseconds.
// Docs: https://infini-gram.readthedocs.io/en/latest/api.html

const INFINIGRAM_ENDPOINT = 'https://api.infini-gram.io/';
// Dolma v1.7: ~2.6 trillion tokens of web pages, books, Wikipedia, Reddit,
// and code — a broad, general-purpose pretraining corpus, tokenized
// server-side with the Llama-2 tokenizer. Built from (and linked to) the
// real Hugging Face dataset allenai/dolma.
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

// A rotating set of short, extremely common phrases used purely as a way to
// land on a huge, broad swath of the corpus — the actual phrase doesn't
// matter (it's not a "topic"), we just need something virtually every
// English document contains so `find` returns a big, representative range
// of documents to sample from. A different seed each run (plus a random
// rank within its match range) is what makes each fetch pull a different
// slice of real text, the same role a random `offset` played in the old
// Hugging Face row-based fetcher it replaced.
const SAMPLE_SEEDS = [
  'in the', 'on the', 'as well as', 'one of the', 'for example',
  'at the same time', 'according to', 'in addition to', 'as a result of',
  'more than', 'because of', 'in order to', 'at least', 'such as',
];

// Retrieves one real document's text by its rank within a shard's matching
// range (see infini-gram's `find` -> `get_doc_by_rank` two-step document
// search). `spans` in the response is the document already split into text
// spans, so just concatenating them reconstructs the readable text.
async function infiniFetchDocument(shard, rank) {
  const data = await infiniQuery({ query_type: 'get_doc_by_rank', s: shard, rank });
  if (!data.spans) return '';
  return data.spans.map((span) => span[0]).join('');
}

// Pulls `numDocs` real, distinct documents from the live infini-gram index
// and concatenates them into one text blob — ready to be tokenized into
// training sentences exactly the way the old fetchLiveCorpusText output was
// (see embeddings.js / rnn.js, which just want a blob of real prose).
async function fetchInfiniGramCorpusText(opts = {}) {
  const numDocs = opts.numDocs || 14;
  const maxChars = opts.maxChars || 250000;

  const seed = SAMPLE_SEEDS[Math.floor(Math.random() * SAMPLE_SEEDS.length)];
  const findResult = await infiniQuery({ query_type: 'find', query: seed });
  const shards = (findResult.segment_by_shard || [])
    .map((range, shard) => ({ shard, lo: range[0], hi: range[1] }))
    .filter((s) => s.hi > s.lo);
  if (!shards.length) throw new Error('no matching documents found in the live index');

  const picks = [];
  for (let i = 0; i < numDocs; i++) {
    const s = shards[Math.floor(Math.random() * shards.length)];
    const rank = s.lo + Math.floor(Math.random() * (s.hi - s.lo));
    picks.push({ shard: s.shard, rank });
  }

  const texts = await Promise.all(
    picks.map(({ shard, rank }) => infiniFetchDocument(shard, rank).catch(() => ''))
  );

  let text = texts.filter(Boolean).join('\n\n');
  if (!text.trim()) throw new Error('sample came back empty');
  if (text.length > maxChars) text = text.slice(0, maxChars);
  return text;
}

if (typeof module !== 'undefined') {
  module.exports = {
    infiniNgramDistribution, infiniNgramWithBackoff, fetchInfiniGramCorpusText,
    INFINIGRAM_INDEX, INFINIGRAM_LABEL,
  };
}
