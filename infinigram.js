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

// --- Request throttling + retry -------------------------------------------
// infini-gram is a shared public API with real rate limits. This app can
// easily fire off 8+ requests in a burst (one per n-gram order per blank),
// which was hitting 429s and silently falling back to local-only stats
// every time (see ngramPredict's catch block in app.js — it swallows the
// error and just doesn't tell you clearly it happened).
//
// Two independent fixes, both needed:
//   1. A queue that only lets ONE infini-gram request be in flight at a
//      time, with a minimum gap between requests starting — so a burst of
//      8 calls from one run() becomes a gentle drip instead of a slam.
//   2. Real retry-with-exponential-backoff+jitter specifically for 429s
//      (and 5xx, which are also usually transient), so a request that gets
//      rate-limited actually succeeds on the 2nd or 3rd try instead of
//      giving up immediately.
const MIN_GAP_MS = 250;       // minimum time between request starts
const MAX_RETRIES = 4;        // retries after the first attempt
const BASE_BACKOFF_MS = 600;  // doubled each retry, plus jitter

let queueTail = Promise.resolve();
let lastRequestAt = 0;

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// Serializes all infini-gram calls through one queue so they never overlap,
// and enforces a minimum gap between request starts even when several
// callers enqueue at once.
function enqueue(task) {
  const run = async () => {
    const wait = Math.max(0, lastRequestAt + MIN_GAP_MS - Date.now());
    if (wait > 0) await sleep(wait);
    lastRequestAt = Date.now();
    return task();
  };
  const result = queueTail.then(run, run); // run even if a prior task rejected
  // Keep the chain alive regardless of this task's outcome, so one failure
  // doesn't stall every request queued after it.
  queueTail = result.catch(() => {});
  return result;
}

async function rawInfiniQuery(payload) {
  const res = await fetch(INFINIGRAM_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ index: INFINIGRAM_INDEX, ...payload }),
  });
  if (!res.ok) {
    const err = new Error(`infini-gram API returned ${res.status}`);
    err.status = res.status;
    throw err;
  }
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

async function infiniQuery(payload) {
  return enqueue(async () => {
    let attempt = 0;
    for (;;) {
      try {
        return await rawInfiniQuery(payload);
      } catch (err) {
        const retryable = err.status === 429 || (err.status >= 500 && err.status < 600);
        if (!retryable || attempt >= MAX_RETRIES) throw err;
        const backoff = BASE_BACKOFF_MS * Math.pow(2, attempt) + Math.random() * 300;
        console.warn(`infini-gram request failed (${err.message}), retrying in ${Math.round(backoff)}ms — attempt ${attempt + 1}/${MAX_RETRIES}`);
        await sleep(backoff);
        attempt += 1;
      }
    }
  });
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
//
// Throws if every level's request actually FAILS (after retries) — the
// caller (app.js) distinguishes "infini-gram had nothing to say" (empty
// pairs, not an error) from "infini-gram was unreachable" (thrown error),
// so the UI can label a fallback row honestly instead of claiming a blend
// happened when it didn't.
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