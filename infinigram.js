// infinigram.js
// Queries the REAL, live infini-gram API for exact n-gram statistics —
// this is the sole source of every unigram/bigram/trigram/4-gram
// prediction in the app. No local corpus, no local training, no story
// blending: every number here is an exact count from the real index.
//
// infini-gram (Liu, Min, Zettlemoyer, Choi & Hajishirzi, 2024, "Infini-gram:
// Scaling Unbounded n-gram Language Models to a Trillion Tokens") is a free,
// public, no-key API hosted by the Allen Institute/University of Washington.
// It holds a suffix-array index over real trillion-token corpora and
// returns the *exact* next-token distribution following any prompt, in
// milliseconds — not an estimate, not a sample.
// Docs: https://infini-gram.readthedocs.io/en/latest/api.html

const INFINIGRAM_ENDPOINT = 'https://cold-credit-665b.nathan-reitinger.workers.dev/';
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
// Two distinct failure modes can hit this API:
//   1. HTTP-level failures (429 rate limit, 5xx) — the response comes back
//      with a bad status code.
//   2. Network-level failures — fetch() itself throws (CORS block, DNS,
//      timeout, connection reset). These have NO status code at all.
// This retries EITHER kind, budgets real wall-clock time toward it, and
// logs the actual underlying error so it's possible to tell which failure
// mode is occurring from the browser console.
const MIN_GAP_MS = 300;          // minimum time between request starts
const MAX_RETRIES = 6;           // retries after the first attempt
const BASE_BACKOFF_MS = 700;     // doubled each retry, plus jitter
const MAX_BACKOFF_MS = 6000;     // cap per-wait
const TOTAL_TIME_BUDGET_MS = 28000; // give up around here regardless of retries left
const FETCH_TIMEOUT_MS = 9000;   // a single attempt shouldn't hang forever

let queueTail = Promise.resolve();
let lastRequestAt = 0;

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// Serializes all infini-gram calls through one queue so they never overlap,
// and enforces a minimum gap between request starts even when several
// callers enqueue at once (e.g. every blank of every selected order).
function enqueue(task) {
  const run = async () => {
    const wait = Math.max(0, lastRequestAt + MIN_GAP_MS - Date.now());
    if (wait > 0) await sleep(wait);
    lastRequestAt = Date.now();
    return task();
  };
  const result = queueTail.then(run, run); // run even if a prior task rejected
  queueTail = result.catch(() => {}); // one failure doesn't stall the queue
  return result;
}

async function fetchWithTimeout(url, opts, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function rawInfiniQuery(payload) {
  let res;
  try {
    res = await fetchWithTimeout(
      INFINIGRAM_ENDPOINT,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: INFINIGRAM_INDEX, ...payload }),
      },
      FETCH_TIMEOUT_MS
    );
  } catch (networkErr) {
    const err = new Error(`network error reaching infini-gram: ${networkErr.message || networkErr.name}`);
    err.networkFailure = true;
    throw err;
  }
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
    const startedAt = Date.now();
    let attempt = 0;
    for (;;) {
      try {
        const result = await rawInfiniQuery(payload);
        if (attempt > 0) console.info(`infini-gram request succeeded after ${attempt} retr${attempt === 1 ? 'y' : 'ies'}`);
        return result;
      } catch (err) {
        const retryable = err.networkFailure || err.status === 429 || (err.status >= 500 && err.status < 600);
        const elapsed = Date.now() - startedAt;
        console.warn(`infini-gram request failed (attempt ${attempt + 1}): ${err.message}`);
        if (!retryable || attempt >= MAX_RETRIES || elapsed >= TOTAL_TIME_BUDGET_MS) {
          console.error(`infini-gram giving up after ${attempt + 1} attempt(s), ${elapsed}ms — last error: ${err.message}`);
          throw err;
        }
        const backoff = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * Math.pow(2, attempt)) + Math.random() * 300;
        const remaining = TOTAL_TIME_BUDGET_MS - elapsed;
        await sleep(Math.min(backoff, Math.max(0, remaining)));
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

// Classic n-gram backoff, aimed at the real corpus: try the full
// (order-1)-word context first; if Dolma has never seen that exact string
// (empty distribution), drop the oldest word and try again, all the way
// down to the plain unigram distribution (always non-empty). This is real
// backoff over real data — when it happens, the caller is told exactly
// which order was actually used, so it's never silent.
async function infiniNgramWithBackoff(rawWords, order) {
  // Never ask for more words of context than actually exist — clamp the
  // starting point instead of relying on slice()'s silent negative-index
  // behavior, which quietly reused fewer words than requested without
  // ever reporting it.
  const startN = Math.min(order - 1, rawWords.length);
  const contextLimited = startN < order - 1; // fewer real words existed than this order wanted
  for (let n = startN; n >= 0; n--) {
    const contextText = n === 0 ? '' : rawWords.slice(rawWords.length - n).join(' ');
    const { pairs, promptCnt, approx } = await infiniNgramDistribution(contextText);
    if (pairs.length > 0) {
      return { pairs, usedOrder: n + 1, promptCnt, approx, backedOff: n < startN, contextLimited };
    }
  }
  return { pairs: [], usedOrder: 0, promptCnt: 0, approx: false, backedOff: true, contextLimited };
}

if (typeof module !== 'undefined') {
  module.exports = {
    infiniNgramDistribution, infiniNgramWithBackoff,
    INFINIGRAM_INDEX, INFINIGRAM_LABEL,
  };
}