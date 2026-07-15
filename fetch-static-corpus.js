// fetch-static-corpus.js
// Run ONCE, locally, with Node 18+ (needs global fetch): `node fetch-static-corpus.js`
//
// Pulls real documents from infini-gram (the same reliable, no-key API
// infinigram.js already uses live) until it's collected ~TARGET_TOTAL_CHARS
// of real text, then writes the result out as many small chunk files —
// each safely under GitHub's 100MB-per-file hard limit (no Git LFS needed,
// and Git LFS pointer files don't resolve correctly on GitHub Pages anyway)
// — plus a manifest.json listing them. Commit the whole static-corpus/
// directory this produces straight into the repo.
//
// Adjust TARGET_TOTAL_CHARS / CHUNK_CHARS below to control total size and
// per-file size. ~200MB of UTF-8 text is roughly 200 million characters.

const fs = require('fs');
const path = require('path');

const INFINIGRAM_ENDPOINT = 'https://api.infini-gram.io/';
const INFINIGRAM_INDEX = 'v4_dolma-v1_7_llama';
const OUT_DIR = 'static-corpus';
const TARGET_TOTAL_CHARS = 200 * 1024 * 1024; // ~200MB of text
const CHUNK_CHARS = 8 * 1024 * 1024;          // ~8MB per chunk file

// A broad rotating set of extremely common phrases used purely to land on
// a wide, representative swath of the corpus each time — the phrase itself
// isn't a "topic," it's just something virtually every English document
// contains, so `find` returns a large match range to sample ranks from.
const SAMPLE_SEEDS = [
  'in the', 'on the', 'as well as', 'one of the', 'for example',
  'at the same time', 'according to', 'in addition to', 'as a result of',
  'more than', 'because of', 'in order to', 'at least', 'such as',
  'in fact', 'due to', 'in general', 'in particular', 'with respect to',
  'in contrast', 'as a result', 'in other words', 'for instance', 'in this case',
];

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

async function fetchDocument(shard, rank) {
  const data = await infiniQuery({ query_type: 'get_doc_by_rank', s: shard, rank });
  if (!data.spans) return '';
  return data.spans.map((span) => span[0]).join('');
}

async function findShards(seed) {
  const findResult = await infiniQuery({ query_type: 'find', query: seed });
  return (findResult.segment_by_shard || [])
    .map((range, shard) => ({ shard, lo: range[0], hi: range[1] }))
    .filter((s) => s.hi > s.lo);
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  let totalChars = 0;
  let chunkIndex = 0;
  let buffer = [];
  let bufferChars = 0;
  const manifest = { chunks: [], totalChars: 0, generatedAt: new Date().toISOString() };

  function flushChunk() {
    if (!bufferChars) return;
    chunkIndex += 1;
    const name = `chunk-${String(chunkIndex).padStart(4, '0')}.txt`;
    fs.writeFileSync(path.join(OUT_DIR, name), buffer.join('\n\n'));
    manifest.chunks.push(name);
    console.log(`wrote ${name} (${bufferChars} chars, ${totalChars} total so far)`);
    buffer = [];
    bufferChars = 0;
  }

  let seedIdx = 0;
  while (totalChars < TARGET_TOTAL_CHARS) {
    const seed = SAMPLE_SEEDS[seedIdx % SAMPLE_SEEDS.length];
    seedIdx += 1;

    let shards;
    try {
      shards = await findShards(seed);
    } catch (e) {
      console.error('find failed, skipping seed:', e.message);
      continue;
    }
    if (!shards.length) continue;

    // Pull a batch of documents per seed so we're not doing one HTTP round
    // trip per document for the whole run.
    const batchSize = 25;
    const picks = [];
    for (let i = 0; i < batchSize; i++) {
      const s = shards[Math.floor(Math.random() * shards.length)];
      const rank = s.lo + Math.floor(Math.random() * (s.hi - s.lo));
      picks.push({ shard: s.shard, rank });
    }

    const texts = await Promise.all(
      picks.map(({ shard, rank }) => fetchDocument(shard, rank).catch(() => ''))
    );

    for (const t of texts) {
      if (!t) continue;
      buffer.push(t);
      bufferChars += t.length;
      totalChars += t.length;
      if (bufferChars >= CHUNK_CHARS) flushChunk();
      if (totalChars >= TARGET_TOTAL_CHARS) break;
    }
    console.log(`progress: ${(totalChars / 1024 / 1024).toFixed(1)}MB / ${(TARGET_TOTAL_CHARS / 1024 / 1024).toFixed(0)}MB`);
  }
  flushChunk();

  manifest.totalChars = totalChars;
  fs.writeFileSync(path.join(OUT_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2));
  console.log(`Done. ${chunkIndex} chunk files, ${(totalChars / 1024 / 1024).toFixed(1)}MB total, manifest.json written to ${OUT_DIR}/`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
