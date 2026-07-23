// tests/test-static-ngram-model.js
// Run with: node tests/test-static-ngram-model.js
// Exercises static-ngram-model.js against REAL output of train.py (a tiny
// fixture corpus is trained by test-corpus-pipeline.sh before this runs),
// not mocked data — so a format mismatch between the Python writer and the
// JS reader would actually be caught here.

const assert = require('assert');
const path = require('path');
const { serve } = require('./serve');
const {
  shardFor,
  loadStaticModelMeta,
  staticModelAvailable,
  loadStaticNgramModel,
  staticRunSamplingExperiment,
} = require('../static-ngram-model.js');

const PORT = 8973;
const ROOT = path.join(__dirname, '..');

let passed = 0, failed = 0;
async function test(name, fn) {
  try {
    await fn();
    console.log(`  ok  - ${name}`);
    passed++;
  } catch (e) {
    console.log(`FAIL  - ${name}`);
    console.log(`        ${e.message}`);
    failed++;
  }
}

async function main() {
  const server = await serve(ROOT, PORT);
  // Node 18+ ships a global fetch; just point relative paths at our server
  // by monkey-patching fetch to prefix the base URL, mirroring how a
  // browser resolves "ngram-model/x.json" against the page origin.
  const base = `http://localhost:${PORT}/`;
  const realFetch = fetch;
  global.fetch = (url, opts) => realFetch(new URL(url, base).toString(), opts);

  // --- shardFor must match train.py's shard_for() bit-for-bit -----------
  await test('shardFor matches known Python reference values', () => {
    // Cross-checked directly against train.py's shard_for() via a
    // one-off `python3 -c` run for these exact inputs (see
    // tests/README.md for the reference commands).
    assert.strictEqual(shardFor([], 32), 7 % 32);
    assert.strictEqual(shardFor([0], 32), (7 * 131 + 0) % 32);
    assert.strictEqual(shardFor([5, 12], 128), ((((7 * 131 + 5) >>> 0) * 131 + 12) >>> 0) % 128);
  });

  await test('shardFor stays within [0, numShards) for large ids', () => {
    for (const ids of [[65535], [65535, 65535, 65535], [0, 0, 0]]) {
      const s = shardFor(ids, 256);
      assert.ok(s >= 0 && s < 256, `shard ${s} out of range for ${ids}`);
    }
  });

  // --- model loading ------------------------------------------------------
  await test('staticModelAvailable() is true when files exist', async () => {
    // NOTE: loadStaticModelMeta caches its promise at module scope, so this
    // must run first, against ngram-model-test's manifest via a relative
    // fetch — the module always requests "ngram-model/manifest.json", so
    // the test server's document root must have ngram-model/ pointed at
    // our fixture (see test-corpus-pipeline.sh, which symlinks it).
    const ok = await staticModelAvailable();
    assert.strictEqual(ok, true, 'expected ngram-model/ (symlinked to the fixture) to load');
  });

  await test('vocab.json and manifest.json parse into expected shape', async () => {
    const meta = await loadStaticModelMeta();
    assert.ok(meta.manifest.orders.includes(2));
    assert.ok(meta.manifest.orders.includes(3));
    assert.ok(meta.manifest.orders.includes(4));
    assert.ok(meta.vocab.words.length > 10);
    assert.strictEqual(meta.vocab.words[0], '<s>');
    assert.strictEqual(meta.vocab.words[1], '</s>');
    assert.ok(meta.wordToId.has('the'));
  });

  // --- distribution correctness -------------------------------------------
  await test('distribution() normalizes to (approximately) 1', async () => {
    const model = await loadStaticNgramModel(3);
    const dist = await model.distribution(['the', 'cat']);
    const total = dist.reduce((s, [, p]) => s + p, 0);
    assert.ok(Math.abs(total - 1) < 1e-6, `sum was ${total}`);
  });

  await test('distribution() covers the whole vocabulary', async () => {
    const meta = await loadStaticModelMeta();
    const model = await loadStaticNgramModel(2);
    const dist = await model.distribution(['the']);
    assert.strictEqual(dist.length, meta.vocab.words.length);
  });

  await test('a strongly-attested bigram continuation outranks an unrelated word', async () => {
    // The fixture corpus is generated from templates like "the cat sat on
    // the table." / "the cat looked at the old book." etc — after "the cat"
    // a verb phrase ("sat", "looked", "ran", "found", "carried", "pointed")
    // is real signal, while a content noun that's never adjacent to "cat"
    // in the templates ("lamp") should not outrank it.
    const model = await loadStaticNgramModel(2);
    const dist = await model.distribution(['cat']);
    const byWord = new Map(dist);
    assert.ok(byWord.get('sat') > byWord.get('lamp'),
      `expected P(sat|cat)=${byWord.get('sat')} > P(lamp|cat)=${byWord.get('lamp')}`);
  });

  await test('unseen long context backs off instead of throwing or returning empty', async () => {
    const model = await loadStaticNgramModel(4);
    const dist = await model.distribution(['xyzzy_not_a_real_word', 'the', 'cat']);
    assert.ok(dist.length > 0);
    const total = dist.reduce((s, [, p]) => s + p, 0);
    assert.ok(Math.abs(total - 1) < 1e-6);
  });

  await test('4-gram order caps down gracefully for a short context', async () => {
    const model = await loadStaticNgramModel(4);
    // Only one word of left context available — must not crash reaching
    // for context it doesn't have.
    const dist = await model.distribution(['cat']);
    assert.ok(dist.length > 0);
  });

  await test('staticRunSamplingExperiment returns counts that sum to `runs`', async () => {
    const model = await loadStaticNgramModel(2);
    const runs = 500;
    const pairs = await staticRunSamplingExperiment(model, ['the'], runs);
    const sum = pairs.reduce((s, [, c]) => s + c, 0);
    assert.strictEqual(sum, runs);
  });

  await test('a never-seen order-2 model (unigram fallback) still returns a real distribution', async () => {
    const model = await loadStaticNgramModel(1);
    const dist = await model.distribution(['the', 'cat', 'sat']); // order forced down to 1 internally
    assert.ok(dist.length > 0);
    const total = dist.reduce((s, [, p]) => s + p, 0);
    assert.ok(Math.abs(total - 1) < 1e-6);
  });

  server.close();
  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
