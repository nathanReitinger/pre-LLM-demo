// tests/test-predictions.js
// Run with: node tests/test-predictions.js
// Node 18+ (needs global fetch). Hits the REAL public infini-gram API live
// — this needs actual internet access from wherever you run it, and will
// NOT work in an offline/sandboxed environment. No API key needed.
//
// Runs the project's manual test prompts at orders 2 (bigram), 3
// (trigram), and 4 (4-gram) — the same infiniNgramWithBackoff call app.js
// makes — and reports whether the expected word appears in the top-8.

const { infiniNgramWithBackoff } = require('../infinigram.js');

const CASES = [
  { text: 'I am sitting in a <blank>', expect: 'chair' },
  { text: 'The cat and the <blank>', expect: 'hat' },
  { text: 'finders <blank> losers weepers', expect: 'keepers' },
  { text: 'new york <blank>', expect: 'state' },
  { text: 'apple, banana, <blank>', expect: 'orange' },
];

function rawLeftContextWords(chunkBefore) {
  const lastEnd = Math.max(
    chunkBefore.lastIndexOf('.'), chunkBefore.lastIndexOf('!'), chunkBefore.lastIndexOf('?')
  );
  const segment = lastEnd >= 0 ? chunkBefore.slice(lastEnd + 1) : chunkBefore;
  return segment.trim().split(/\s+/).filter(Boolean);
}

function splitOnBlank(text) {
  const i = text.indexOf('<blank>');
  if (i < 0) throw new Error(`no <blank> in: ${text}`);
  return text.slice(0, i);
}

async function main() {
  let passed = 0, failed = 0;
  for (const order of [2, 3, 4]) {
    console.log(`\n== order ${order} ==`);
    for (const { text, expect } of CASES) {
      const rawWords = rawLeftContextWords(splitOnBlank(text));
      try {
        const { pairs, usedOrder, backedOff, promptCnt } = await infiniNgramWithBackoff(rawWords, order);
        const top8 = pairs.slice(0, 8).map(([w]) => w);
        const hit = top8.includes(expect.toLowerCase());
        console.log(
          `  ${hit ? 'ok  ' : 'FAIL'} - "${text}" -> want "${expect}" ${hit ? 'FOUND' : 'not found'} in [${top8.join(', ')}]` +
          (backedOff ? ` (backed off to order ${usedOrder}, ctx seen ${promptCnt}x)` : ` (ctx seen ${promptCnt}x)`)
        );
        hit ? passed++ : failed++;
      } catch (err) {
        console.log(`  FAIL - "${text}" -> request error: ${err.message}`);
        failed++;
      }
    }
  }
  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

main();