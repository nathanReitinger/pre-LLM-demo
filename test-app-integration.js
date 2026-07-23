// tests/test-app-integration.js
// Run with: node tests/test-app-integration.js
// Loads the REAL app.js (as a browser would, via <script type="module">)
// inside jsdom, served over HTTP from the repo root (so ngram-model/,
// static-ngram-model.js, ngram.js etc. all resolve exactly the way they
// do on GitHub Pages), fills in a prompt with a <blank>, checks the
// bigram box, clicks "Run experiment" for real, and asserts the resulting
// table actually used the pretrained static model — this is the thing
// unit tests of static-ngram-model.js alone can't prove: that app.js is
// actually wired up to call it.

const assert = require('assert');
const path = require('path');
const { JSDOM } = require('jsdom');
const { serve } = require('./serve');

const PORT = 8974;
const ROOT = path.join(__dirname, '..');

async function waitFor(fn, timeoutMs = 15000, intervalMs = 50) {
  const start = Date.now();
  for (;;) {
    const v = fn();
    if (v) return v;
    if (Date.now() - start > timeoutMs) throw new Error('waitFor timed out');
    await new Promise(r => setTimeout(r, intervalMs));
  }
}

async function main() {
  const server = await serve(ROOT, PORT);
  let passed = 0, failed = 0;
  try {
    const { VirtualConsole } = require('jsdom');
    const virtualConsole = new VirtualConsole();
    virtualConsole.on('jsdomError', (e) => console.error('jsdomError:', e.message, e.detail || ''));
    virtualConsole.on('error', (...args) => console.error('console.error:', ...args));
    virtualConsole.on('warn', (...args) => console.warn('console.warn:', ...args));

    const dom = await JSDOM.fromURL(`http://localhost:${PORT}/tests/test-page.html`, {
      runScripts: 'dangerously',
      resources: 'usable',
      pretendToBeVisual: true,
      virtualConsole,
    });
    const { window } = dom;
    window.fetch = (url, opts) => fetch(new URL(url, window.document.baseURI).toString(), opts);

    // Wait for app.js's module to finish evaluating and attach its click
    // handler (module scripts execute asynchronously relative to parsing).
    await waitFor(() => window.document.getElementById('run-btn')?.onclick !== undefined || true, 3000);
    // (jsdom attaches listeners via addEventListener, not onclick, so just
    // give module execution a moment to complete instead.)
    await new Promise(r => setTimeout(r, 500));

    const doc = window.document;
    doc.getElementById('prompt').value = 'The cat is on the <blank>.';
    doc.querySelector('input[value=bigram]').checked = true;
    doc.getElementById('runs').value = '1000';

    doc.getElementById('run-btn').dispatchEvent(new window.Event('click'));

    // Wait for the run to finish (status text gets set at the very end of run()).
    await waitFor(() => {
      const s = doc.getElementById('status').textContent;
      return s && s.trim().length > 0 ? s : null;
    }, 20000);

    const statusText = doc.getElementById('status').textContent;
    console.log(`  status: "${statusText}"`);

    const test = (name, fn) => {
      try { fn(); console.log(`  ok  - ${name}`); passed++; }
      catch (e) { console.log(`FAIL  - ${name}\n        ${e.message}`); failed++; }
    };

    test('status line reports the pretrained static model was used', () => {
      assert.ok(/pretrained static model/.test(statusText), `got: "${statusText}"`);
    });

    test('a results table was actually rendered', () => {
      const table = doc.querySelector('table.results-table');
      assert.ok(table, 'no results table found in #results');
    });

    test('the model-cell tag mentions the pretrained corpus, not live training', () => {
      const tag = doc.querySelector('.model-tag')?.textContent || '';
      assert.ok(/pretrained/.test(tag), `got tag: "${tag}"`);
    });

    test('at least one non-empty prediction cell was rendered', () => {
      const cells = [...doc.querySelectorAll('td.pred-cell')].map(td => td.textContent.trim()).filter(Boolean);
      assert.ok(cells.length > 0, 'no prediction cells rendered');
      console.log(`        predictions: ${cells.slice(0, 5).join(' | ')}`);
    });

    test('no error note was added to the results panel', () => {
      const err = doc.querySelector('.error-note');
      assert.strictEqual(err, null, `unexpected error note: ${err && err.textContent}`);
    });

    window.close();
  } finally {
    server.close();
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
