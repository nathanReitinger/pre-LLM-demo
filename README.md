# Blank Tape — n-gram vs. LLM fill-in-the-blank lab

A static, client-side site that fills in the blank(s) of a short story
you provide using six stages of language modeling history side by side:
classic n-gram models (unigram / bigram / trigram / 4-gram), live-trained
word embeddings, and a live-trained RNN — all trained on your own story
blended with a real bundled text corpus — plus a real small language
model (DistilBERT, running in-browser via
[transformers.js](https://github.com/xenova/transformers.js)) filling
the same blank(s) using full bidirectional context. Each model samples
each blank up to 1000 times and shows the resulting word-frequency
distribution.

Mark a word to guess with `<blank>` (case-insensitive; a run of three or
more underscores, `___`, still works too, for backwards compatibility).
A passage can contain **more than one** `<blank>` — each one gets its
own results section, and every model predicts each blank independently
from its own surrounding real words. Other, still-unresolved blanks in
the same passage are never fed in as context to the n-gram/embeddings/RNN
models (there's no real word there to condition on yet); the masked LLM
is the exception, since a single forward pass over the whole passage with
one `[MASK]` per blank lets it use the *actual* surrounding text for every
hole at once, which is exactly what bidirectional attention is for.

## Files

```
index.html      – page structure & copy
style.css       – visual design (teletype / punch-tape theme)
ngram.js        – tokenizer + n-gram training/sampling engine
embeddings.js   – co-occurrence → PPMI → SVD word vectors
rnn.js          – tiny live-trained recurrent network
corpus.js       – small built-in background corpus (fallback only)
static-corpus.js – fetches a random slice of the bundled static-corpus/ chunk files
app.js          – UI wiring, experiment runner, LLM integration
make-manifest.py – regenerates static-corpus/manifest.json from chunk-*.txt files
deploy.yml      – GitHub Actions workflow (copy into .github/workflows/)
```

## How the models work

- **N-grams (uni/bi/tri/4-gram):** each order is trained *locally*, in
  your browser, using generalized interpolated Kneser-Ney smoothing — the
  technique that actually made trigram/4-gram models competitive in the
  late 1990s–2000s (Chen & Goodman 1998). Every order blends its own
  evidence with the order below it, weighted by how many times that
  specific context has actually been seen. Training text is your own
  story (repeated `STORY_WEIGHT` times, currently 3x, so a well-attested
  story-specific pattern can win out) mixed with a real text sample
  pulled from the bundled `static-corpus/` chunk files (repeated
  `NGRAM_BACKGROUND_WEIGHT` times, currently 1x) — see `buildModel` in
  `ngram.js` and the n-gram training loop in `app.js`. This used to
  instead blend local counts with a live query to the
  [infini-gram](https://infini-gram.io/) API against Dolma-v1.7; that API
  stopped being reliable enough to depend on for a page load, so every
  model now trains entirely on the same locally-available text and
  there's no runtime network dependency at all beyond this site's own
  static files.

  Two refinements beyond a textbook single-discount version of
  Kneser-Ney:

  - **Continuation counts at every backed-off order, not just the
    unigram.** The classic explanation of KN only mentions "a word that
    follows many different words ranks higher than one that follows
    the same word often" at the unigram level. This implementation
    applies that same idea whenever *any* order is being backed off to
    — a trigram model backing off to bigram uses how many distinct
    contexts each bigram continues, not just its raw count.
  - **Per-order discount estimated from the corpus**, `D = n1/(n1+2·n2)`
    (the original Kneser-Ney formula), instead of one hardcoded constant
    for every order — a sparse order (4-gram on a short story, where
    almost every count is 1) gets pulled toward a smaller, safer
    discount automatically.

  See `ngram.js` for the full rationale. Prediction only ever considers
  the words immediately to the left of the blank — no n-gram order here
  has any idea what comes after it.
- **Embeddings (PPMI + SVD) and RNN:** trained live in-browser the same
  way as before — story text plus the same bundled `static-corpus/`
  background sample, fetched once per run (see `static-corpus.js`,
  `embeddings.js`, `rnn.js`).
- **LLM (DistilBERT, masked-language-model mode):** the blank is
  replaced with `[MASK]` and the *whole* passage — both sides of the
  blank — is fed through a small transformer that runs entirely in
  your browser via WebAssembly/WebGPU (no server call). The first run
  downloads and caches roughly 65 MB of model weights from the Hugging
  Face CDN; after that it's instant.

## The bundled static corpus

`static-corpus/` holds a real text sample (~1GB total, from
[Wikipedia dumps](https://dumps.wikimedia.org/)) split into
`chunk-0001.txt`, `chunk-0002.txt`, … so no single file approaches
GitHub's 100MB limit. `static-corpus.js` reads `static-corpus/manifest.json`
to know which chunk files exist, picks one at random, and takes a random
slice out of it — plenty of real prose for a fast in-browser training run
without downloading the whole thing. Regenerate the manifest any time you
add/remove chunk files:

```bash
python3 make-manifest.py
```

## Deploying to GitHub Pages

1. Create a new GitHub repository (or use an existing one).
2. Copy all the files listed above — plus your `static-corpus/` directory
   (chunk files + `manifest.json`) — into the repository root (no build
   step needed — it's plain HTML/CSS/JS).
3. Commit and push:
   ```bash
   git add index.html style.css ngram.js embeddings.js rnn.js corpus.js static-corpus.js app.js README.md static-corpus/ deploy.yml
   git commit -m "Add Blank Tape n-gram/LLM demo"
   git push
   ```
4. In the repo, go to **Settings → Pages**.
5. Under **Build and deployment**, set **Source** to "Deploy from a
   branch," pick the branch (usually `main`) and the root folder
   (`/`), then save.
6. Wait a minute or two, then visit the URL GitHub shows you (usually
   `https://<username>.github.io/<repo-name>/`).

No other configuration is required. If you rename files, update the
`<script>` and `<link>` tags in `index.html` to match.

## Real-world corpora these techniques were trained on

Every model in this app now trains on the same bundled
[Wikipedia](https://dumps.wikimedia.org/) sample under `static-corpus/`.
The actual datasets each era of this field was trained/evaluated on at
scale, for reference:

- [Wikipedia Dumps](https://dumps.wikimedia.org/) — the raw source this
  app's bundled `static-corpus/` sample is drawn from, and the source
  most embedding models (word2vec, GloVe) historically trained on too.
- [Wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) —
  the standard long-range-dependency Wikipedia language modeling
  benchmark used to evaluate n-gram and early neural LMs.
- [Google Books Ngrams](https://books.google.com/ngrams) — 1–5 gram
  frequency counts over the entire scanned Google Books corpus.
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) — the
  open-source recreation of the WebText corpus GPT-2 was trained on.
- [Dolma-v1.7](https://huggingface.co/datasets/allenai/dolma) — the
  ~2.6-trillion-token corpus behind [infini-gram](https://infini-gram.io/),
  which this app used to query live for the n-gram models' background
  signal. That live dependency was removed (see above); Dolma is listed
  here purely for historical/reference interest.

These are linked in the site footer too.

## Customizing

- Swap in a different LLM by changing the model id in `app.js`
  (`getLLMPipeline`) to any other `fill-mask`-compatible model
  published on the Hugging Face Hub that transformers.js supports
  (e.g. `Xenova/bert-base-uncased` for a bigger, slower, sometimes
  more accurate model).
- Adjust how much every model trusts the story vs. the bundled
  background corpus by changing `STORY_WEIGHT` (story repeat count) and
  `NGRAM_BACKGROUND_WEIGHT` (background repeat count, n-grams only) in
  `app.js` — a smaller background weight (or a larger story weight)
  makes the story win out with less repetition; the reverse leans harder
  on the generic background text.
- Add or regenerate chunk files under `static-corpus/` to change what
  background text every model smooths over (see "The bundled static
  corpus" above) — more/different background text pulls predictions
  toward whatever's common in that text.
- Add more model checkboxes by extending `MODEL_ORDER` /
  `MODEL_LABELS` in `app.js` and the `.model-picker` markup in
  `index.html`.
