# Blank Tape — n-gram vs. LLM fill-in-the-blank lab

A static, client-side site that trains classic n-gram language models
(unigram / bigram / trigram / 4-gram) live in the browser on a short
story you provide, samples each blank 1000 times per model, and shows
the resulting word-frequency distribution — next to a real small
language model (DistilBERT, running in-browser via
[transformers.js](https://github.com/xenova/transformers.js)) filling
the same blank(s) using full bidirectional context.

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
index.html    – page structure & copy
style.css     – visual design (teletype / punch-tape theme)
ngram.js      – tokenizer + n-gram training/sampling engine
embeddings.js – co-occurrence → PPMI → SVD word vectors
rnn.js        – tiny live-trained recurrent network
corpus.js     – small original background corpus used for smoothing
app.js        – UI wiring, experiment runner, LLM integration
deploy.yml    – GitHub Actions workflow (copy into .github/workflows/)
```

## How the models work

- **N-grams (bi/tri/4-gram):** trained fresh in your browser on the
  story + target sentence you type in, blended with a large (~42k-word,
  ~3,700-sentence) built-in background corpus at a fixed weight
  (`BG_WEIGHT` in `app.js`), spanning twenty grounded everyday topics —
  including a fruit basket/kitchen topic and a hiking/wardrobe topic, so
  common concrete prompts ("inside the fruit basket was a(n) ___", "I'm on
  a hike so I'm wearing my ___") have real vocabulary to draw on instead of
  falling back to generic filler words. The user's own story is counted
  `STORY_WEIGHT` times (3, in `app.js`) relative to the background corpus,
  so a well-attested story-specific pattern can still win out over generic
  background statistics. Prediction only ever looks at the *n − 1* words
  immediately to the left of the blank — it has no idea what comes after,
  and no idea what the story is "about" beyond that local window.

  Smoothing uses **generalized interpolated Kneser-Ney**, the technique
  that actually made trigram/4-gram models competitive in the late
  1990s–2000s (Chen & Goodman 1998). Every order blends its own evidence
  with the order below it, weighted by how many times that specific
  context has actually been seen — so a well-attested trigram context
  can dominate, while a one-off trigram barely nudges the bigram
  estimate. Two refinements beyond a textbook single-discount version:

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

  Combined with a topic-grounded background corpus (concrete objects
  tied to real places — beach/seashell, forest/deer, city/taxi, etc.,
  not just abstract filler sentences), this means bigram/trigram/4-gram
  genuinely diverge from each other and from the unigram distribution
  instead of collapsing into generic high-frequency words — see
  `ngram.js` for the full rationale.
- **LLM (DistilBERT, masked-language-model mode):** the blank is
  replaced with `[MASK]` and the *whole* passage — both sides of the
  blank — is fed through a small transformer that runs entirely in
  your browser via WebAssembly/WebGPU (no server call). The first run
  downloads and caches roughly 65 MB of model weights from the Hugging
  Face CDN; after that it's instant.

## Deploying to GitHub Pages

1. Create a new GitHub repository (or use an existing one).
2. Copy these five files into the repository root (no build step
   needed — it's plain HTML/CSS/JS).
3. Commit and push:
   ```bash
   git add index.html style.css ngram.js corpus.js app.js README.md
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

The background corpus in `corpus.js` is a small, hand-written stand-in
so the demo trains instantly in-browser. The actual datasets each era
of this field was trained/evaluated on at scale are much bigger:

- [Wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) —
  the standard long-range-dependency Wikipedia language modeling
  benchmark used to evaluate n-gram and early neural LMs.
- [Google Books Ngrams](https://books.google.com/ngrams) — 1–5 gram
  frequency counts over the entire scanned Google Books corpus.
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) — the
  open-source recreation of the WebText corpus GPT-2 was trained on.
- [Wikipedia Dumps](https://dumps.wikimedia.org/) — the raw source most
  embedding models (word2vec, GloVe) and many LLM pretraining corpora
  draw from.

These are linked in the site footer too.

## Customizing

- Swap in a different LLM by changing the model id in `app.js`
  (`getLLMPipeline`) to any other `fill-mask`-compatible model
  published on the Hugging Face Hub that transformers.js supports
  (e.g. `Xenova/bert-base-uncased` for a bigger, slower, sometimes
  more accurate model).
- Add more background text to `corpus.js` to change how the n-gram
  models smooth over sparse story text — more background text pulls
  predictions toward generic, frequent English words.
- Add more model checkboxes by extending `MODEL_ORDER` /
  `MODEL_LABELS` in `app.js` and the `.model-picker` markup in
  `index.html`.
