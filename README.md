# Blank Tape — n-gram vs. LLM fill-in-the-blank lab

A static, client-side site that trains classic n-gram language models
(unigram / bigram / trigram / 4-gram) live in the browser on a short
story you provide, samples the blank 1000 times per model, and shows
the resulting word-frequency distribution — next to a real small
language model (DistilBERT, running in-browser via
[transformers.js](https://github.com/xenova/transformers.js)) filling
the same blank using full bidirectional context.

Everything runs client-side. There is no backend, no API key, and
nothing to configure — which is why it works as a plain GitHub Pages
site.

## Files

```
index.html   – page structure & copy
style.css    – visual design (teletype / punch-tape theme)
ngram.js     – tokenizer + n-gram training/sampling engine
corpus.js    – small original background corpus used for smoothing
app.js       – UI wiring, experiment runner, LLM integration
```

## How the models work

- **N-grams (bi/tri/4-gram):** trained fresh in your browser on the
  story + target sentence you type in, blended with a large (~19k-word,
  ~2,200-sentence) built-in background corpus at a fixed weight
  (`BG_WEIGHT` in `app.js`). Prediction only ever looks at the *n − 1*
  words immediately to the left of the blank — it has no idea what comes
  after, and no idea what the story is "about."

  Smoothing uses **linear (Jelinek-Mercer) interpolation**, the standard
  technique for this era of language modeling, not "stupid backoff."
  Each order blends its own evidence with the order below it, weighted by
  how many times that specific context has actually been seen — so a
  well-attested trigram context can dominate, while a one-off trigram
  barely nudges the bigram estimate. Combined with a much larger
  background corpus, this means bigram/trigram/4-gram genuinely diverge
  from each other instead of all just collapsing toward the unigram
  distribution the moment an exact context is unseen (which is what
  happens with stupid backoff on a small corpus — see `ngram.js` for the
  full rationale).
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
