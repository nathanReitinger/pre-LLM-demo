# Blank Tape — n-gram vs. LLM fill-in-the-blank lab

A static, client-side site that fills in the blank(s) of a short story
you provide using six stages of language modeling history side by side:
classic n-gram models (unigram / bigram / trigram / 4-gram) that blend
what your own story says with a live query to a real pretrained
trillion-token n-gram corpus ([infini-gram](https://infini-gram.io/)
over Dolma-v1.7), live-trained word embeddings and a live-trained RNN,
and a real small language model (DistilBERT, running in-browser via
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
index.html    – page structure & copy
style.css     – visual design (teletype / punch-tape theme)
ngram.js      – tokenizer + n-gram training/sampling engine (story-only now)
infinigram.js – live queries to the real infini-gram API (pretrained n-gram corpus)
embeddings.js – co-occurrence → PPMI → SVD word vectors
rnn.js        – tiny live-trained recurrent network
corpus.js     – small original background corpus (used by embeddings/RNN only)
livecorpus.js – fetches a live HF dataset sample (used by embeddings/RNN only)
app.js        – UI wiring, experiment runner, LLM integration
deploy.yml    – GitHub Actions workflow (copy into .github/workflows/)
```

## How the models work

- **N-grams (uni/bi/tri/4-gram):** each order is trained *locally*, in
  your browser, on nothing but the story + target sentence you typed in
  (see `ngram.js` — generalized interpolated Kneser-Ney, same as before,
  full rationale below). That local model supplies the story-specific
  signal. For the *background* signal — what used to come from a small
  hand-written corpus or a ~40-row live HF sample — every prediction now
  also makes a live query to **[infini-gram](https://infini-gram.io/)**
  (Liu, Min, Zettlemoyer, Choi & Hajishirzi, 2024), a free public API with
  no key and no download, backed by a suffix-array index over
  **[Dolma-v1.7](https://huggingface.co/datasets/allenai/dolma)**
  (~2.6 trillion tokens of real web text, books, Wikipedia, Reddit, and
  code). This is an *actual* pretrained n-gram language model: exact
  next-word counts and probabilities, computed on the fly against real
  data, in milliseconds — not something trained live on a toy corpus.
  See `infinigram.js`.

  The two are blended per blank: `alpha = localCount / (localCount + K)`
  weights the local (story-only) distribution, and `(1 - alpha)` weights
  the live infini-gram distribution, so a well-attested story-specific
  pattern (a name, a repeated phrase) can still win out, exactly like the
  old `STORY_WEIGHT`-vs-background-corpus design — just with a real
  trillion-token corpus standing in for "background" instead of a few
  thousand local sentences. If infini-gram has never seen the exact
  `order − 1`-word context verbatim, it backs off to a shorter context
  itself (bigram → unigram, etc.), the same idea as the local
  Kneser-Ney backoff, just aimed at the real corpus. If the API is
  unreachable, that blank's row falls back to pure local story
  statistics rather than failing outright. Prediction only ever
  considers the words immediately to the left of the blank — no
  n-gram order here has any idea what comes after it.

  The **local** half still uses generalized interpolated Kneser-Ney, the
  technique that actually made trigram/4-gram models competitive in the
  late 1990s–2000s (Chen & Goodman 1998). Every order blends its own
  evidence with the order below it, weighted by how many times that
  specific context has actually been seen. Two refinements beyond a
  textbook single-discount version:

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

  See `ngram.js` for the full local-model rationale and `infinigram.js`
  for the live query / backoff logic.
- **LLM (DistilBERT, masked-language-model mode):** the blank is
  replaced with `[MASK]` and the *whole* passage — both sides of the
  blank — is fed through a small transformer that runs entirely in
  your browser via WebAssembly/WebGPU (no server call). The first run
  downloads and caches roughly 65 MB of model weights from the Hugging
  Face CDN; after that it's instant.

## Deploying to GitHub Pages

1. Create a new GitHub repository (or use an existing one).
2. Copy all the files listed above into the repository root (no build
   step needed — it's plain HTML/CSS/JS).
3. Commit and push:
   ```bash
   git add index.html style.css ngram.js infinigram.js embeddings.js rnn.js corpus.js livecorpus.js app.js README.md deploy.yml
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

The n-gram models query **[infini-gram](https://infini-gram.io/)** live
against **[Dolma-v1.7](https://huggingface.co/datasets/allenai/dolma)**
(~2.6 trillion tokens) — a real corpus at real scale, not a stand-in.
`corpus.js` is still a small, hand-written stand-in, but it's now only
the default background text for the embeddings/RNN models, which train
live in-browser and so need to stay small for speed. The actual datasets
each era of this field was trained/evaluated on at scale:

- [Dolma-v1.7](https://huggingface.co/datasets/allenai/dolma) — the
  ~2.6-trillion-token pretraining corpus the n-gram models above query
  live through infini-gram (web text, books, Wikipedia, Reddit, code).
- [Wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) —
  the standard long-range-dependency Wikipedia language modeling
  benchmark used to evaluate n-gram and early neural LMs. Selectable as
  a live embeddings/RNN background source above.
- [Google Books Ngrams](https://books.google.com/ngrams) — 1–5 gram
  frequency counts over the entire scanned Google Books corpus.
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) — the
  open-source recreation of the WebText corpus GPT-2 was trained on.
  Selectable as a live embeddings/RNN background source above.
- [Wikipedia Dumps](https://dumps.wikimedia.org/) — the raw source most
  embedding models (word2vec, GloVe) and many LLM pretraining corpora
  draw from. Selectable (via a snapshot) as a live embeddings/RNN
  background source above.

These are linked in the site footer too.

## Customizing

- Swap in a different LLM by changing the model id in `app.js`
  (`getLLMPipeline`) to any other `fill-mask`-compatible model
  published on the Hugging Face Hub that transformers.js supports
  (e.g. `Xenova/bert-base-uncased` for a bigger, slower, sometimes
  more accurate model).
- Swap in a different infini-gram index by changing `INFINIGRAM_INDEX`
  in `infinigram.js` (e.g. to `v4_rpj_llama_s4` for RedPajama, or
  `v4_dclm-baseline_llama` for DCLM-baseline) — see the full list of
  available indexes in the
  [infini-gram API docs](https://infini-gram.readthedocs.io/en/latest/api.html).
- Adjust how much the n-gram models trust story-local evidence vs. the
  live infini-gram background by changing `NGRAM_LOCAL_K` /
  `UNIGRAM_LOCAL_K` in `app.js` — smaller values make the story win out
  with less repetition; larger values lean harder on infini-gram.
- Add more background text to `corpus.js` to change how the embeddings
  and RNN models smooth over sparse story text (this no longer affects
  the n-gram models, which always use infini-gram for background) —
  more background text pulls predictions toward generic, frequent
  English words.
- Add more model checkboxes by extending `MODEL_ORDER` /
  `MODEL_LABELS` in `app.js` and the `.model-picker` markup in
  `index.html`.
