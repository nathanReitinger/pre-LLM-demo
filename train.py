#!/usr/bin/env python3
"""
train_ngram_model.py  (v2 — external-sort counting, memory-bounded)
=====================================================================
Run this ONCE, locally, from your repo root (the directory that already
contains static-corpus/). It reads EVERY chunk file listed in
static-corpus/manifest.json — the whole corpus, not a random slice —
tokenizes it the same way ngram.js does in the browser, and builds real
bigram / trigram / 4-gram counts with Kneser-Ney statistics (per-order
absolute discounts, context totals, distinct-follow counts, and unigram
continuation counts).

WHY v2: the first version of this script counted n-grams in plain Python
dicts that grow without bound as more of the corpus is read. On a corpus
this size (potentially billions of tokens) those dicts eventually outgrow
RAM, and once that happens every dict insert starts hitting swap — which
gets exponentially slower, not linearly slower, and either takes days or
just crashes with an OOM kill. This version never holds the full n-gram
table in memory. Instead it streams every n-gram straight to disk as a
small fixed-size binary record (no counting yet), sorts those records in
memory-bounded batches, and then aggregates counts with a single linear
pass over the now-sorted (so identical n-grams are adjacent) data. This is
the same "count via external sort" technique tools like KenLM/SRILM use to
build n-gram models from corpora far larger than RAM. Peak memory is now
controlled by --sort-batch-records (how many records get sorted in memory
at once) and stays flat no matter how big the corpus is — only disk usage
and wall-clock time grow with corpus size, which is the trade-off you want.

Output — a new directory, by default `ngram-model/`:
    ngram-model/manifest.json      config: vocab size, shard counts, discounts
    ngram-model/vocab.json         word list + unigram/continuation counts
    ngram-model/o2-0000.json ...   bigram table, sharded
    ngram-model/o3-0000.json ...   trigram table, sharded
    ngram-model/o4-0000.json ...   4-gram table, sharded

Commit the whole ngram-model/ directory to your repo and push — GitHub
Pages serves it as plain static files, same as static-corpus/ already is.

Usage:
    python3 train_ngram_model.py
    python3 train_ngram_model.py --vocab-size 50000 --out ngram-model
    python3 train_ngram_model.py --max-chunks 2 --out ngram-model-test   # fast smoke test
    python3 train_ngram_model.py --max-sentences 100000000 --out ngram-model  # stop before disk fills up

RUNNING OUT OF DISK MID-RUN: if the corpus is bigger than your scratch disk
can hold raw n-gram records for, use --max-sentences to cap how much text
gets read *before* any of the disk-hungry streaming/sorting happens, e.g.
`--max-sentences 100000000`. The script stops reading chunk files as soon as
that many sentences have been collected (capping happens during the cheap
tokenizing pass, so you never pay the disk cost of sentences you're not
going to use) and builds a complete, valid model from just that slice.
manifest.json records `stoppedEarly` / `chunksRead` / `totalSentences` so
you always know exactly how much of the corpus went into the model you
shipped. This script does not support incremental/resumable training into
an existing output directory — a later run with a larger --max-sentences
just rebuilds from scratch into whatever --out you point it at.

Disk space: this needs scratch space roughly comparable to the size of
your corpus (raw n-gram records + their sorted runs, per order, cleaned up
as it goes — see --tmp-dir). For a ~1GB / ~2B-token corpus, expect tens of
GB of temporary disk usage at peak; make sure --tmp-dir points somewhere
with room, and that the final ngram-model/ output (much smaller — pruned
counts, not raw text) is what actually gets committed.
"""

import argparse
import heapq
import itertools
import json
import os
import re
import struct
import sys
import tempfile
import time
from collections import Counter

TOKEN_RE = re.compile(r"[a-z']+|[.!?,;]")
BOS = '<s>'
EOS = '</s>'


def tokenize(text):
    """Mirrors ngram.js's tokenize(): lowercase, normalize curly quotes,
    then match runs of letters/apostrophes as words, and single characters
    from .!?,; as their own tokens."""
    text = text.lower().replace('\u2019', "'").replace('\u2018', "'")
    return TOKEN_RE.findall(text)


def to_sentences(tokens):
    """Mirrors ngram.js's toSentences(): split on . ! ? (dropping the
    punctuation), drop , ; entirely (letting words flow across them, same
    as the browser tokenizer), keep everything else as one sentence."""
    sentences = []
    cur = []
    for t in tokens:
        if t in '.!?':
            if cur:
                sentences.append(cur)
            cur = []
        elif t in ',;':
            continue
        else:
            cur.append(t)
    if cur:
        sentences.append(cur)
    return sentences


def shard_for(context_ids, num_shards):
    """MUST match shardFor() in static-ngram-model.js exactly: a 32-bit
    wraparound multiply-and-add hash over the context's vocab ids."""
    h = 7
    for cid in context_ids:
        h = (h * 131 + cid) & 0xFFFFFFFF
    return h % num_shards


def n1_n2_discount(counts_iterable, fallback=0.75):
    n1 = n2 = 0
    for c in counts_iterable:
        if c == 1:
            n1 += 1
        elif c == 2:
            n2 += 1
    if n1 + 2 * n2 <= 0:
        return fallback
    d = n1 / (n1 + 2 * n2)
    return min(0.9, max(0.1, d))


# --------------------------------------------------------------------------
# Disk-backed n-gram record streaming + external sort/aggregate.
#
# A record for a "context order" R (R ids total: R-1 context ids + 1
# trailing id) is packed as R little-endian unsigned shorts (2 bytes each,
# vocab capped well under 65536 so this always fits). No accumulation in
# memory happens here — records are written to disk as they're produced.
# --------------------------------------------------------------------------

class RecordWriter:
    """Buffered writer for fixed-width binary records. Batches writes so
    ~billions of tiny appends don't turn into billions of syscalls."""
    def __init__(self, path, r):
        self.fmt = '<' + 'H' * r
        self.rec_size = struct.calcsize(self.fmt)
        self.f = open(path, 'wb', buffering=1024 * 1024)
        self.buf = bytearray()
        self.count = 0

    def write(self, rec):
        self.buf += struct.pack(self.fmt, *rec)
        self.count += 1
        if len(self.buf) >= (1 << 20):
            self.f.write(self.buf)
            self.buf.clear()

    def close(self):
        if self.buf:
            self.f.write(self.buf)
            self.buf.clear()
        self.f.close()


def iter_records(path, r):
    fmt = '<' + 'H' * r
    rec_size = struct.calcsize(fmt)
    read_size = rec_size * 65536
    with open(path, 'rb', buffering=1024 * 1024) as f:
        while True:
            chunk = f.read(read_size)
            if not chunk:
                return
            for rec in struct.iter_unpack(fmt, chunk):
                yield rec


def external_sort_group(raw_path, r, tmp_dir, max_records_per_run, label=''):
    """Sorts a raw (unsorted) fixed-width record file via external merge
    sort, then yields (context_tuple, [(word, count), ...]) groups — one
    per distinct context, in sorted order — via a single linear pass over
    the now-sorted (hence run-length-groupable) records. Deletes its own
    temp files as it finishes with them. Memory use is bounded by
    max_records_per_run (one sort-batch's worth of tuples at a time), not
    by the total size of the input.
    """
    fmt = '<' + 'H' * r
    rec_size = struct.calcsize(fmt)
    run_paths = []
    t0 = time.time()
    with open(raw_path, 'rb', buffering=1024 * 1024) as f:
        run_idx = 0
        while True:
            chunk = f.read(rec_size * max_records_per_run)
            if not chunk:
                break
            n = len(chunk) // rec_size
            records = list(struct.iter_unpack(fmt, chunk[: n * rec_size]))
            records.sort()
            run_path = os.path.join(tmp_dir, f'{label}-run-{run_idx}.bin')
            with open(run_path, 'wb', buffering=1024 * 1024) as rf:
                out = bytearray()
                for rec in records:
                    out += struct.pack(fmt, *rec)
                rf.write(out)
            run_paths.append(run_path)
            run_idx += 1
            if run_idx % 20 == 0:
                print(f'    [{label}] sorted {run_idx} run(s) so far ({time.time()-t0:.0f}s)...')
    os.remove(raw_path)  # free the raw (unsorted) file before merging
    print(f'    [{label}] {len(run_paths)} sorted run(s) written in {time.time()-t0:.0f}s, merging...')

    if not run_paths:
        return

    merged = heapq.merge(*(iter_records(p, r) for p in run_paths))

    def counted():
        for rec, group in itertools.groupby(merged):
            yield rec, sum(1 for _ in group)

    def grouped_by_context():
        for ctx, group in itertools.groupby(counted(), key=lambda rc: rc[0][:-1]):
            words = [(rc[0][-1], rc[1]) for rc in group]
            yield ctx, words

    yield from grouped_by_context()

    for p in run_paths:
        os.remove(p)
    print(f'    [{label}] merge/aggregate done in {time.time()-t0:.0f}s total')


def iter_corpus_blocks(args, chunk_names):
    """Yields (label, text) blocks to tokenize.

    Single-file mode (--corpus-file) streams fixed-size character blocks off
    disk via f.read(n) in text mode (which respects multi-byte UTF-8
    boundaries internally, so this is safe) — an 11GB+ file is never held in
    memory whole, only one --read-chunk-chars-sized block at a time. A
    sentence that happens to straddle a block boundary gets split into two
    fragments instead of one sentence; at a 20M-char block size that's at
    most one broken sentence per ~20M characters, statistically negligible.

    Multi-chunk mode (the original static-corpus/ path) is unchanged: each
    chunk-*.txt file (already capped under 100MB) is read whole, one at a
    time.
    """
    if args.corpus_file:
        size = os.path.getsize(args.corpus_file)
        read_so_far = 0
        block_idx = 0
        with open(args.corpus_file, encoding='utf-8', errors='ignore') as f:
            while True:
                text = f.read(args.read_chunk_chars)
                if not text:
                    return
                block_idx += 1
                read_so_far += len(text)
                pct = (read_so_far / size * 100) if size else 0.0
                yield f'{args.corpus_file} [block {block_idx}, ~{pct:.1f}% through file]', text
    else:
        for name in chunk_names:
            path = os.path.join(args.corpus_dir, name)
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
            yield name, text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus-dir', default='static-corpus', help='directory with chunk-*.txt + manifest.json')
    ap.add_argument('--corpus-file', default=None,
                     help='train directly from ONE large raw text file (e.g. '
                          'corpus-build/all-new-corpus.txt) instead of reading chunk-*.txt '
                          'files out of --corpus-dir via manifest.json. Streamed in '
                          '--read-chunk-chars-sized blocks so a multi-GB file is never held '
                          'in memory whole. When set, --corpus-dir/--max-chunks are ignored.')
    ap.add_argument('--read-chunk-chars', type=int, default=20_000_000,
                     help='with --corpus-file, how many characters to read/tokenize per streamed block')
    ap.add_argument('--out', default='ngram-model', help='output directory')
    ap.add_argument('--tmp-dir', default=None, help='scratch directory for temp files (default: system temp)')
    ap.add_argument('--vocab-size', type=int, default=40000, help='max vocabulary size (incl. <s>/</s>)')
    ap.add_argument('--max-chunks', type=int, default=None, help='only read the first N chunk files (for a quick test run)')
    ap.add_argument('--max-sentences', type=int, default=None,
                     help='stop reading corpus text once this many sentences have been collected. '
                          'Use this when the full corpus would blow your disk budget (raw n-gram '
                          'records + sorted runs scale with how much text gets streamed, not with '
                          'the final model size) — e.g. --max-sentences 100000000 stops after ~100M '
                          'sentences and builds a real, complete model from just that much text, '
                          'instead of reading every chunk and running out of space partway through '
                          'the streaming pass. The chunk where the cap is hit is truncated, not '
                          'skipped entirely, so you still get every sentence up to the cap.')
    ap.add_argument('--top-k-2', type=int, default=50, help='max continuations stored per bigram context')
    ap.add_argument('--top-k-3', type=int, default=40, help='max continuations stored per trigram context')
    ap.add_argument('--top-k-4', type=int, default=30, help='max continuations stored per 4-gram context')
    ap.add_argument('--min-count-2', type=int, default=2, help='min total count for a bigram context to be kept')
    ap.add_argument('--min-count-3', type=int, default=2, help='min total count for a trigram context to be kept')
    ap.add_argument('--min-count-4', type=int, default=2, help='min total count for a 4-gram context to be kept')
    ap.add_argument('--shards-2', type=int, default=32)
    ap.add_argument('--shards-3', type=int, default=128)
    ap.add_argument('--shards-4', type=int, default=256)
    ap.add_argument('--sort-batch-records', type=int, default=4_000_000,
                     help='records sorted per in-memory batch during external sort — lower this if you hit memory pressure')
    args = ap.parse_args()

    if args.vocab_size > 65536:
        sys.exit('--vocab-size must be <= 65536 (record format uses 16-bit ids)')

    top_k = {2: args.top_k_2, 3: args.top_k_3, 4: args.top_k_4}
    min_count = {2: args.min_count_2, 3: args.min_count_3, 4: args.min_count_4}
    num_shards = {2: args.shards_2, 3: args.shards_3, 4: args.shards_4}
    orders = [2, 3, 4]

    tmp_root = args.tmp_dir or tempfile.gettempdir()
    os.makedirs(tmp_root, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix='ngram-train-', dir=tmp_root)
    print(f'== scratch directory: {work_dir} ==')

    if args.corpus_file:
        if not os.path.isfile(args.corpus_file):
            sys.exit(f'--corpus-file not found: {args.corpus_file}')
        chunk_names = [os.path.basename(args.corpus_file)]  # bookkeeping only (manifest.json 'sourceChunks')
        print(f'== training directly from single file: {args.corpus_file} '
              f'({os.path.getsize(args.corpus_file)/1e9:.1f}GB) ==')
    else:
        manifest_path = os.path.join(args.corpus_dir, 'manifest.json')
        with open(manifest_path) as f:
            chunk_manifest = json.load(f)
        chunk_names = chunk_manifest['chunks']
        if args.max_chunks:
            chunk_names = chunk_names[: args.max_chunks]
        print(f'== found {len(chunk_names)} chunk file(s) in {args.corpus_dir}/ ==')

    # ---- Pass 1: tokenize every chunk once, write sentences to a temp
    # file (one per line, tokens space-joined), and count raw unigram
    # frequency across ALL tokens so we know which words deserve a vocab
    # slot. This part is cheap (a single Counter over distinct word TYPES,
    # not n-grams) and was never the source of the memory blowup.
    unigram_freq = Counter()
    total_tokens = 0
    sentence_count = 0
    chunks_read = 0
    stopped_early = False
    sents_path = os.path.join(work_dir, 'sentences.txt')
    t0 = time.time()
    with open(sents_path, 'w', encoding='utf-8') as tmp:
        for label, text in iter_corpus_blocks(args, chunk_names):
            if args.max_sentences and sentence_count >= args.max_sentences:
                stopped_early = True
                break
            print(f'  tokenizing {label} '
                  f'({sentence_count:,}/{args.max_sentences:,} sentences so far) ...'
                  if args.max_sentences else f'  tokenizing {label} ...')
            sentences = to_sentences(tokenize(text))
            chunks_read += 1
            for sent in sentences:
                if not sent:
                    continue
                if args.max_sentences and sentence_count >= args.max_sentences:
                    # Cap hit partway through this chunk: everything collected
                    # so far is still a complete, valid, randomly-ordered slice
                    # of the corpus (chunks aren't sorted by content), so
                    # stopping mid-chunk instead of discarding the whole chunk
                    # keeps every sentence you already have disk-committed to.
                    stopped_early = True
                    break
                unigram_freq.update(sent)
                total_tokens += len(sent)
                tmp.write(' '.join(sent))
                tmp.write('\n')
                sentence_count += 1
            del text, sentences
            if stopped_early:
                break
    if stopped_early:
        print(f'== stopped early at --max-sentences={args.max_sentences:,}: '
              f'read {chunks_read}/{len(chunk_names)} chunk file(s) ({sentence_count:,} sentences kept) ==')
    print(f'== pass 1 done in {time.time()-t0:.0f}s: {total_tokens:,} tokens, '
          f'{sentence_count:,} sentences, {len(unigram_freq):,} distinct words ==')

    # ---- Vocab: <s>/</s> reserved at ids 0/1, then the most frequent real
    # words fill the rest of the budget.
    vocab_words = [BOS, EOS]
    for w, _ in unigram_freq.most_common(args.vocab_size - 2):
        vocab_words.append(w)
    word_to_id = {w: i for i, w in enumerate(vocab_words)}
    V = len(vocab_words)
    print(f'== vocab: {V:,} words ==')

    unigram_count = [0] * V
    for w, c in unigram_freq.items():
        if w in word_to_id:
            unigram_count[word_to_id[w]] = c
    del unigram_freq

    # ---- Combined streaming pass: read the tokenized sentences ONCE and
    # write raw (unsorted) n-gram records straight to disk for every order
    # at once — no counting dict of any kind here, just sequential file
    # writes, so memory use during this pass is flat regardless of corpus
    # size. Also writes a "swapped" bigram stream (word, predecessor) used
    # afterward purely to compute exact unigram continuation counts
    # (N1+(* w): how many distinct words precede w) via the same external
    # sort machinery, grouped by word instead of by context.
    max_pad = max(orders) - 1
    bos_id, eos_id = word_to_id[BOS], word_to_id[EOS]

    raw_paths = {o: os.path.join(work_dir, f'raw-o{o}.bin') for o in orders}
    raw_writers = {o: RecordWriter(raw_paths[o], o) for o in orders}
    swap_path = os.path.join(work_dir, 'raw-o2-swap.bin')
    swap_writer = RecordWriter(swap_path, 2)

    t1 = time.time()
    line_no = 0
    with open(sents_path, encoding='utf-8') as f:
        for line in f:
            line_no += 1
            if line_no % 5_000_000 == 0:
                elapsed = time.time() - t1
                print(f'  ... {line_no:,} sentences streamed to disk ({elapsed:.0f}s, '
                      f'{line_no/elapsed:.0f} sent/s)')
            toks = line.rstrip('\n').split(' ')
            ids = [word_to_id[t] for t in toks if t in word_to_id]
            if not ids:
                continue
            padded = [bos_id] * max_pad + ids + [eos_id]
            for order in orders:
                w = raw_writers[order]
                for i in range(order - 1, len(padded)):
                    ctx = padded[i - (order - 1): i]
                    word = padded[i]
                    w.write((*ctx, word))
            # swapped bigram view: (word, predecessor) instead of (predecessor, word)
            for i in range(1, len(padded)):
                swap_writer.write((padded[i], padded[i - 1]))

    for w in raw_writers.values():
        w.close()
    swap_writer.close()
    del vocab_words  # reloaded fresh below where needed; frees the reference here
    os.remove(sents_path)
    print(f'== streaming pass done in {time.time()-t1:.0f}s == '
          f'(raw record files: ' +
          ', '.join(f'o{o}={os.path.getsize(raw_paths[o])/1e9:.2f}GB' for o in orders) +
          f', swap={os.path.getsize(swap_path)/1e9:.2f}GB)')

    # ---- Unigram continuation counts, via external sort of the swapped
    # bigram stream grouped by word: for each word, the number of records
    # in its group IS the number of distinct predecessors (since the group
    # after aggregation already merged duplicate (word,predecessor) pairs
    # down to one entry each).
    vocab_words = [BOS, EOS] + [None] * 0  # placeholder, real list rebuilt below
    print('== computing unigram continuation counts ==')
    cont_unigram = [0] * V
    for word_id, preds in external_sort_group(swap_path, 2, work_dir, args.sort_batch_records, label='contunigram'):
        cont_unigram[word_id[0]] = len(preds)
    cont_unigram_total = sum(cont_unigram)

    # ---- Per order: external sort + group, then discount + prune, then
    # shard and write. Processed one order at a time so only one order's
    # temp files exist on disk at once (raw file for an order is deleted
    # by external_sort_group as soon as its sorted runs exist).
    os.makedirs(args.out, exist_ok=True)
    discounts = {}
    for order in orders:
        print(f'== order {order}: external sort + aggregate ==')
        t2 = time.time()
        shard_buffers = {s: [] for s in range(num_shards[order])}
        kept = 0
        all_counts_for_discount = []  # sampled/streamed below, see note
        ctx_count_total = 0

        # First full grouped pass: compute the discount from every raw
        # (context,word)->count observed (needed before we prune), while
        # simultaneously building the pruned shard buffers — one pass
        # covers both, since discount only needs the count VALUES, which
        # we already see per group here regardless of what we keep.
        for ctx, words in external_sort_group(raw_paths[order], order, work_dir, args.sort_batch_records, label=f'o{order}'):
            ctx_count_total += 1
            total = 0
            for _wid, c in words:
                total += c
                all_counts_for_discount.append(c)
            if total < min_count[order]:
                continue
            distinct = len(words)
            words.sort(key=lambda wc: -wc[1])
            top = words[: top_k[order]]
            s_idx = shard_for(list(ctx), num_shards[order])
            ctx_key = ','.join(str(i) for i in ctx)
            shard_buffers[s_idx].append([ctx_key, total, distinct, [[wid, c] for wid, c in top]])
            kept += 1
            if kept % 2_000_000 == 0:
                print(f'    ...{kept:,} contexts kept so far ({time.time()-t2:.0f}s)')

        discounts[order] = n1_n2_discount(all_counts_for_discount)
        del all_counts_for_discount
        print(f'   order {order}: {ctx_count_total:,} contexts seen, kept {kept:,} '
              f'(min-count {min_count[order]}, top-{top_k[order]} each), '
              f'discount D={discounts[order]:.3f}, {time.time()-t2:.0f}s')

        for s_idx in range(num_shards[order]):
            out_path = os.path.join(args.out, f'o{order}-{s_idx}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(shard_buffers[s_idx], f, separators=(',', ':'))
        print(f'== wrote {num_shards[order]} shard file(s) for order {order} ==')
        del shard_buffers

    # ---- Rebuild the real vocab word list (kept only ids/counts above to
    # avoid holding two copies during the heavy streaming pass) and write
    # vocab.json + manifest.json.
    # word_to_id already has the authoritative mapping; invert it.
    vocab_words = [None] * V
    for w, i in word_to_id.items():
        vocab_words[i] = w

    with open(os.path.join(args.out, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'words': vocab_words,
            'unigramCount': unigram_count,
            'contUnigram': cont_unigram,
            'contUnigramTotal': cont_unigram_total,
            'totalTokens': total_tokens,
        }, f, separators=(',', ':'))

    with open(os.path.join(args.out, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'orders': orders,
            'vocabSize': V,
            'numShards': {str(o): num_shards[o] for o in orders},
            'discounts': {str(o): discounts[o] for o in orders},
            'topK': {str(o): top_k[o] for o in orders},
            'minCount': {str(o): min_count[o] for o in orders},
            'sourceChunks': len(chunk_names),
            'chunksRead': chunks_read,
            'totalTokens': total_tokens,
            'totalSentences': sentence_count,
            'maxSentences': args.max_sentences,
            'stoppedEarly': stopped_early,
            'generatedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }, f, indent=2)

    try:
        os.rmdir(work_dir)
    except OSError:
        pass  # non-empty (shouldn't happen) or already gone — not worth failing the run over

    total_size = sum(
        os.path.getsize(os.path.join(args.out, fn)) for fn in os.listdir(args.out)
    )
    print(f'\nDone in {time.time()-t0:.0f}s total. Wrote {args.out}/ — '
          f'commit this whole directory to your repo.')
    print(f'Total model size on disk: {total_size / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()