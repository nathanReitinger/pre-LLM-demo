#!/usr/bin/env bash
# grow-corpus.sh — pulls in more real text and chunks it into static-corpus/
# in the exact format static-corpus.js / make-manifest.py already expect:
# chunk-NNNN.txt files, each <=90MB, named to continue after whatever's
# already there.
#
# Run this from your repo root (the directory that already contains
# static-corpus/ and make-manifest.py).
#
# Requires: curl, unzip, rsync (all normally preinstalled on Linux/macOS),
# python3.

set -euo pipefail

# Fails with a clear message instead of unzip's cryptic "End-of-central-
# directory signature not found" when a URL 403s/404s and curl silently
# saves the error page instead of the real file.
require_valid_zip() {
  if ! unzip -tq "$1" >/dev/null 2>&1; then
    echo "ERROR: $1 is not a valid zip (the download likely failed/403'd)." >&2
    echo "  First 200 bytes of what was actually downloaded:" >&2
    head -c 200 "$1" >&2; echo >&2
    exit 1
  fi
}

OUTDIR="static-corpus"
BUILD="corpus-build"
CHUNK_SIZE="90m"

mkdir -p "$OUTDIR" "$BUILD"

# ---------------------------------------------------------------------------
# Sources. Pick/comment out whichever you want. All are public-domain or
# CC-BY-SA, no login/API-key required, and known-stable download URLs (as
# opposed to scraping a site directly, which can get you rate-limited or
# blocked).
# ---------------------------------------------------------------------------

echo "== WikiText-103 (raw) — ~545MB, clean Wikipedia prose, keeps punctuation =="
# The original S3 URL (s3.amazonaws.com/research.metamind.io/...) has been
# dead/403 since 2024 — Salesforce pulled it off S3 (see
# https://github.com/huggingface/tokenizers/issues/1683). This pulls the
# same raw files from a maintained Hugging Face re-host instead
# (huggingface.co/datasets/segyges/wikitext-103), which serves plain files
# over HTTP with no login/token needed for a public dataset repo.
# -C - resumes a partial download instead of restarting from zero, and is a
# no-op (near-instant) if the file's already fully there.
if [ ! -f "$BUILD/wikitext103.txt" ]; then
  BASE="https://huggingface.co/datasets/segyges/wikitext-103/resolve/main"
  curl -L --fail -C - -o "$BUILD/wiki.train.raw" "$BASE/wiki.train.raw"
  curl -L --fail -C - -o "$BUILD/wiki.valid.raw" "$BASE/wiki.valid.raw"
  curl -L --fail -C - -o "$BUILD/wiki.test.raw"  "$BASE/wiki.test.raw"
  cat "$BUILD/wiki.train.raw" "$BUILD/wiki.valid.raw" "$BUILD/wiki.test.raw" > "$BUILD/wikitext103.txt"
fi

echo "== enwik9 — ~1GB Wikipedia XML dump sample, different register than cleaned Wikipedia =="
if [ ! -f "$BUILD/enwik9" ]; then
  curl -L --fail -C - -o "$BUILD/enwik9.zip" http://mattmahoney.net/dc/enwik9.zip
  require_valid_zip "$BUILD/enwik9.zip"
  unzip -o "$BUILD/enwik9.zip" -d "$BUILD"
fi

echo "== Project Gutenberg — public-domain books, genuinely different prose (fiction, not encyclopedic) =="
# IMPORTANT: don't hit www.gutenberg.org directly in a loop — their robot
# policy asks you to use an official rsync mirror instead, and will start
# blocking an IP that scrapes the main site in bulk. This pulls only the
# plain-text (.txt) files from ONE digit-shard of their mirror tree, and
# --max-size skips oversized outliers so this stays bounded. Still sizable —
# check du -sh after before committing. See https://www.gutenberg.org/policy/robot_access.html
#
# NOTE: this block is intentionally run every time, NOT gated on
# `[ ! -d gutenberg ]` — rsync is already incremental (it diffs against what's
# on disk and only transfers what's missing/changed), so re-running this is
# cheap and safe, and it's the only way an interrupted first run actually
# finishes. A directory-existence check would be wrong here: the directory
# gets created before rsync starts, so a run that dies partway through would
# look "done" on the next invocation and silently skip the rest — which is
# exactly what happened with the "Can't assign requested address" error.
# That error is a transient local networking hiccup, not a problem with the
# mirror, so this retries a few times with a short pause before giving up.
mkdir -p "$BUILD/gutenberg"
GUTENBERG_ATTEMPTS=5
for attempt in $(seq 1 "$GUTENBERG_ATTEMPTS"); do
  if rsync -av --partial --timeout=60 \
      --include "*/" --include "*.txt" --exclude "*" \
      --max-size=5m \
      rsync://aleph.gutenberg.org/gutenberg/1/ "$BUILD/gutenberg/"; then
    break
  fi
  echo "rsync attempt $attempt/$GUTENBERG_ATTEMPTS failed, retrying in 10s..." >&2
  if [ "$attempt" -eq "$GUTENBERG_ATTEMPTS" ]; then
    echo "WARNING: giving up on Gutenberg after $GUTENBERG_ATTEMPTS attempts; continuing with whatever downloaded so far." >&2
  fi
  sleep 10
done
find "$BUILD/gutenberg" -name '*.txt' -exec cat {} + > "$BUILD/gutenberg-books.txt" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Combine everything you fetched above into one blob. Comment out any line
# for a source you skipped.
# ---------------------------------------------------------------------------
echo "== combining into one blob =="
: > "$BUILD/all-new-corpus.txt"
[ -f "$BUILD/wikitext103.txt" ]      && cat "$BUILD/wikitext103.txt"      >> "$BUILD/all-new-corpus.txt"
[ -f "$BUILD/enwik9" ]               && cat "$BUILD/enwik9"               >> "$BUILD/all-new-corpus.txt"
[ -f "$BUILD/gutenberg-books.txt" ]  && cat "$BUILD/gutenberg-books.txt"  >> "$BUILD/all-new-corpus.txt"

echo "Combined size: $(du -h "$BUILD/all-new-corpus.txt" | cut -f1)"

# ---------------------------------------------------------------------------
# Chunk into 90MB files, continuing the numbering from whatever's already
# in static-corpus/ (e.g. if chunk-0011.txt is the last one, this starts
# at chunk-0012.txt).
# ---------------------------------------------------------------------------
LAST=$(ls "$OUTDIR"/chunk-*.txt 2>/dev/null | sed -E 's/.*chunk-([0-9]+)\.txt/\1/' | sort -n | tail -1 || true)
LAST=${LAST:-0}
START=$((10#$LAST + 1))
echo "== chunking, starting at chunk-$(printf '%04d' "$START").txt =="

# Prefer GNU split (supports --numeric-suffixes=START directly) if present —
# e.g. `gsplit` from `brew install coreutils` on macOS, or plain `split` on
# most Linux distros. Otherwise fall back to BSD split, which can only ever
# start numbering at 0000 (it has no long-option syntax at all — passing it
# `--numeric-suffixes=N` is what caused "illegal option -- -"), so in that
# case we split into a scratch dir and rename the parts up by the existing
# offset afterward.
if command -v gsplit >/dev/null 2>&1; then
  SPLIT_BIN="gsplit"
elif split --version >/dev/null 2>&1; then
  # GNU split reports a --version; BSD split errors out on unknown option.
  SPLIT_BIN="split"
else
  SPLIT_BIN=""
fi

if [ -n "$SPLIT_BIN" ]; then
  "$SPLIT_BIN" -b "$CHUNK_SIZE" -d -a 4 --numeric-suffixes="$START" \
    "$BUILD/all-new-corpus.txt" "$OUTDIR/chunk-"
  # split's numeric suffixes come out as plain digits (chunk-0012); add .txt
  for f in "$OUTDIR"/chunk-[0-9][0-9][0-9][0-9]; do
    [ -e "$f" ] && mv "$f" "$f.txt"
  done
else
  echo "  (no GNU split found — falling back to BSD split + manual renumbering)" >&2
  echo "  (tip: 'brew install coreutils' gives you gsplit, which is faster on repeat runs)" >&2
  SCRATCH="$BUILD/split-scratch"
  rm -rf "$SCRATCH"
  mkdir -p "$SCRATCH"
  # BSD split always starts numbering at 0000, no way to override.
  split -b "$CHUNK_SIZE" -d -a 4 "$BUILD/all-new-corpus.txt" "$SCRATCH/part-"
  for f in "$SCRATCH"/part-[0-9][0-9][0-9][0-9]; do
    [ -e "$f" ] || continue
    idx=$(basename "$f" | sed -E 's/part-([0-9]+)/\1/')
    newnum=$((10#$idx + START))
    mv "$f" "$OUTDIR/chunk-$(printf '%04d' "$newnum").txt"
  done
  rm -rf "$SCRATCH"
fi

echo "== regenerating manifest.json =="
python3 make-manifest.py

echo "Done. New total size of $OUTDIR:"
du -sh "$OUTDIR"