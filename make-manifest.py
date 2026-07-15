#!/usr/bin/env python3
# Run from your repo root: python3 make-manifest.py
# Scans static-corpus/ for chunk-*.txt files and writes manifest.json so
# static-corpus.js doesn't have to guess filenames/counts.
import json, os

d = 'static-corpus'
chunks = sorted(f for f in os.listdir(d) if f.startswith('chunk-') and f.endswith('.txt'))
if not chunks:
    raise SystemExit(f'No chunk-*.txt files found in {d}/')

manifest = {'chunks': chunks, 'count': len(chunks)}
with open(os.path.join(d, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'Wrote {d}/manifest.json with {len(chunks)} chunks:')
for c in chunks:
    print(' ', c)
