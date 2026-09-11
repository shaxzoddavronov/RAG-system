#!/usr/bin/env python3
"""Build the retrieval index. Run on the dev machine; commit the output.

The index is a build-time artifact, not something the PM regenerates. At ~570
chunks the whole thing is a couple of megabytes, so committing it means:

  * the PM never downloads or runs BGE-M3 over the document, only over their
    own one-sentence question;
  * their retrieval results come from exactly the vectors we tested against.

Writes index/faiss.bin, index/bm25/, index/chunks.jsonl and index/manifest.json.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.chunking import build_chunks, chunks_to_dicts
from app.config import settings
from app.normalize import fold


def corpus_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(settings.corpus_path)
    if not chunks:
        print("no chunks produced -- check data/corpus.jsonl", file=sys.stderr)
        return 1
    print(f"chunks        : {len(chunks)}")

    device = settings.device
    print(f"device        : {device}")
    print(f"embedder      : {settings.embed_model}")

    model = SentenceTransformer(settings.embed_model, device=device)
    texts = [c.embed_text for c in chunks]
    vectors = model.encode(
        texts, batch_size=16, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype("float32")
    print(f"vectors       : {vectors.shape}")

    # Cosine similarity == inner product on normalized vectors. Flat = exact,
    # no training, no tuning, and trivially fast at this corpus size.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out / "faiss.bin"))

    # Lexical half. fold() is shared with the query path so the two agree on
    # apostrophes and casing.
    bm25_dir = out / "bm25"
    if bm25_dir.exists():
        shutil.rmtree(bm25_dir)
    tokens = bm25s.tokenize([fold(t) for t in texts], stopwords=None, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    retriever.save(str(bm25_dir))

    with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for record in chunks_to_dicts(chunks):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "dim": int(vectors.shape[1]),
        "chunks": len(chunks),
        "corpus_hash": corpus_digest(settings.corpus_path),
        "built_on_device": device,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"index size    : {total / 1e6:.1f} MB")
    print(f"written to    : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
