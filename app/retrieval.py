"""Hybrid retrieval: BM25 + dense, fused with RRF, then cross-encoder reranked.

Uzbek is agglutinative and under-represented in embedding training data, so
dense retrieval alone misses exact terms, band numbers and fees. BM25 catches
those; the reranker then fixes the ordering. Each stage is cheap at this corpus
size and the combination is what makes the refusal threshold trustworthy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import faiss
import numpy as np

from app.config import Settings, settings as default_settings
from app.normalize import expand_query, fold, has_cyrillic, normalize, to_latin


@dataclass
class Hit:
    chunk: dict
    rrf_score: float
    rerank_score: float = 0.0

    @property
    def node_id(self) -> str:
        return self.chunk["node_id"]


class IndexMismatch(RuntimeError):
    """The committed index does not match the corpus it is being served with."""


class Retriever:
    def __init__(self, settings: Settings = default_settings, verify_corpus: bool = True):
        self.settings = settings
        index_dir: Path = settings.index_dir

        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run scripts/build_index.py first.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if verify_corpus:
            self._verify_corpus()

        self.chunks: list[dict] = [
            json.loads(line)
            for line in (index_dir / "chunks.jsonl").open(encoding="utf-8")
        ]
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.bin"))
        self.bm25 = bm25s.BM25.load(str(index_dir / "bm25"), load_corpus=False)

        self._embedder = None
        self._reranker = None

    # -- lazy model loading; both honour the resolved device ------------------

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.settings.embed_model, device=self.settings.device)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                self.settings.rerank_model, device=self.settings.device)
        return self._reranker

    def _verify_corpus(self) -> None:
        """Stale index means citations point at the wrong text -- refuse to serve."""
        import hashlib
        path = self.settings.corpus_path
        if not path.exists():
            return
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        expected = self.manifest.get("corpus_hash")
        if expected and digest != expected:
            raise IndexMismatch(
                f"index was built from corpus {expected}, but data/corpus.jsonl is "
                f"{digest}. Re-run scripts/build_index.py.")

    # -- retrieval ------------------------------------------------------------

    def _dense(self, variants: list[str]) -> list[list[int]]:
        vectors = self.embedder.encode(
            variants, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False).astype("float32")
        k = min(self.settings.top_k_dense, len(self.chunks))
        _, ids = self.faiss_index.search(vectors, k)
        return [[int(i) for i in row if i >= 0] for row in ids]

    def _lexical(self, variants: list[str]) -> list[list[int]]:
        k = min(self.settings.top_k_lexical, len(self.chunks))
        out = []
        for variant in variants:
            tokens = bm25s.tokenize(fold(variant), stopwords=None, show_progress=False)
            try:
                ids, _ = self.bm25.retrieve(tokens, k=k, show_progress=False)
            except Exception:
                out.append([])
                continue
            out.append([int(i) for i in np.asarray(ids).ravel()])
        return out

    def _rrf(self, rankings: list[list[int]]) -> dict[int, float]:
        """Reciprocal rank fusion -- rank-based, so the two score scales never meet."""
        k = self.settings.rrf_k
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def search(self, query: str, top_k: int | None = None) -> tuple[list[Hit], list[str]]:
        top_k = top_k or self.settings.top_k_rerank
        variants = expand_query(query)

        fused = self._rrf(self._dense(variants) + self._lexical(variants))
        if not fused:
            return [], variants

        candidates = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        candidates = candidates[: max(self.settings.top_k_dense, self.settings.top_k_lexical)]
        hits = [Hit(chunk=self.chunks[i], rrf_score=s) for i, s in candidates]

        # The cross-encoder is multilingual, but giving it Latin when the user
        # typed Cyrillic removes a script gap it would otherwise have to bridge.
        rerank_query = to_latin(query) if has_cyrillic(query) else normalize(query)
        pairs = [(rerank_query, h.chunk["text"]) for h in hits]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype="float64")))  # -> 0..1
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)

        hits.sort(key=lambda h: h.rerank_score, reverse=True)
        return hits[:top_k], variants

    @staticmethod
    def build_context(hits: list[Hit]) -> list[dict]:
        """Small-to-big: retrieve on the chunk, hand the LLM the whole parent band.

        Citations key on chunk_id, not node_id. Every row of the Appendix 1
        table lives in one document node, so several blocks can share a
        node_id -- labelling them identically leaves the model unable to cite
        one unambiguously, and collapsing them by node_id would attach the
        wrong row's text to a citation. node_id is still what the lex.uz
        anchor needs, so both travel together.
        """
        seen: set[str] = set()
        blocks = []
        for hit in hits:
            key = hit.chunk.get("context_text") or hit.chunk["text"]
            if key in seen:
                continue
            seen.add(key)
            blocks.append({
                "cite_id": hit.chunk["chunk_id"],
                "node_id": hit.chunk["node_id"],
                "path": hit.chunk["path"],
                "url": hit.chunk["url"],
                "text": key,
                "score": hit.rerank_score,
            })
        return blocks
