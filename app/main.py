"""FastAPI service.

Retrieval and reranking are synchronous torch calls, so they go through
run_in_threadpool -- otherwise every request would block the event loop and the
"asynchronous processing" requirement would be decorative. Models are loaded
once in the lifespan handler, never per request.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.config import REFUSAL, settings
from app.gates import Verdict, gate_retrieval, verify_answer
from app.generate import Generator, OllamaError, OllamaOutputError
from app.retrieval import IndexMismatch, Retriever
from app.schemas import (
    AskRequest, AskResponse, Citation, HealthResponse, SearchHit, SearchResponse,
)

QUOTE_RETRIES = 2

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        retriever = Retriever(settings)
    except (FileNotFoundError, IndexMismatch) as exc:
        raise RuntimeError(str(exc)) from exc

    # Touch both models now so the first user request is not the one paying the
    # load cost (or discovering the weights were never downloaded).
    await run_in_threadpool(lambda: retriever.embedder)
    await run_in_threadpool(lambda: retriever.reranker)

    state["retriever"] = retriever
    state["generator"] = Generator(settings)
    state["llm_semaphore"] = asyncio.Semaphore(settings.max_concurrent_llm)
    state["corpus"] = {
        rec["node_id"]: rec
        for rec in (json.loads(l) for l in settings.corpus_path.open(encoding="utf-8"))
    }
    try:
        yield
    finally:
        await state["generator"].aclose()


app = FastAPI(
    title="Qaror 234 RAG API",
    description="Vazirlar Mahkamasining 11.05.2026 yildagi 234-son qarori boʻyicha "
                "savol-javob. Javoblar faqat shu hujjat matniga asoslanadi.",
    version="1.0.0",
    lifespan=lifespan,
)


def _refuse(reason: str, started: float) -> AskResponse:
    return AskResponse(
        answer=REFUSAL, grounded=False, citations=[], quotes=[],
        refusal_reason=reason, latency_ms=int((time.perf_counter() - started) * 1000),
    )


async def _answer(question: str) -> AskResponse:
    started = time.perf_counter()
    retriever: Retriever = state["retriever"]
    generator: Generator = state["generator"]

    hits, _ = await run_in_threadpool(retriever.search, question)

    # Gate 1: refuse before the model is ever called.
    if not gate_retrieval(hits, settings.rerank_threshold).ok:
        return _refuse("retrieval", started)

    blocks = retriever.build_context(hits)

    async with state["llm_semaphore"]:
        try:
            payload = await generator.generate(question, blocks)
        except OllamaOutputError:
            # Truncated or malformed output -- nothing verifiable came back.
            return _refuse("generation", started)
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        verdict = verify_answer(payload, blocks)

        # Retries on a bad quote. Each must vary the sampling -- with a fixed
        # seed at temperature 0 a retry deterministically reproduces the same
        # rejected answer, which is what this used to do. Observed in practice:
        # the model transcribes a long sentence with one morpheme wrong, and
        # succeeds on a later attempt by quoting a shorter span.
        for attempt in range(1, QUOTE_RETRIES + 1):
            if verdict.ok or verdict.reason != "quote":
                break
            try:
                payload = await generator.generate(
                    question, blocks, temperature=0.2,
                    seed=settings.seed + attempt)
                verdict = verify_answer(payload, blocks)
            except OllamaOutputError:
                verdict = Verdict(False, "generation")
                break
            except OllamaError:
                break

    if not verdict.ok:
        return _refuse(verdict.reason or "unknown", started)

    by_id = {b["cite_id"]: b for b in blocks}
    citations = [
        Citation(node_id=by_id[cid]["node_id"], path=by_id[cid]["path"],
                 url=by_id[cid]["url"], snippet=by_id[cid]["text"][:300])
        for cid in dict.fromkeys(payload["citations"]) if cid in by_id
    ]
    return AskResponse(
        answer=payload["answer"], grounded=True, citations=citations,
        quotes=payload["quotes"],
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.stream:
        return await _answer(req.question)

    # Streaming still verifies the whole answer before emitting a single token.
    # Streaming an unverified answer and retracting it afterwards would defeat
    # the point of the gates.
    result = await _answer(req.question)

    async def events():
        for word in result.answer.split(" "):
            yield f"data: {json.dumps({'token': word + ' '}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
        tail = result.model_dump(exclude={"answer"})
        yield f"event: done\ndata: {json.dumps(tail, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/search", response_model=SearchResponse)
async def search(req: AskRequest):
    """Retrieval only -- no LLM. Use this to inspect what the model would see."""
    started = time.perf_counter()
    retriever: Retriever = state["retriever"]
    hits, variants = await run_in_threadpool(retriever.search, req.question)
    return SearchResponse(
        query=req.question, variants=variants,
        hits=[SearchHit(node_id=h.chunk["node_id"], path=h.chunk["path"],
                        url=h.chunk["url"], text=h.chunk["text"],
                        rerank_score=h.rerank_score, rrf_score=h.rrf_score)
              for h in hits],
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


@app.get("/nodes/{node_id}", response_model=dict)
async def node(node_id: str):
    rec = state["corpus"].get(node_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
    return rec


@app.get("/health", response_model=HealthResponse)
async def health():
    retriever: Retriever = state["retriever"]
    reachable, model_present = await state["generator"].reachable()
    return HealthResponse(
        status="ok" if reachable and model_present else "degraded",
        device=settings.device,
        ollama_host=settings.ollama_host,
        ollama_model=settings.ollama_model,
        ollama_reachable=reachable,
        ollama_model_present=model_present,
        embed_model_cached=retriever._embedder is not None,
        rerank_model_cached=retriever._reranker is not None,
        index_chunks=len(retriever.chunks),
        corpus_hash_ok=True,
    )
