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
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.config import REFUSAL, settings
from app.gates import Verdict, gate_retrieval, resolve_citation, verify_answer
from app.logging_setup import configure as configure_logging
from app.generate import Generator, OllamaError, OllamaOutputError
from app.retrieval import IndexMismatch, Retriever
from app.schemas import (
    AskRequest, AskResponse, Citation, HealthResponse, SearchHit, SearchResponse,
)

QUOTE_RETRIES = 2

state: dict = {}
log = configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    boot = time.perf_counter()
    try:
        retriever = Retriever(settings)
    except (FileNotFoundError, IndexMismatch) as exc:
        # This is the failure people actually hit (a stale or rewritten index),
        # so say so plainly before the traceback buries it.
        log.error("startup failed: %s", exc)
        raise RuntimeError(str(exc)) from exc

    log.info("index    | %d chunks, corpus verified, embed=%s rerank=%s",
             len(retriever.chunks), settings.embed_model, settings.rerank_model)
    log.info("device   | %s  (retrieval only; generation runs in Ollama)",
             settings.device)

    # Touch both models now so the first user request is not the one paying the
    # load cost (or discovering the weights were never downloaded).
    loading = time.perf_counter()
    await run_in_threadpool(lambda: retriever.embedder)
    await run_in_threadpool(lambda: retriever.reranker)
    log.info("models   | loaded in %.1fs", time.perf_counter() - loading)

    state["retriever"] = retriever
    state["generator"] = Generator(settings)
    state["llm_semaphore"] = asyncio.Semaphore(settings.max_concurrent_llm)
    state["corpus"] = {
        rec["node_id"]: rec
        for rec in (json.loads(l) for l in settings.corpus_path.open(encoding="utf-8"))
    }
    reachable, present = await state["generator"].reachable()
    log.info("ollama   | %s model=%s reachable=%s model_present=%s",
             settings.ollama_host, settings.ollama_model, reachable, present)
    if not reachable:
        log.warning("ollama   | unreachable -- /ask will return 503. "
                    "/search and retrieval still work.")
    elif not present:
        log.warning("ollama   | model %s not pulled -- run: ollama pull %s",
                    settings.ollama_model, settings.ollama_model)
    log.info("ready    | startup took %.1fs, gate-1 threshold %.2f",
             time.perf_counter() - boot, settings.rerank_threshold)
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


# Which gate produced a refusal, phrased for someone reading the log rather
# than someone reading gates.py.
GATE_EXPLANATION = {
    "retrieval": "gate 1: nothing retrieved above the threshold (LLM not called)",
    "sufficiency": "gate 2: model reported the passages do not answer this",
    "quote": "gate 3: a supporting quote was not verbatim in the retrieved text",
    "citation": "gate 4: cited an id that was not retrieved",
    "generation": "model returned unusable JSON (likely truncated at num_predict)",
}


def _refuse(reason: str, started: float, rid: str = "----") -> AskResponse:
    elapsed = int((time.perf_counter() - started) * 1000)
    log.info("[%s] REFUSED  %s  %dms", rid, GATE_EXPLANATION.get(reason, reason), elapsed)
    return AskResponse(
        answer=REFUSAL, grounded=False, citations=[], quotes=[],
        refusal_reason=reason, latency_ms=elapsed,
    )


async def _answer(question: str) -> AskResponse:
    started = time.perf_counter()
    rid = uuid.uuid4().hex[:4]
    retriever: Retriever = state["retriever"]
    generator: Generator = state["generator"]

    log.info("[%s] ask      %r", rid, question[:100])

    stage = time.perf_counter()
    hits, variants = await run_in_threadpool(retriever.search, question)
    retrieval_ms = int((time.perf_counter() - stage) * 1000)
    top = hits[0].rerank_score if hits else 0.0
    log.info("[%s] retrieve %d hits, top score %.3f (threshold %.2f)  %dms",
             rid, len(hits), top, settings.rerank_threshold, retrieval_ms)
    if len(variants) > 1:
        log.debug("[%s] variants %s", rid, variants)
    for hit in hits:
        log.debug("[%s]   %.3f  %s  %s",
                  rid, hit.rerank_score, hit.chunk["chunk_id"], hit.chunk["path"][:70])

    # Gate 1: refuse before the model is ever called.
    if not gate_retrieval(hits, settings.rerank_threshold).ok:
        return _refuse("retrieval", started, rid)

    blocks = retriever.build_context(hits)
    log.debug("[%s] context  %d blocks, %d chars sent to the model",
              rid, len(blocks), sum(len(b["text"]) for b in blocks))

    async with state["llm_semaphore"]:
        stage = time.perf_counter()
        try:
            payload = await generator.generate(question, blocks)
        except OllamaOutputError as exc:
            # Truncated or malformed output -- nothing verifiable came back.
            log.warning("[%s] ollama   %s", rid, exc)
            return _refuse("generation", started, rid)
        except OllamaError as exc:
            log.error("[%s] ollama   unreachable: %s", rid, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        log.info("[%s] generate sufficient=%s quotes=%d citations=%d  %dms",
                 rid, payload["sufficient"], len(payload["quotes"]),
                 len(payload["citations"]), int((time.perf_counter() - stage) * 1000))

        verdict = verify_answer(payload, blocks)
        if not verdict.ok:
            log.info("[%s] gates    failed on %s", rid, verdict.reason)

        # Retries on a bad quote. Each must vary the sampling -- with a fixed
        # seed at temperature 0 a retry deterministically reproduces the same
        # rejected answer, which is what this used to do. Observed in practice:
        # the model transcribes a long sentence with one morpheme wrong, and
        # succeeds on a later attempt by quoting a shorter span.
        for attempt in range(1, QUOTE_RETRIES + 1):
            if verdict.ok or verdict.reason != "quote":
                break
            log.info("[%s] retry    %d/%d at seed %d, temp 0.2 (quote mismatch)",
                     rid, attempt, QUOTE_RETRIES, settings.seed + attempt)
            try:
                payload = await generator.generate(
                    question, blocks, temperature=0.2,
                    seed=settings.seed + attempt)
                verdict = verify_answer(payload, blocks)
                if verdict.ok:
                    log.info("[%s] retry    recovered on attempt %d", rid, attempt)
            except OllamaOutputError:
                verdict = Verdict(False, "generation")
                break
            except OllamaError:
                break

    if not verdict.ok:
        return _refuse(verdict.reason or "unknown", started, rid)

    # Resolve through the same function gate 4 used, or a citation it accepted
    # as a bare node_id would silently vanish from the response.
    by_id = {b["cite_id"]: b for b in blocks}
    resolved = dict.fromkeys(
        cid for cid in (resolve_citation(c, blocks) for c in payload["citations"])
        if cid is not None
    )
    citations = [
        Citation(node_id=by_id[cid]["node_id"], path=by_id[cid]["path"],
                 url=by_id[cid]["url"], snippet=by_id[cid]["text"][:300])
        for cid in resolved
    ]
    elapsed = int((time.perf_counter() - started) * 1000)
    log.info("[%s] ANSWERED grounded, %d citation(s): %s  %dms",
             rid, len(citations), ", ".join(c.node_id for c in citations), elapsed)
    return AskResponse(
        answer=payload["answer"], grounded=True, citations=citations,
        quotes=payload["quotes"], latency_ms=elapsed,
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
    log.info("search   %r -> %d hits, top %.3f  %dms",
             req.question[:80], len(hits),
             hits[0].rerank_score if hits else 0.0,
             int((time.perf_counter() - started) * 1000))
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
