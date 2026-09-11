# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local RAG question-answering API over a single legal document: Uzbekistan
Cabinet of Ministers Resolution No. 234 of 11.05.2026 (environmental impact
assessment). Questions and answers are in Uzbek. Retrieval runs in-process
(FAISS + BM25 + a cross-encoder reranker); generation is delegated to **Ollama
running on the host**, never in the container. Nothing leaves the machine at
query time.

The audience for the deliverable is a non-developer running it on their own
laptop, so *the corpus, the index and the eval set are committed artifacts* —
build scripts exist but are for maintainers only.

## Commands

```bash
make setup          # scripts/setup_profile.py — detects GPU/VRAM, writes .env
make warmup         # download embedder + reranker (~4.6 GB) before demoing
make up / down / logs   # Docker compose
make dev            # native run: uvicorn app.main:app --host 0.0.0.0 --port 8000

make verify         # retrieval metrics + Gate-1 calibration, in the container (no Ollama needed)
make verify-full    # adds end-to-end answer + refusal accuracy (needs Ollama)
make verify-host    # same as verify-full but against a local Python env

make corpus         # maintainers: re-scrape lex.uz -> PDF + data/corpus.jsonl
make index          # maintainers: re-chunk + re-embed -> index/
make evalset        # maintainers: regenerate eval/gold.jsonl
```

There is no unit-test suite. `scripts/evaluate.py` **is** the test harness; it
scores against `eval/gold.jsonl` (67 Uzbek questions: 42 answerable, 25
deliberately on-topic-but-unanswerable). Useful invocations:

```bash
python scripts/evaluate.py                      # retrieval only, deterministic
python scripts/evaluate.py --full               # + generation, needs Ollama
python scripts/evaluate.py --full --max-latency 60
CUDA_VISIBLE_DEVICES="" python scripts/evaluate.py   # force the CPU path
python -m app.chunking                          # chunking stats + sample chunks
```

`--full` fails the run (exit 1) when any question exceeds the latency ceiling,
even if every answer was correct.

Python: the container pins **3.11**; several dependencies have no wheels on
3.13+. The dev host runs 3.14, so native runs need an explicit 3.11/3.12 venv.

## Architecture

Request path (`app/main.py`):

```
question
  -> Retriever.search()        expand_query -> BM25 + FAISS -> RRF -> rerank
  -> Gate 1 (gate_retrieval)   below RERANK_THRESHOLD => refuse, LLM never called
  -> build_context()           small-to-big: retrieve on chunk, send parent band
  -> Generator.generate()      Ollama /api/chat, forced JSON schema
  -> Gates 2-4 (verify_answer) sufficiency, verbatim quotes, valid citations
  -> AskResponse
```

Models load once in the FastAPI lifespan handler. Every synchronous torch call
goes through `run_in_threadpool`; `MAX_CONCURRENT_LLM` bounds generation with a
semaphore.

### The four gates (`app/gates.py`)

All four are plain Python, never model judgement — that is the point, so
behaviour is identical on a 4B and a 32B model. Any failure returns the single
constant `REFUSAL` string from `app/config.py`, so refusals are testable by
equality. **Do not weaken this to fuzzy matching or model-judged grounding.**

Gate 3 (quotes) matches exactly on the `fold()`ed form and deliberately rejects
near-misses: Uzbek negation is the infix `-ma-`, so a two-character edit inverts
a legal obligation. A gate-3 failure earns up to `QUOTE_RETRIES` regenerations
at a **different seed and temperature** — retrying at temperature 0 with a fixed
seed reproduces the same rejected answer.

### Text normalization (`app/normalize.py`)

The corpus builder, the index builder and the query path must all use the same
functions or BM25 silently stops matching. `normalize()` is the display/quote
form (context-correct U+02BB vs U+02BC apostrophes); `fold()` is the aggressive
lexical form. `expand_query()` adds a Cyrillic→Latin transliteration and
GLOSSARY abbreviation expansions. Changes here invalidate the committed index.

### Chunking (`app/chunking.py`)

Structure-aware, not similarity-based. A numbered *band* is the atomic unit;
Appendix 1's 267-row table is verbalized one sentence per row. `chunk_id`
(`<node_id>#<n>` or `<node_id>#r<band>`) is what the LLM cites — several chunks
can share a `node_id`, so citations must never be keyed on `node_id` alone.
`node_id` is still what the lex.uz anchor URL needs, so both travel together.

### Index integrity

`index/manifest.json` stores a SHA-256 prefix of `data/corpus.jsonl`, computed
by `corpus_digest()` in `app/retrieval.py` (imported by `scripts/build_index.py`
so the builder and the startup check can never drift). It normalizes CRLF→LF
first: Windows clones with `core.autocrlf=true` rewrite line endings and
otherwise make a healthy clone fail startup. `.gitattributes` marks the
committed artifacts `-text`/`binary` so git stops rewriting them at all. On
startup `Retriever._verify_corpus()` raises `IndexMismatch` and the API refuses
to start if they disagree — a stale index means citations point at the wrong
text. **Any change to the corpus or to normalization/chunking requires
`make index` and committing the result.**

### Configuration

All hardware-dependent resolution happens once in `app/config.py`. Nothing else
may call `.cuda()` or assume a GPU exists. `.env` (written by
`scripts/setup_profile.py`) is the only config surface; real environment
variables win over it. Notable settings:

- `OLLAMA_NUM_CTX` — always sent explicitly; Ollama's default silently
  truncates the prompt and the model then answers from memory.
- `OLLAMA_NUM_PREDICT` (800) — caps runaway generation.
- `OLLAMA_THINK` (off) — reasoning mode costs ~25× latency for no accuracy gain
  here, since answers are extracted from retrieved text, not derived.
- `RERANK_THRESHOLD` (0.60) — Gate 1. The reranker saturates (relevant ≈0.73,
  irrelevant ≈0.51), so this sits in the empty space between clusters and is
  deliberately *not* tuned to the calibrator's best-on-paper value, which would
  flip on float noise. On-topic-but-unanswerable questions are *expected* to
  pass Gate 1 and be caught by gates 2–4.

### Logging (`app/logging_setup.py`)

`configure()` is called once at import in `main.py`. Every `/ask` logs one line
per stage tagged with a 4-char request id, since `MAX_CONCURRENT_LLM` plus
threadpooled retrieval means requests interleave. Refusals log *which gate*
fired via `GATE_EXPLANATION`, so a refusal is never anonymous. `LOG_LEVEL=DEBUG`
adds query variants and per-hit rerank scores.

The access-log filter reads uvicorn's structured `record.args`, not the rendered
line: uvicorn appends the status phrase in its formatter, so at filter time the
message still ends at the bare status number and a text match on `" 200 "`
silently never fires. HF progress bars are disabled in `_silence_progress_bars()`
— they render as hundreds of carriage-returned frames in `docker compose logs`.

`OLLAMA_HOST` is confusingly overloaded: for the Ollama server it is a bind
address, for this API it is a connect URL. Docker compose overrides it to
`http://host.docker.internal:11434`.

## Docs

`README.md` is the overview and honest-limitations write-up; `RUNNING.md` is the
full end-user guide (install, offline use, config reference, troubleshooting).
Both describe measured behaviour — update them when numbers or behaviour change.
