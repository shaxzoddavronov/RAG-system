# Running the Qaror 234 RAG system

Step-by-step guide to running this on your own machine. Everything runs
locally — after the one-time downloads in Step 5, the system needs no internet
at all.

If you just want the short version, see [README.md](README.md). This document
covers the full path including the parts that commonly go wrong.

---

## Contents

1. [What you need](#1-what-you-need)
2. [Install Ollama](#2-install-ollama)
3. [Get the code](#3-get-the-code)
4. [Detect your hardware](#4-detect-your-hardware)
5. [Download the models](#5-download-the-models)
6. [Make Ollama reachable from Docker](#6-make-ollama-reachable-from-docker)
7. [Start the API](#7-start-the-api)
8. [Verify it works](#8-verify-it-works)
9. [Using the API](#9-using-the-api)
10. [Sample questions](#10-sample-questions)
11. [Running offline](#11-running-offline)
12. [Configuration reference](#12-configuration-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Stopping and cleaning up](#14-stopping-and-cleaning-up)
15. [For maintainers](#15-for-maintainers-rebuilding-the-corpus-and-index)

---

## 1. What you need

| Requirement | Notes |
|---|---|
| **Docker** | [Docker Desktop](https://docs.docker.com/get-started/get-docker/) on Windows/Mac, Docker Engine on Linux |
| **Ollama** | Normal installer, [ollama.com/download](https://ollama.com/download) |
| **Disk** | ~15 GB (models are the bulk of it) |
| **RAM** | 16 GB comfortable, 8 GB workable with the smallest model |
| **GPU** | **Optional.** Makes it faster, nothing more |

You do **not** need Python, CUDA, `nvidia-container-toolkit`, or a vector
database server. The search index ships prebuilt inside the repository.

---

## 2. Install Ollama

Download and run the installer for your platform from
<https://ollama.com/download>.

Verify:

```bash
ollama --version
```

On Linux without root, you can install into your home directory instead:

```bash
mkdir -p ~/.local/ollama
curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
  -o /tmp/ollama.tar.zst
tar --zstd -xf /tmp/ollama.tar.zst -C ~/.local/ollama
export PATH="$HOME/.local/ollama/bin:$PATH"
ollama serve &
```

---

## 3. Get the code

```bash
git clone https://github.com/shaxzoddavronov/RAG-system.git
cd RAG-system
```

About 12 MB. This includes the resolution as a PDF, the parsed corpus, and the
**prebuilt search index** — which is why you never have to run the embedding
model over the document yourself.

---

## 4. Detect your hardware

```bash
python scripts/setup_profile.py
```

This reads `nvidia-smi` (and copes fine if it is absent), then writes a `.env`
picking a model that will actually fit:

| Detected | Model | Context |
|---|---|---|
| 24 GB+ VRAM | `qwen3:14b` | 16384 |
| 12–16 GB VRAM | `qwen3:8b` | 16384 |
| 8–12 GB VRAM | `qwen3:4b` | 8192 |
| No GPU | `qwen3:4b` | 8192 |
| Apple Silicon | `qwen3:8b` | 8192 |

No Python on your machine? Copy `.env.example` to `.env` and set
`OLLAMA_MODEL` by hand using the table above.

Context length is capped alongside the model deliberately: the KV cache costs
VRAM *on top of* the weights, and that is what usually triggers an
out-of-memory error on a small card — not the model itself.

---

## 5. Download the models

Two downloads, once each.

**The language model** (2.6–9 GB depending on your tier):

```bash
ollama pull qwen3:8b      # or whichever model .env selected
```

**The embedder and reranker** (~4.6 GB total):

```bash
docker compose run --rm api python scripts/warmup.py
```

Run this deliberately rather than letting it happen inside your first API
request, where a long silence is easily mistaken for a crash. It prints
progress and retries on transient network failures.

These land in a Docker named volume, so `docker compose down` does **not**
throw them away.

---

## 6. Make Ollama reachable from Docker

**This is the step people miss.** By default Ollama listens only on
`127.0.0.1`, and a container cannot reach the host's loopback interface. The
confusing result is that `ollama list` works perfectly in your terminal while
the API reports connection refused.

Make Ollama listen on all interfaces:

```bash
# Linux / macOS, running it yourself
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Linux, under systemd
sudo systemctl edit ollama
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama

# macOS desktop app
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"    # then quit and reopen Ollama
```

**Windows**: add a system environment variable `OLLAMA_HOST` =
`0.0.0.0:11434`, then restart Ollama from the system tray.

Confirm a container can actually reach it:

```bash
docker run --rm --add-host host.docker.internal:host-gateway curlimages/curl \
  -s http://host.docker.internal:11434/api/version
```

You want `{"version":"0.x.y"}`. Empty output means Ollama is still bound to
loopback.

> **Security.** Ollama has no authentication, so `0.0.0.0` exposes it to your
> entire local network. On an untrusted network bind only the Docker bridge
> (`OLLAMA_HOST=172.17.0.1:11434` on Linux) or firewall port 11434.

> **Naming trap.** `OLLAMA_HOST` means two different things. For the Ollama
> *server* it is the address to bind. For this API, in `.env`, it is the URL to
> connect to. Unrelated settings that happen to share a name.

---

## 7. Start the API

```bash
docker compose up -d
```

Then open <http://localhost:8000/docs> for the interactive Swagger UI.

Browsing from another machine? Use the server's IP —
`http://<server-ip>:8000/docs`. The API binds `0.0.0.0`.

---

## 8. Verify it works

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

Healthy output:

```json
{
  "status": "ok",
  "device": "cpu",
  "ollama_reachable": true,
  "ollama_model_present": true,
  "embed_model_cached": true,
  "rerank_model_cached": true,
  "index_chunks": 573,
  "corpus_hash_ok": true
}
```

`"device": "cpu"` is expected and correct — the container never uses the GPU.
Ollama does, on the host.

Run the full evaluation suite:

```bash
make verify        # retrieval quality + gate-1 calibration, no Ollama needed
make verify-full   # end-to-end: answer accuracy, refusal accuracy, latency
```

**Allow about 5 minutes.** The container reranks on CPU at roughly 4.8 s per
question, and there are 68 of them. It is working, not hung.

Expected (`qwen3:8b`):

| Metric | Value |
|---|---|
| Retrieval Recall@5 | 1.000 |
| Retrieval MRR | 0.984 |
| Answered correctly | ~0.95 |
| **Refusal accuracy** | **~0.96** |
| Latency p50 / p95 | 1.3 s / 2.9 s |

`make verify-full` fails the run if any single question exceeds the latency
ceiling (45 s by default, `--max-latency` to change), even when the answer was
correct. A question that takes minutes ties up an LLM slot and is
indistinguishable from a hang.

---

## 9. Using the API

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Ask a question |
| `POST /search` | Retrieval only, no LLM — shows what the model would see |
| `GET /nodes/{node_id}` | Raw text of one document node |
| `GET /health` | Device, model, reachability, cache state |

### POST /ask

```bash
curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "Malaka sertifikati necha yil muddatga beriladi?", "stream": false}'
```

```json
{
  "answer": "Malaka sertifikati uch yil muddatga beriladi.",
  "grounded": true,
  "citations": [{
    "node_id": "-8206039",
    "path": "234-qaror › 7-ilova › 6-bob. Malaka imtihoni",
    "url": "https://lex.uz/uz/docs/-8193120#-8206039",
    "snippet": "30. Malaka sertifikati uch yil muddatga beriladi."
  }],
  "quotes": ["Malaka sertifikati uch yil muddatga beriladi."],
  "refusal_reason": null,
  "latency_ms": 2914
}
```

| Field | Meaning |
|---|---|
| `grounded` | `true` = answered from the document, `false` = refused |
| `citations[].url` | Deep link to the exact clause on lex.uz |
| `quotes` | Spans verified to appear **verbatim** in the retrieved text |
| `refusal_reason` | Which gate refused: `retrieval`, `sufficiency`, `quote`, `citation`, `generation` |

When the document does not cover the question, `answer` is always exactly:

```
Hujjatda bu haqida maʼlumot yoʻq.
```

A constant string, so you can test refusals by exact equality.

### POST /search

Same request shape, but no LLM. Returns the five retrieved passages with their
rerank scores — the fastest way to tell whether a bad answer is a retrieval
problem or a generation problem.

---

## 10. Sample questions

Copy into `POST /ask` at `/docs`.

**Deadline (prose)**
```json
{"question": "Jamoat ekologik ekspertizasini tashkil etish toʻgʻrisidagi ariza necha ish kunida koʻrib chiqiladi?", "stream": false}
```
→ *oʻn ish kuni*

**Fee table lookup**
```json
{"question": "Geotermal elektr stansiyalar uchun ekspertiza muddati necha ish kuni va toʻlov miqdori qancha?", "stream": false}
```
→ *15 ish kuni, 3,9 BXM* — proves the 267-row fee table in Appendix 1 is searchable

**Eligibility**
```json
{"question": "Malaka sertifikati olish uchun kamida qancha ish staji talab qilinadi?", "stream": false}
```
→ *kamida uch yil*

**Category count**
```json
{"question": "Ekologik ekspertiza obyektlari necha toifaga boʻlinadi?", "stream": false}
```
→ *uch toifaga*

**Refusal — must NOT answer**
```json
{"question": "Ekologik ekspertiza xulosasiga sudga shikoyat qilish muddati necha kun?", "stream": false}
```
→ `Hujjatda bu haqida maʼlumot yoʻq.` — the resolution sets out procedures and
defers appeals to other legislation

**Refusal — adjacent topic**
```json
{"question": "Ekologik ekspertizani oʻtkazmaganlik uchun jarima qancha?", "stream": false}
```
→ refused. Six clauses discuss liability; none names an amount.

---

## 11. Running offline

After Steps 5 and 6 the system makes no external calls. To prove it:

```bash
docker compose down && docker compose up -d
# disconnect the network, then:
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"Malaka sertifikati necha yil muddatga beriladi?"}'
```

You still get an answer. `docker compose down` preserves the model cache
volume, so restarts never re-download.

---

## 12. Configuration reference

All settings live in `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:8b` | Model tag; must be `ollama pull`ed first |
| `OLLAMA_HOST` | `http://localhost:11434` | Where the API finds Ollama |
| `OLLAMA_NUM_CTX` | `16384` | Context window. Lower this first on OOM |
| `OLLAMA_NUM_PREDICT` | `800` | Max output tokens. Guards against runaway generation |
| `OLLAMA_TEMPERATURE` | `0.0` | Keep at 0 for reproducibility |
| `OLLAMA_SEED` | `42` | Fixed seed |
| `OLLAMA_THINK` | `0` | Reasoning mode. **Leave off** — it costs ~25× latency for no accuracy gain here |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `RERANK_THRESHOLD` | `0.60` | Gate 1. Higher refuses more readily |
| `MAX_CONCURRENT_LLM` | `2` | Parallel generation cap |

---

## 13. Troubleshooting

### `connection refused` to Ollama

Almost always the loopback bind described in [Step 6](#6-make-ollama-reachable-from-docker),
not a stopped service. Check with the container-side `curl` there.

### First request takes minutes

Models are still downloading. Run `make warmup` and watch `/health` —
`embed_model_cached` and `rerank_model_cached` turn `true` when ready.

### Out of memory

Lower `OLLAMA_NUM_CTX` **before** switching to a smaller model. The KV cache
is usually what overflows, not the weights. `16384 → 8192` frees a lot.

### Very slow answers

Expected on CPU: 10–40 s per question. Set `OLLAMA_MODEL=qwen3:4b`.

Confirm `OLLAMA_THINK=0`. With reasoning enabled, Qwen3 spends ~70 s per
question on an L40S generating a chain of thought the system does not use.

### `refusal_reason: "generation"`

Generation was truncated or malformed, usually a runaway that hit
`OLLAMA_NUM_PREDICT`. The system refuses rather than returning something
unverified. Raising the cap is rarely the fix; it normally means the question
has no answer in the document.

### It refuses something that IS in the document

Inspect retrieval first:

```bash
curl -s -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"question":"your question"}' | python3 -m json.tool
```

- **The right passage is not in the results** → a retrieval problem. Try the
  document's own terminology; it is formal legal Uzbek.
- **The right passage is there with a good score** → a generation problem. The
  model likely mistyped its supporting quote, which Gate 3 requires to be
  verbatim. Retries at varying seeds usually recover it.

### API refuses to start: index mismatch

`data/corpus.jsonl` changed without `index/` being rebuilt. Serving that
combination would attach citations to the wrong text, so it is refused
deliberately. Run `make index`, or `git checkout data/ index/`.

---

## 14. Stopping and cleaning up

```bash
docker compose down          # stop the API, keep the model cache
docker compose down -v       # also delete the cache (re-downloads 4.6 GB)
pkill -x ollama              # stop Ollama and its model process
```

Use `pkill -x ollama`, not `pkill -f "ollama serve"` — the `-f` form also
matches the shell you typed it into and will kill that too.

---

## 15. For maintainers: rebuilding the corpus and index

Never needed to *run* the system; the outputs are committed.

```bash
pip install -r requirements-build.txt

make corpus    # re-scrape lex.uz -> qaror_234_2026.pdf + data/corpus.jsonl
make index     # re-chunk and re-embed -> index/
make evalset   # regenerate eval/gold.jsonl
```

Rebuilding the corpus **requires** rebuilding the index. `index/manifest.json`
stores a hash of the corpus it was built from, and the API refuses to start on
a mismatch rather than serve citations pointing at the wrong clause.

`make index` wants a machine with a GPU for reasonable speed, though CPU works
given patience.
