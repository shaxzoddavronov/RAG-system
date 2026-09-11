# Qaror 234 RAG API

Savol-javob API over **Cabinet of Ministers Resolution No. 234 of 11.05.2026**
("Atrof-muhitga taʼsirni baholashning yangi mexanizmlarini joriy qilish
chora-tadbirlari toʻgʻrisida").

Answers come only from the text of that resolution. When the document does not
cover something, the API returns exactly:

```
Hujjatda bu haqida maʼlumot yoʻq.
```

Runs entirely on your own machine. Nothing is sent to any external service at
query time.

---

> **Full step-by-step guide: [RUNNING.md](RUNNING.md)** — installation,
> configuration, offline operation, sample questions and troubleshooting.

## Quick start

You need [Docker](https://docs.docker.com/get-started/get-docker/) and
[Ollama](https://ollama.com/download) (a normal installer on Windows/Mac/Linux).

```bash
git clone <repo> && cd rag-system

python scripts/setup_profile.py     # detects your GPU, writes .env
ollama pull $(grep OLLAMA_MODEL .env | cut -d= -f2)
docker compose up -d
```

Open <http://localhost:8000/docs>.

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"Malaka sertifikati necha yil muddatga beriladi?"}'
```

## Does it need a GPU?

No. A GPU makes it faster, nothing more. `setup_profile.py` picks a model that
fits whatever you have:

| Your hardware | Model chosen | Answer latency |
|---|---|---|
| RTX 4090 / 5090 (24 GB+) | `qwen3:14b` | ~2 s |
| RTX 4070 / 4080 / 3060-12G | `qwen3:8b` | ~3 s |
| RTX 4060 / 3070 (8 GB) | `qwen3:4b` | ~4 s |
| No NVIDIA GPU | `qwen3:4b` on CPU | 10–40 s |
| Apple Silicon Mac | `qwen3:8b` (Metal) | ~4 s |

Ollama detects your GPU by itself and falls back to CPU when there is none.
The API container is CPU-only on purpose, so there is no
`nvidia-container-toolkit` to set up on any platform.

**Retrieval quality does not change with hardware.** The search index is
prebuilt and committed, so the passages found on a laptop are the same ones
found on the development server. Only the wording of the final answer, and the
speed, depend on which model you run.

## Downloads and offline use

Three one-off downloads, then nothing:

| When | What | Size |
|---|---|---|
| `git clone` | code, document, prebuilt index | ~12 MB |
| `ollama pull` | the language model | 2.6–9 GB |
| first start (or `make warmup`) | embedder + reranker | ~4.6 GB |

Run `make warmup` before demoing so the model download is not mistaken for a
hang. After that the system works with the network unplugged. `docker compose
down` keeps the weights in a named volume, so restarts do not re-download.

## API

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Ask a question. `{"question": "...", "stream": false}` |
| `POST /search` | Retrieval only, no LLM — shows what the model would see |
| `GET /nodes/{node_id}` | Raw text of one document node |
| `GET /health` | Device, model, Ollama reachability, cache state |

A grounded answer carries citations that deep-link into lex.uz:

```json
{
  "answer": "Malaka sertifikati uch yil muddatga beriladi.",
  "grounded": true,
  "citations": [{
    "node_id": "-8206038",
    "path": "234-qaror › 7-ilova › 6-bob. Malaka imtihoni",
    "url": "https://lex.uz/uz/docs/-8193120#-8206038"
  }],
  "quotes": ["Malaka sertifikati uch yil muddatga beriladi."]
}
```

## How hallucination is prevented

Four gates. **All four are plain Python, not model judgement**, so they behave
identically whether you run a 4B model or a 32B one:

1. **Retrieval gate** — if nothing is retrieved above a relevance threshold,
   the refusal is returned and the LLM is never called at all.
2. **Sufficiency** — the model answers in a fixed JSON shape with a
   `sufficient` boolean, and must set it to `false` when the passages do not
   contain the answer.
3. **Quote verification** — every supporting quote the model gives must appear
   verbatim in the retrieved text. Fabricated evidence is caught mechanically,
   not judged. One retry, then refusal.
4. **Citation validity** — every cited node id must be one that was actually
   retrieved.

Any gate failing returns the same constant refusal string, so refusals can be
tested by exact equality.

## Verifying it works

Both run inside the container, so they need nothing installed beyond Docker.
Allow about 5 minutes: the container reranks on CPU (~4.8 s per question), so
67 questions is not quick. It is working, not hung.

```bash
make verify        # retrieval metrics + gate 1 calibration (no Ollama needed)
make verify-full   # adds answer accuracy and refusal accuracy
```

`make verify` scores the system against `eval/gold.jsonl` — 67 Uzbek questions,
42 answerable and 25 deliberately unanswerable-but-on-topic. The unanswerable
set is the one that matters: it is what proves the system refuses rather than
invents.

### Measured results

`qwen3:8b`, 67-question gold set, Ollama healthy, 0 errors:

| Metric | Value |
|---|---|
| Retrieval Recall@5 | 1.000 |
| Retrieval MRR | 0.984 |
| Answered correctly (of 42 answerable) | 0.929 |
| False refusals | 0.071 |
| **Refusal accuracy (of 25 unanswerable)** | **0.960** |
| Latency p50 / p95 | 1.3 s / 3.4 s |

Two caveats, stated plainly:

* Recall@5 is optimistic. Most answerable questions were generated from the
  chunks they target, so they reuse the document's own wording. The eight
  hand-written paraphrases are the honest subset, and they also score 8/8.
* The gates verify **grounding, not responsiveness**. The one unanswerable
  question that still gets answered quotes the document verbatim with a valid
  citation — the model simply answered a nearby question instead of refusing.
  Nothing is fabricated; no deterministic check we tested separates that case
  from a correct answer, so it is a known limit rather than a fixed bug.

To confirm no GPU assumption leaked in, force the CPU path and compare:

```bash
CUDA_VISIBLE_DEVICES="" make verify-host   # maintainers, with a local Python env
```

The container is CPU-only already, so `make verify` is itself the CPU-path run.
Measured on a 2x L40S host, the GPU and CPU paths agree exactly -- Recall@5
1.000, MRR 0.984, identical gate-1 score distributions -- differing only in
latency (292 ms vs 4,794 ms per question). Retrieval quality does not depend
on your hardware.

Retrieval metrics should match the GPU run.

## Rebuilding (maintainers only)

The PM never needs these; the outputs are committed.

```bash
pip install -r requirements-build.txt
make corpus    # re-scrape lex.uz -> qaror_234_2026.pdf + data/corpus.jsonl
make index     # re-chunk and re-embed -> index/
```

`index/manifest.json` records a hash of the corpus it was built from. If the
corpus changes without the index being rebuilt, the API refuses to start rather
than serve citations that point at the wrong text.

## Troubleshooting

See [RUNNING.md](RUNNING.md#13-troubleshooting) for the full list.

**`connection refused` to Ollama** — the usual cause is not that Ollama is
stopped, but that it is listening only on `127.0.0.1`. A container cannot reach
the host's loopback, so `docker compose up` fails even though `ollama list`
works fine in your terminal. Make Ollama listen on all interfaces:

```bash
# Linux / macOS, running it yourself
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Linux, if Ollama runs under systemd
sudo systemctl edit ollama
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama

# macOS, using the desktop app
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"   # then quit and reopen Ollama
```

Windows with Docker Desktop: set a system environment variable
`OLLAMA_HOST=0.0.0.0:11434` and restart Ollama from the tray.

Check it worked:

```bash
docker run --rm --add-host host.docker.internal:host-gateway curlimages/curl \
  -s http://host.docker.internal:11434/api/version
```

> Ollama has no authentication, so `0.0.0.0` exposes it to your whole local
> network. On an untrusted network bind the Docker bridge only
> (`OLLAMA_HOST=172.17.0.1:11434` on Linux) or firewall port 11434.

Confusingly, `OLLAMA_HOST` means two different things: for the Ollama *server*
it is the address to bind, and for this API (in `.env`) it is the URL to
connect to. They are unrelated settings that happen to share a name.

**First request takes minutes** — the models are still downloading. Run
`make warmup` and watch `/health`.

**Out of memory** — lower `OLLAMA_NUM_CTX` in `.env` before switching to a
smaller model; the KV cache, not the weights, is usually what overflows.

**Slow on CPU** — expected. Set `OLLAMA_MODEL=qwen3:4b` in `.env`. Refusal
behaviour is unaffected by model size.

### Using a remote Ollama

If the machine cannot run a model at acceptable speed, point only generation
elsewhere by setting `OLLAMA_HOST` in `.env`. Retrieval still runs locally.

> Ollama has no authentication. Only do this over a private network or an SSH
> tunnel — never expose port 11434 to the internet.
