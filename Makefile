.PHONY: help setup warmup dev up up-cpu up-gpu down logs verify verify-full \
        verify-host corpus index evalset clean

help:
	@echo "Setup (run once):"
	@echo "  make setup        detect your hardware and write .env"
	@echo "  make warmup       download the embedder + reranker (~4.6 GB)"
	@echo "  make up           start the API on http://localhost:8000 (Docker)"
	@echo "                    uses the GPU automatically if this host has one"
	@echo "  make up-cpu       force the CPU image even on a GPU host"
	@echo "  make up-gpu       force the GPU image (fails if there is no GPU)"
	@echo "  make dev          start the API without Docker (needs pip install)"
	@echo ""
	@echo "Checks:"
	@echo "  make verify       retrieval metrics + gate 1 calibration (runs in the container)"
	@echo "  make verify-full  end-to-end, including refusal accuracy (needs Ollama)"
	@echo ""
	@echo "Rebuild (maintainers only -- outputs are committed):"
	@echo "  make corpus       re-scrape lex.uz -> PDF + data/corpus.jsonl"
	@echo "  make index        re-chunk and re-embed -> index/"
	@echo "  make evalset      regenerate eval/gold.jsonl"

PY ?= python

# GPU is selected by hardware, not by a flag. nvidia-smi ships with the NVIDIA
# driver, so its presence is a reliable proxy for "this host has a usable card"
# without needing docker, python or torch to answer the question. The overlay
# swaps torch for a CUDA build AND reserves the device; neither works alone.
#
# Not available on Windows, which has no make -- "Rerank on the GPU" in
# RUNNING.md gives the explicit two-file command there.
COMPOSE := docker compose -f docker-compose.yml
GPU_OVERLAY := -f docker-compose.gpu.yml
ifneq ($(shell command -v nvidia-smi 2>/dev/null),)
COMPOSE_AUTO := $(COMPOSE) $(GPU_OVERLAY)
COMPOSE_MODE := GPU (nvidia-smi found)
else
COMPOSE_AUTO := $(COMPOSE)
COMPOSE_MODE := CPU (no nvidia-smi on PATH)
endif

setup:
	$(PY) scripts/setup_profile.py

warmup:
	$(PY) scripts/warmup.py

# Native run, no Docker. Also puts the reranker on a local GPU if there is one.
dev:
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

up:
	@echo "build mode  : $(COMPOSE_MODE)"
	$(COMPOSE_AUTO) up -d --build
	@echo "API on http://localhost:8000/docs"
	@echo "Confirm the device with: curl -s localhost:8000/health"

up-cpu:
	$(COMPOSE) up -d --build

# Explicit opt-in. Fails loudly with no GPU rather than silently serving on CPU.
up-gpu:
	$(COMPOSE) $(GPU_OVERLAY) up -d --build

down:
	$(COMPOSE_AUTO) down

logs:
	$(COMPOSE_AUTO) logs -f api

# Run inside the container: the PM installs Docker and Ollama, not a Python
# environment, so these must not depend on host packages.
verify:
	$(COMPOSE_AUTO) exec -T api python scripts/evaluate.py

verify-full:
	$(COMPOSE_AUTO) exec -T api python scripts/evaluate.py --full

# Same checks against a local Python env, for maintainers.
verify-host:
	$(PY) scripts/evaluate.py --full

corpus:
	$(PY) scripts/build_corpus.py

index:
	$(PY) scripts/build_index.py

evalset:
	$(PY) scripts/build_evalset.py

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
