.PHONY: help setup warmup up down logs verify verify-full corpus index evalset clean

help:
	@echo "Setup (run once):"
	@echo "  make setup        detect your hardware and write .env"
	@echo "  make warmup       download the embedder + reranker (~4.6 GB)"
	@echo "  make up           start the API on http://localhost:8000"
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

setup:
	$(PY) scripts/setup_profile.py

warmup:
	$(PY) scripts/warmup.py

up:
	docker compose up -d --build
	@echo "API on http://localhost:8000/docs"

down:
	docker compose down

logs:
	docker compose logs -f api

# Run inside the container: the PM installs Docker and Ollama, not a Python
# environment, so these must not depend on host packages.
verify:
	docker compose exec -T api python scripts/evaluate.py

verify-full:
	docker compose exec -T api python scripts/evaluate.py --full

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
