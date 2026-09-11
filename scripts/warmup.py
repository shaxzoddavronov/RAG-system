#!/usr/bin/env python3
"""Download the embedder and reranker up front.

Roughly 4.6 GB arrives from HuggingFace the first time. Left to happen inside
the first API request it looks like a hang, and behind a corporate proxy it is
the step most likely to fail. Doing it deliberately, with retries and visible
progress, turns a mysterious stall into an ordinary download.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

ATTEMPTS = 3


def fetch(label: str, loader) -> bool:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            print(f"[{label}] downloading (attempt {attempt}/{ATTEMPTS}) ...", flush=True)
            started = time.perf_counter()
            loader()
            print(f"[{label}] ready in {time.perf_counter() - started:.1f}s", flush=True)
            return True
        except Exception as exc:                      # network, proxy, disk
            print(f"[{label}] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if attempt < ATTEMPTS:
                time.sleep(5 * attempt)
    return False


def main() -> int:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = settings.device
    print(f"device: {device}\n")

    ok = fetch(settings.embed_model,
               lambda: SentenceTransformer(settings.embed_model, device=device))
    ok &= fetch(settings.rerank_model,
                lambda: CrossEncoder(settings.rerank_model, device=device))

    if not ok:
        print("\nOne or more models could not be downloaded.", file=sys.stderr)
        print("If you are behind a proxy, set HTTPS_PROXY, or ask for the models "
              "as a release asset and point HF_HOME at them.", file=sys.stderr)
        return 1

    from huggingface_hub.constants import HF_HUB_CACHE
    cache = Path(HF_HUB_CACHE)
    size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 1e9
    print(f"\nboth models cached in {cache} ({size:.1f} GB)")
    print("This machine can now run fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
