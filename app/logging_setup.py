"""Logging for the API.

Uvicorn's access line -- `POST /ask HTTP/1.1 200 OK` -- reports that a request
happened and nothing about what the system did with it. When an answer looks
wrong the questions are always the same: what was retrieved, what did the
reranker score it, which gate refused, where did the time go. This module makes
the pipeline answer them itself.

One line per stage, each tagged with a short request id so that concurrent
requests stay readable when they interleave (MAX_CONCURRENT_LLM allows two
generations at once, and retrieval is threadpooled on top of that).

Two kinds of noise are suppressed deliberately:

  * HuggingFace weight-loading progress bars. They render as hundreds of
    carriage-returned frames, which is fine on a TTY and unreadable in
    `docker compose logs`.
  * Swagger's own traffic (/docs, /openapi.json, /favicon.ico). Opening the UI
    fires three requests that say nothing about the service.
"""
from __future__ import annotations

import logging
import os
import sys

LOGGER_NAME = "rag"

# Swagger UI chatter. Suppressed at the access-log level only -- a failure on
# one of these still surfaces, because the filter keeps non-2xx responses.
_QUIET_PATHS = ("/docs", "/openapi.json", "/favicon.ico", "/redoc")


class _AccessNoiseFilter(logging.Filter):
    """Drop successful requests for the documentation UI, keep everything else.

    Reads uvicorn's structured args -- (client, method, path, http_version,
    status) -- rather than the rendered line. The rendered line is not usable
    here: uvicorn's formatter appends the status phrase afterwards, so at filter
    time the record still ends at the bare number and a text match on the status
    silently never fires.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not (isinstance(args, tuple) and len(args) == 5):
            return True                      # not an access record; leave it alone
        _, method, path, _, status = args
        if method != "GET":
            return True
        if str(path).split("?")[0] not in _QUIET_PATHS:
            return True
        try:
            # Keep failures: a 404 on /docs is worth seeing, a 200 is not.
            return int(status) >= 400
        except (TypeError, ValueError):
            return True


def _silence_progress_bars() -> None:
    """Stop transformers/HF from writing progress bars into the log stream.

    The env var covers hub downloads; the explicit call covers weight loading,
    which is what produces the `Loading weights: 100%|...| 391/391` lines. Both
    are best-effort -- a missing or renamed API must never break startup.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for module, attr in (
        ("transformers.utils.logging", "disable_progress_bar"),
        ("huggingface_hub.utils.tqdm", "disable_progress_bars"),
    ):
        try:
            __import__(module)
            getattr(sys.modules[module], attr)()
        except Exception:
            pass


def configure(level: str = "INFO") -> logging.Logger:
    """Install the log format and return the application logger."""
    _silence_progress_bars()

    resolved = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    ))

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved)
    logger.handlers = [handler]
    logger.propagate = False       # uvicorn owns the root logger; don't double-print

    logging.getLogger("uvicorn.access").addFilter(_AccessNoiseFilter())
    # httpx logs an INFO line for every Ollama call, duplicating what we log
    # ourselves with far more context.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
