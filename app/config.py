"""Runtime configuration.

Every hardware-dependent value is resolved here, once. Nothing else in the
codebase may call .cuda() or assume a GPU exists -- that is what lets the same
commit run on a 2x L40S server and on a PM's laptop with no NVIDIA card at all.
"""
import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load .env when running outside Docker. Compose injects the file itself via
# env_file, so this is a no-op there; without it a native run silently ignores
# the .env that setup_profile.py just wrote. Real environment variables always
# win over the file.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # optional dependency
    pass

# The single canonical refusal. Every gate returns exactly this string so the
# eval suite can assert on equality rather than fuzzy-matching prose.
REFUSAL = "Hujjatda bu haqida maʼlumot yoʻq."


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Ollama -----------------------------------------------------------
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b"))
    # Ollama silently truncates the prompt to num_ctx and the model then answers
    # from nothing. Always send it explicitly.
    num_ctx: int = field(default_factory=lambda: _int("OLLAMA_NUM_CTX", 16384))
    temperature: float = field(default_factory=lambda: _float("OLLAMA_TEMPERATURE", 0.0))
    seed: int = field(default_factory=lambda: _int("OLLAMA_SEED", 42))
    request_timeout: float = field(default_factory=lambda: _float("OLLAMA_TIMEOUT", 120.0))
    # Hard cap on output tokens. Without it the model can fail to terminate --
    # observed running past 7,500 tokens on a question the document cannot
    # answer, holding an LLM slot for minutes. A grounded answer plus quotes is
    # a few hundred tokens; anything beyond this is a runaway, not an answer.
    num_predict: int = field(default_factory=lambda: _int("OLLAMA_NUM_PREDICT", 800))
    # Qwen3 and friends emit a long reasoning chain before the answer. With a
    # forced JSON schema that reasoning buys nothing -- the answer is extracted
    # from retrieved text, not derived -- but it cost 70s per question on an
    # L40S, which would be minutes on a laptop CPU. Off by default.
    think: bool = field(
        default_factory=lambda: os.getenv("OLLAMA_THINK", "0") not in ("0", "false", ""))
    max_concurrent_llm: int = field(default_factory=lambda: _int("MAX_CONCURRENT_LLM", 2))

    # --- Models -----------------------------------------------------------
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-m3"))
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    device_pref: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))

    # --- Retrieval --------------------------------------------------------
    top_k_lexical: int = field(default_factory=lambda: _int("TOP_K_LEXICAL", 30))
    top_k_dense: int = field(default_factory=lambda: _int("TOP_K_DENSE", 30))
    top_k_rerank: int = field(default_factory=lambda: _int("TOP_K_RERANK", 5))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    # Gate 1. bge-reranker-v2-m3 saturates: every genuinely relevant passage
    # scores ~0.73 and clearly irrelevant ones ~0.51, so this is close to a
    # binary signal rather than a graded one. 0.60 sits in the wide empty space
    # between those clusters. Deliberately NOT tuned to the last percent --
    # a threshold fitted to a 0.007 margin would flip on float noise alone.
    # Topically-relevant-but-unanswerable questions are meant to pass Gate 1
    # and be caught by gates 2-4 instead.
    rerank_threshold: float = field(
        default_factory=lambda: _float("RERANK_THRESHOLD", 0.60))

    # --- Paths ------------------------------------------------------------
    index_dir: Path = field(default_factory=lambda: ROOT / os.getenv("INDEX_DIR", "index"))
    corpus_path: Path = field(
        default_factory=lambda: ROOT / os.getenv("CORPUS_PATH", "data/corpus.jsonl"))

    @property
    def device(self) -> str:
        return resolve_device(self.device_pref)


@functools.lru_cache(maxsize=None)
def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device that actually exists on this machine.

    torch does not fall back on its own: loading a model with device="cuda" on a
    machine without a GPU raises. Ollama, by contrast, handles its own detection
    and needs nothing from us.
    """
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


settings = Settings()
