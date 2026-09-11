#!/usr/bin/env python3
"""Detect this machine's capability and write a matching .env.

The PM should not have to know how much VRAM their card has. This reads
nvidia-smi if it exists, falls back cleanly when it does not, and picks a model
that will actually fit. Context length is capped alongside the model because the
KV cache costs VRAM on top of the weights -- that, not the weights, is what
usually triggers an out-of-memory error on a small card.
"""
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

# (minimum VRAM in GB, model, context window)
TIERS = [
    (24, "qwen3:14b", 16384),
    (12, "qwen3:8b", 16384),
    (8, "qwen3:4b", 8192),
]
CPU_TIER = ("qwen3:4b", 8192)
APPLE_TIER = ("qwen3:8b", 8192)


def detect_vram_gb() -> int:
    """Largest single-GPU VRAM in GB, or 0 if there is no usable NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return 0
    sizes = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
    return max(sizes) // 1024 if sizes else 0


def choose() -> tuple[str, int, str]:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        model, ctx = APPLE_TIER
        return model, ctx, "Apple Silicon (Metal)"

    vram = detect_vram_gb()
    for minimum, model, ctx in TIERS:
        if vram >= minimum:
            return model, ctx, f"NVIDIA GPU, {vram} GB VRAM"
    model, ctx = CPU_TIER
    return model, ctx, "no usable GPU, running on CPU" if vram == 0 else \
        f"NVIDIA GPU, {vram} GB VRAM (below 8 GB, using CPU tier)"


def main() -> int:
    model, ctx, why = choose()

    if ENV_PATH.exists() and "--force" not in sys.argv:
        print(f".env already exists at {ENV_PATH} -- leaving it alone (--force to overwrite)")
        return 0

    ENV_PATH.write_text(
        "# Written by scripts/setup_profile.py -- safe to edit by hand.\n"
        f"# Detected: {why}\n"
        f"OLLAMA_MODEL={model}\n"
        f"OLLAMA_NUM_CTX={ctx}\n"
        "OLLAMA_HOST=http://localhost:11434\n"
        "DEVICE=auto\n",
        encoding="utf-8")

    print(f"detected : {why}")
    print(f"model    : {model}  (context {ctx})")
    print(f"written  : {ENV_PATH}")
    print(f"\nNext:  ollama pull {model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
