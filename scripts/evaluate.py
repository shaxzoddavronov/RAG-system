#!/usr/bin/env python3
"""Measure retrieval quality, calibrate Gate 1, and (with --full) test answers.

Run without --full to get retrieval metrics and a threshold recommendation
without needing Ollama at all. That half is deterministic and is what must stay
identical between the dev server and the PM's laptop.

  python scripts/evaluate.py                 # retrieval + threshold calibration
  python scripts/evaluate.py --full          # also generate and check answers
  CUDA_VISIBLE_DEVICES="" python scripts/evaluate.py    # force the CPU path
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import REFUSAL, settings
from app.retrieval import Retriever

GOLD = Path(__file__).resolve().parent.parent / "eval" / "gold.jsonl"


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD.open(encoding="utf-8")]


def retrieval_pass(retriever: Retriever, gold: list[dict]) -> list[dict]:
    rows = []
    for item in gold:
        started = time.perf_counter()
        hits, _ = retriever.search(item["question"])
        rows.append({
            **item,
            "top_score": hits[0].rerank_score if hits else 0.0,
            "chunk_ids": [h.chunk["chunk_id"] for h in hits],
            "latency_ms": (time.perf_counter() - started) * 1000,
        })
    return rows


def retrieval_metrics(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    hit_ranks = []
    for r in answerable:
        try:
            hit_ranks.append(r["chunk_ids"].index(r["expect_chunk"]) + 1)
        except ValueError:
            hit_ranks.append(0)
    recall = sum(1 for r in hit_ranks if r) / len(answerable)
    mrr = sum(1 / r for r in hit_ranks if r) / len(answerable)
    return {"recall_at_5": recall, "mrr": mrr, "ranks": hit_ranks}


def calibrate(rows: list[dict]) -> tuple[float, list[tuple]]:
    """Sweep Gate 1 and pick the threshold with the best balanced accuracy."""
    pos = [r["top_score"] for r in rows if r["answerable"]]
    neg = [r["top_score"] for r in rows if not r["answerable"]]
    table, best = [], (0.0, -1.0)
    for step in range(30, 86):
        threshold = step / 100
        kept = sum(1 for s in pos if s >= threshold) / len(pos)        # true positive
        refused = sum(1 for s in neg if s < threshold) / len(neg)      # true negative
        balanced = (kept + refused) / 2
        table.append((threshold, kept, refused, balanced))
        if balanced > best[1]:
            best = (threshold, balanced)
    return best[0], table


async def answer_pass(rows: list[dict], ceiling: float = 45.0) -> dict:
    """End-to-end: does the whole pipeline answer and refuse correctly?"""
    from app.gates import gate_retrieval, verify_answer
    from app.generate import Generator, OllamaError
    from app.retrieval import Retriever as R

    retriever = R(settings)
    generator = Generator(settings)
    reachable, present = await generator.reachable()
    if not (reachable and present):
        await generator.aclose()
        return {"skipped": f"ollama unreachable or {settings.ollama_model} not pulled"}

    correct_refusals = wrong_refusals = grounded = errors = 0
    latencies = []
    # A question that answers correctly but takes minutes is still a failure:
    # it holds an LLM slot and, on a CPU-only machine, looks like a hang. This
    # caught a runaway generation that every accuracy metric happily ignored.
    over_ceiling: list[tuple[str, float]] = []
    for row in rows:
        started = time.perf_counter()
        hits, _ = retriever.search(row["question"])
        answered = False
        if gate_retrieval(hits, settings.rerank_threshold).ok:
            blocks = retriever.build_context(hits)
            try:
                payload = await generator.generate(row["question"], blocks)
                answered = verify_answer(payload, blocks).ok
            except OllamaError:
                errors += 1
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        if elapsed_ms > ceiling * 1000:
            over_ceiling.append((row["question"], elapsed_ms / 1000))

        if row["answerable"]:
            grounded += answered
            wrong_refusals += not answered
        else:
            correct_refusals += not answered

    await generator.aclose()
    n_ans = sum(1 for r in rows if r["answerable"])
    n_ref = len(rows) - n_ans
    latencies.sort()
    return {
        "answered_correctly": grounded / n_ans,
        "false_refusals": wrong_refusals / n_ans,
        "refusal_accuracy": correct_refusals / n_ref,
        "ollama_errors": errors,
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95) - 1],
        "max_ms": latencies[-1],
        "over_ceiling": over_ceiling,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="also generate answers (needs Ollama)")
    parser.add_argument("--max-latency", type=float, default=45.0,
                        help="per-question ceiling in seconds; exceeding it fails "
                             "the run even if the answer was correct (default: 45)")
    args = parser.parse_args()

    gold = load_gold()
    retriever = Retriever(settings)
    print(f"device      : {settings.device}")
    print(f"index       : {len(retriever.chunks)} chunks")
    print(f"gold set    : {len(gold)} questions "
          f"({sum(1 for g in gold if g['answerable'])} answerable)\n")

    rows = retrieval_pass(retriever, gold)
    metrics = retrieval_metrics(rows)
    pos = [r["top_score"] for r in rows if r["answerable"]]
    neg = [r["top_score"] for r in rows if not r["answerable"]]

    print("RETRIEVAL")
    print(f"  Recall@5           : {metrics['recall_at_5']:.3f}")
    for category in ("prose", "table", "prose-auto"):
        subset = [r for r in rows if r.get("category") == category]
        if not subset:
            continue
        got = sum(1 for r in subset if r["expect_chunk"] in r["chunk_ids"])
        note = "  (questions reuse the chunk's own wording -- optimistic)" \
            if category != "prose" else "  (genuine paraphrases)"
        print(f"    {category:12}     : {got}/{len(subset)}{note}")
    print(f"  MRR                : {metrics['mrr']:.3f}")
    print(f"  median latency     : {statistics.median(r['latency_ms'] for r in rows):.0f} ms")
    print(f"  score answerable   : median {statistics.median(pos):.3f}  min {min(pos):.3f}")
    print(f"  score unanswerable : median {statistics.median(neg):.3f}  max {max(neg):.3f}")

    best, table = calibrate(rows)
    print("\nGATE 1 CALIBRATION")
    print("  threshold   kept(answerable)  refused(unanswerable)  balanced")
    for threshold, kept, refused, balanced in table:
        if abs(threshold - best) < 0.001 or round(threshold * 100) % 5 == 0:
            marker = "  <-- best" if abs(threshold - best) < 0.001 else ""
            print(f"    {threshold:.2f}         {kept:.3f}                {refused:.3f}"
                  f"              {balanced:.3f}{marker}")
    kept_here = sum(1 for s_ in pos if s_ >= settings.rerank_threshold) / len(pos)
    refused_here = sum(1 for s_ in neg if s_ < settings.rerank_threshold) / len(neg)
    print(f"\n  best-on-paper      : {best:.2f}")
    print(f"  configured         : {settings.rerank_threshold:.2f}  "
          f"(keeps {kept_here:.0%} answerable, refuses {refused_here:.0%} unanswerable)")
    print("\n  The reranker saturates -- relevant passages all land near 0.73 -- so the")
    print("  best-on-paper threshold sits in a margin of a few thousandths and would")
    print("  flip on float noise. The configured value is deliberately looser.")
    print("  Questions that are on-topic but unanswerable are EXPECTED to pass gate 1;")
    print("  gates 2-4 catch them. Run with --full to measure that.")

    if args.full:
        print("\nEND-TO-END")
        result = asyncio.run(answer_pass(rows, args.max_latency))
        if "skipped" in result:
            print(f"  skipped: {result['skipped']}")
            return 0

        slow = result.pop("over_ceiling")
        for key, value in result.items():
            print(f"  {key:20}: {value:.3f}" if isinstance(value, float)
                  else f"  {key:20}: {value}")

        print(f"\n  latency ceiling    : {args.max_latency:g}s")
        if slow:
            print(f"  ** {len(slow)} question(s) OVER CEILING **")
            for question, seconds in sorted(slow, key=lambda x: -x[1]):
                print(f"     {seconds:6.1f}s  {question[:72]}")
            print("\n  A correct answer that takes this long still fails: it ties up an")
            print("  LLM slot, and on a CPU-only machine it is indistinguishable from")
            print("  a hang. Check OLLAMA_NUM_PREDICT and the gate-3 retry count.")
            return 1
        print(f"  all {len(rows)} questions within ceiling (max "
              f"{result['max_ms'] / 1000:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
