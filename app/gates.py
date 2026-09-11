"""The four hallucination gates.

Prompting alone does not produce reliable refusals. These gates are ordinary
Python, so their behaviour does not change when the PM runs a 4B model instead
of a 32B one -- which is the property the spec is actually asking for.

Every failure path returns the same constant string, so the eval suite can
assert refusals by equality rather than by matching prose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.normalize import fold

ELLIPSIS = re.compile(r"\.{3,}|\u2026")


@dataclass
class Verdict:
    ok: bool
    reason: str | None = None


def gate_retrieval(hits, threshold: float) -> Verdict:
    """Gate 1 -- nothing retrieved well enough, so the LLM is never called.

    This is the cheapest and most effective gate: an out-of-scope question never
    reaches the model, so the model cannot invent an answer to it.
    """
    if not hits:
        return Verdict(False, "retrieval")
    if hits[0].rerank_score < threshold:
        return Verdict(False, "retrieval")
    return Verdict(True)


def gate_sufficiency(payload: dict) -> Verdict:
    """Gate 2 -- the model itself reports the context was not enough."""
    if not payload.get("sufficient") or not payload.get("answer"):
        return Verdict(False, "sufficiency")
    return Verdict(True)


def gate_quotes(payload: dict, blocks: list[dict], min_len: int = 12) -> Verdict:
    """Gate 3 -- every supporting quote must occur verbatim in the retrieved text.

    Matching is exact (on the folded form: normalized, lowercased, depunctuated),
    and deliberately so. Tolerating near-misses was tried and rejected: Uzbek
    negation is the infix -ma-, so "oʻtkaziladi" (is carried out) becomes
    "oʻtkazilmaydi" (is not carried out) with a two-character edit. That is
    indistinguishable by edit distance from an innocent inflection slip such as
    "taʼsir" for "taʼsiri", and admitting the second necessarily admits the
    first. In a legal document a silently inverted obligation is far worse than
    a refusal, so the strict rule stands.

    The cost is that an otherwise correct answer can be thrown away over one
    mistyped character. That is handled upstream instead: a failure here earns a
    retry at a different seed, which gives the model another chance to transcribe
    the sentence properly.
    """
    quotes = [q for q in payload.get("quotes", []) if len(q.strip()) >= min_len]
    if not quotes:
        return Verdict(False, "quote")

    haystack = fold(" \n ".join(b["text"] for b in blocks))
    for quote in quotes:
        # Models elide with "..." even when told not to. Each surviving fragment
        # must still appear verbatim, so this tolerates the elision without
        # letting any invented text through.
        fragments = [f for f in ELLIPSIS.split(quote) if len(f.strip()) >= min_len]
        for fragment in fragments or [quote]:
            if fold(fragment) not in haystack:
                return Verdict(False, "quote")
    return Verdict(True)


def resolve_citation(value: str, blocks: list[dict]) -> str | None:
    """Map a model-written citation onto a retrieved block's cite_id, or None.

    Blocks are labelled with chunk_id ("-8205881#0"), but models routinely write
    the bare node_id ("-8205881") instead -- it is the visually obvious prefix,
    and the older prompt example showed exactly that shape. Rejecting those cost
    a correct, quote-verified answer over punctuation.

    A bare node_id is accepted only when exactly one retrieved block carries it.
    That keeps the guarantee intact: the citation still resolves to a passage we
    actually retrieved, with no guessing. When several retrieved blocks share a
    node_id -- every row of the Appendix 1 table lives in one node -- the
    reference is genuinely ambiguous and is refused, because picking one would
    attach the wrong row's text to the citation.
    """
    if value in {b["cite_id"] for b in blocks}:
        return value
    matches = {b["cite_id"] for b in blocks if b["node_id"] == value}
    return matches.pop() if len(matches) == 1 else None


def gate_citations(payload: dict, blocks: list[dict]) -> Verdict:
    """Gate 4 -- cited ids must come from what we actually retrieved."""
    cited = [c for c in payload.get("citations", []) if c]
    if not cited:
        return Verdict(False, "citation")
    if any(resolve_citation(c, blocks) is None for c in cited):
        return Verdict(False, "citation")
    return Verdict(True)


def verify_answer(payload: dict, blocks: list[dict]) -> Verdict:
    """Gates 2-4, in order. Gate 1 runs earlier, before generation."""
    for gate in (gate_sufficiency(payload),
                 gate_quotes(payload, blocks),
                 gate_citations(payload, blocks)):
        if not gate.ok:
            return gate
    return Verdict(True)
