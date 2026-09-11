"""Turn corpus nodes into retrievable chunks.

Two deliberate choices drive this module:

1. Structure-aware, not similarity-based. A *band* (numbered clause) is a
   self-contained legal unit, and the document already marks where each one
   starts. Splitting mid-band is what produces confidently wrong answers about
   deadlines and fees, so bands are the atomic unit.

2. Table rows become sentences. Appendix 1 is a 267-row table of activity types
   with review periods and fees. Embedded as markup it retrieves terribly; one
   verbalized sentence per row retrieves well and keeps the numbers intact.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.normalize import normalize

# Headings give chunks their breadcrumb but are not themselves worth retrieving --
# "1-bob. Umumiy qoidalar" answers no question on its own.
HEADING_KINDS = {
    "ACCEPTING_BODY", "ACT_FORM", "ACT_TITLE", "ACT_TITLE_APPL",
    "APPL_BANNER_LANDSCAPE_TITLE", "TEXT_HEADER_DEFAULT", "TEXT_CENTER",
}

# A band starts with "12." or "12-band." at the very beginning of the node.
BAND_START = re.compile(r"^(\d+)\s*(?:-band)?\s*\.")
SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")

MAX_CHARS = 1800     # ~450 tokens of Uzbek; above this, split on sentences

# Short chunks are NOT merged. "30. Malaka sertifikati uch yil muddatga beriladi."
# is 49 characters and a complete, answerable legal fact; gluing it to a neighbour
# would bury it. The breadcrumb prefix in embed_text supplies the missing context.
# What does need removing is layout scaffolding from the blank application forms
# in appendices 2 and 6 -- "(F.I.O)", "__________ga" -- which can never answer
# anything and only adds noise to the index.
PLACEHOLDER = re.compile(r"[_\u2014\-\.]{3,}")
MIN_CONTENT_CHARS = 15


@dataclass
class Chunk:
    chunk_id: str
    node_id: str
    kind: str
    path: str
    ilova: str | None
    bob: str | None
    url: str
    text: str           # what the LLM is shown
    embed_text: str     # breadcrumb + text; what actually gets embedded
    context_text: str   # the whole parent band, for small-to-big expansion
    band: str | None = None
    extra: dict = field(default_factory=dict)


def _breadcrumb(rec: dict) -> str:
    return (rec.get("path") or "234-qaror").replace(" > ", " \u203a ")


def is_noise(text: str) -> bool:
    """True for blank-form scaffolding and fragments with no answerable content."""
    stripped = PLACEHOLDER.sub(" ", text)
    letters = re.sub(r"[^^\w\u02bb\u02bc]", "", stripped, flags=re.UNICODE)
    if len(letters) < MIN_CONTENT_CHARS:
        return True
    # Mostly underscores: a fill-in-the-blank line rather than a statement.
    return len(PLACEHOLDER.findall(text)) >= 2 and len(letters) < 40


def _split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split oversized text on sentence boundaries, carrying one sentence of overlap."""
    if len(text) <= limit:
        return [text]
    sentences = SENTENCE_END.split(text)
    parts, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > limit:
            parts.append(current.strip())
            tail = current.split(". ")[-1]          # one sentence of overlap
            current = (tail + " " + sentence) if len(tail) < limit // 3 else sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        parts.append(current.strip())
    return parts or [text]


def _verbalize_table(rec: dict) -> list[tuple[str, str, dict]]:
    """One (row_key, sentence, fields) per data row, labelled by the header row.

    Rows whose only non-empty cell is the first one are section headings inside
    the table ("10. Kimyo sanoati"); they are carried onto the rows beneath them
    instead of becoming chunks of their own.
    """
    rows = rec.get("rows") or []
    if len(rows) < 2:
        return []

    header = [normalize(c) for c in rows[0]]
    out: list[tuple[str, str, dict]] = []
    section = None

    for row in rows[1:]:
        cells = [normalize(c) for c in row]
        filled = [c for c in cells if c]
        if not filled:
            continue
        if len(filled) == 1 and cells[0]:
            section = cells[0].rstrip(".")
            continue

        # Only a numeric first cell is a band number. The nested application-form
        # tables in appendices 3 and 6 put a field label there instead, and
        # calling that "<label>-band" produces nonsense.
        first = cells[0].rstrip(".") if cells[0] else ""
        row_key = first if first.isdigit() else ""
        if row_key:
            subject = cells[1] if len(cells) > 1 and cells[1] else ""
            parts = [f"{row_key}-band:", subject.rstrip(".")]
        else:
            subject = " ".join(c for c in cells[:2] if c)
            parts = [subject.rstrip(".")]
        fields: dict[str, str] = {}
        for name, value in zip(header[2:], cells[2:]):
            if value and name:
                fields[name] = value
                parts.append(f". {name}: {value}")
        sentence = " ".join(parts).replace(" . ", ". ").strip()
        if section:
            sentence = f"{section}. {sentence}"
            fields["boʻlim"] = section
        out.append((row_key, sentence, fields))
    return out


def build_chunks(corpus_path: Path) -> list[Chunk]:
    records = [json.loads(line) for line in corpus_path.open(encoding="utf-8")]
    chunks: list[Chunk] = []

    # Group consecutive prose nodes into bands before chunking.
    groups: list[list[dict]] = []
    for rec in records:
        if rec["type"] == "table" or rec["kind"] in HEADING_KINDS:
            groups.append([rec])
            continue
        text = normalize(rec["text"])
        starts_band = bool(BAND_START.match(text))
        same_section = (
            groups and groups[-1][-1]["type"] == "text"
            and groups[-1][-1]["kind"] not in HEADING_KINDS
            and groups[-1][-1].get("path") == rec.get("path")
        )
        if starts_band or not same_section:
            groups.append([rec])
        else:
            groups[-1].append(rec)

    for group in groups:
        head = group[0]
        if head["kind"] in HEADING_KINDS:
            continue

        path = _breadcrumb(head)

        if head["type"] == "table":
            for row_key, sentence, fields in _verbalize_table(head):
                if is_noise(sentence):
                    continue
                chunks.append(Chunk(
                    chunk_id=f"{head['node_id']}#r{row_key or len(chunks)}",
                    node_id=head["node_id"], kind=head["kind"], path=path,
                    ilova=head.get("ilova"), bob=head.get("bob"),
                    url=head["url"], text=sentence,
                    embed_text=f"{path} › {sentence}",
                    context_text=sentence, band=row_key or None, extra=fields,
                ))
            continue

        band_text = normalize(" ".join(normalize(r["text"]) for r in group))
        match = BAND_START.match(band_text)
        band_no = match.group(1) if match else None

        if is_noise(band_text):
            continue
        for i, part in enumerate(_split_long(band_text)):
            chunks.append(Chunk(
                chunk_id=f"{head['node_id']}#{i}",
                node_id=head["node_id"], kind=head["kind"], path=path,
                ilova=head.get("ilova"), bob=head.get("bob"),
                url=head["url"], text=part,
                embed_text=f"{path} › {part}",
                context_text=band_text, band=band_no,
            ))
    return chunks


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]


if __name__ == "__main__":
    import statistics
    from app.config import settings

    cs = build_chunks(settings.corpus_path)
    lengths = [len(c.text) for c in cs]
    tables = [c for c in cs if c.kind == "TABLE_STD2"]
    print(f"chunks       : {len(cs)}")
    print(f"  from tables: {len(tables)}")
    print(f"chars  min/med/max: {min(lengths)} / {int(statistics.median(lengths))} / {max(lengths)}")
    print(f"over {MAX_CHARS}   : {sum(1 for n in lengths if n > MAX_CHARS)}")
    print(f"ilovalar     : {len({c.ilova for c in cs if c.ilova})}")
    print("\n--- sample prose chunk ---")
    print(next(c.embed_text for c in cs if c.band == "14")[:300])
    print("\n--- sample table chunk ---")
    print(tables[100].embed_text[:300])
