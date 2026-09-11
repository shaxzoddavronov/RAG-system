#!/usr/bin/env python3
"""
Fetch Cabinet of Ministers Resolution No. 234 (11.05.2026) from lex.uz,
strip all site chrome, and rebuild it as:
  - qaror_234_2026.pdf   full, readable text of the act only
  - corpus.jsonl         one record per document node (for the RAG index)

lex.uz serves the act as ~998 <div class="TYPE lx_elem"> blocks, each wrapping
a <div name="-<id>"> with the actual content. Those ids are stable and
addressable as https://lex.uz/uz/docs/-8193120#-<id>, so we keep them.
"""
import html
import json
import re
import sys
from pathlib import Path

import lxml.html
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

# Shared with the query path. If the corpus and the queries normalized
# differently, BM25 would silently stop matching on every apostrophe.
from app.normalize import normalize

DOC_URL = "https://lex.uz/uz/docs/-8193120"
OUT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = OUT_DIR / "qaror_234_2026.pdf"
JSONL_PATH = OUT_DIR / "data" / "corpus.jsonl"
CACHE = OUT_DIR / "data" / "lex_page.html"

def fetch() -> str:
    if CACHE.exists():
        return CACHE.read_text(encoding="utf-8")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(DOC_URL, timeout=60,
                        headers={"User-Agent": "Mozilla/5.0 (corpus-builder)"})
    resp.raise_for_status()
    resp.encoding = "utf-8"
    CACHE.write_text(resp.text, encoding="utf-8")
    return resp.text


def cell_text(node) -> str:
    return normalize(" ".join(node.itertext()))


def parse_header(tree) -> dict:
    """Date, number and entry-into-force date live in the page header, not the body."""
    blocks = tree.cssselect("div.docHeader")
    raw = normalize(re.sub(r"\s+", " ", " ".join(blocks[0].itertext()))) if blocks else ""
    raw = raw.split("Qoʻshimcha axborot")[0].strip()
    meta = {"raw": raw}
    if m := re.search(r"(\d{2}\.\d{2}\.\d{4})\s*yildagi\s*(\d+)-son", raw):
        meta["sana"], meta["raqam"] = m.group(1), m.group(2)
    if m := re.search(r"Kuchga kirish sanasi\s*(\d{2}\.\d{2}\.\d{4})", raw):
        meta["kuchga_kirish"] = m.group(1)
    return meta


def parse(page_html: str) -> list[dict]:
    """Walk the document body and emit one record per lx_elem block."""
    tree = lxml.html.fromstring(page_html)
    header = parse_header(tree)
    container = tree.get_element_by_id("main_container1")

    # Drop the per-element toolbars ("send a suggestion", "listen", "get link")
    # and the hidden classifier indexes -- pure site chrome, not part of the act.
    for junk in container.cssselect("div.lx_elem2, div.INDEXES_ON_REF, script, style"):
        junk.getparent().remove(junk)

    records = []
    for block in container.cssselect("div.lx_elem"):
        kind = block.get("class", "").replace("lx_elem", "").strip()
        holder = block.cssselect("div[name^='-']")
        if not holder:
            continue
        holder = holder[0]
        node_id = holder.get("id", "")

        tables = holder.cssselect("table")
        if tables:
            rows = []
            for tr in tables[0].cssselect("tr"):
                cells = [cell_text(td) for td in tr.cssselect("td, th")]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            records.append({"node_id": node_id, "kind": kind, "type": "table",
                            "rows": rows,
                            "text": "\n".join(" | ".join(r) for r in rows)})
        else:
            text = cell_text(holder)
            if text:
                records.append({"node_id": node_id, "kind": kind,
                                "type": "text", "text": text})

    # Prepend the act's identifying details so "when does it take effect?"
    # is answerable from the corpus itself.
    if header.get("raw"):
        records.insert(0, {"node_id": "meta", "kind": "ACT_ESSENTIAL_ELEMENTS",
                           "type": "text", "text": header["raw"],
                           "meta": header})
    return records


def add_structure(records: list[dict]) -> list[dict]:
    """Tag every node with the appendix / sub-appendix / chapter it sits under."""
    ilova = sub_ilova = bob = None
    for rec in records:
        t = rec["text"]
        if rec["kind"] == "APPL_BANNER_LANDSCAPE_TITLE":
            top = re.search(r"qaroriga\s+(\d+)-ILOVA", t)
            nested = re.search(r"(?:nizomga|tartibiga)\s+(?:(\d+)-)?ILOVA", t)
            if top:
                ilova, sub_ilova, bob = f"{top.group(1)}-ilova", None, None
            elif nested:
                sub_ilova = f"{nested.group(1)}-ilova" if nested.group(1) else "ilova"
        elif re.match(r"^\d+-bob\b", t):
            bob = t.rstrip(".")

        rec["ilova"] = ilova
        rec["sub_ilova"] = sub_ilova
        rec["bob"] = bob
        rec["path"] = " > ".join(x for x in ("234-qaror", ilova, sub_ilova, bob) if x)
        rec["url"] = f"{DOC_URL}#{rec['node_id']}"
    return records


# --------------------------------------------------------------------------- PDF

def register_fonts():
    base = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DejaVu", base / "DejaVuSerif.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", base / "DejaVuSerif-Bold.ttf"))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")


def make_styles():
    ss = getSampleStyleSheet()
    def style(name, **kw):
        opts = dict(parent=ss["Normal"], fontName="DejaVu", fontSize=9.5,
                    leading=13.5, spaceAfter=5)
        opts.update(kw)
        return ParagraphStyle(name, **opts)

    return {
        "title": style("title", fontName="DejaVu-Bold", fontSize=14, leading=19,
                       alignment=TA_CENTER, spaceAfter=10, spaceBefore=6),
        "subtitle": style("subtitle", fontName="DejaVu-Bold", fontSize=11,
                          leading=15, alignment=TA_CENTER, spaceAfter=8),
        "heading": style("heading", fontName="DejaVu-Bold", fontSize=10.5,
                         leading=14, spaceBefore=10, spaceAfter=6),
        "appendix": style("appendix", fontName="DejaVu-Bold", fontSize=10,
                          leading=14, alignment=TA_RIGHT, spaceBefore=14,
                          spaceAfter=6),
        "body": style("body", alignment=TA_JUSTIFY, firstLineIndent=10),
        "center": style("center", alignment=TA_CENTER),
        "right": style("right", alignment=TA_RIGHT),
        "footnote": style("footnote", fontSize=8, leading=11,
                          textColor=colors.HexColor("#444444")),
        "cell": style("cell", fontSize=8, leading=10.5, spaceAfter=0,
                      firstLineIndent=0),
        "cellhead": style("cellhead", fontName="DejaVu-Bold", fontSize=8,
                          leading=10.5, spaceAfter=0, alignment=TA_CENTER,
                          firstLineIndent=0),
    }


STYLE_FOR_KIND = {
    "ACCEPTING_BODY": "subtitle",
    "ACT_FORM": "subtitle",
    "ACT_TITLE": "title",
    "ACT_ESSENTIAL_ELEMENTS": "center",
    "ACT_ESSENTIAL_ELEMENTS_NUM": "center",
    "TEXT_HEADER_DEFAULT": "heading",
    "ACT_TITLE_APPL": "heading",
    "APPL_BANNER_LANDSCAPE_TITLE": "appendix",
    "TEXT_CENTER": "center",
    "SIGNATURE": "right",
    "FOOTNOTE": "footnote",
}


def build_table(rec, styles, avail_width):
    rows = rec["rows"]
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    # Size columns by their content, but never so narrow that the header word
    # wraps mid-syllable -- numeric columns have long headers and tiny cells.
    widths = []
    for c in range(ncols):
        body_len = max((len(rows[r][c]) for r in range(1, len(rows))), default=0)
        head_len = min(len(rows[0][c]), 26)
        widths.append(max(body_len, head_len, 3))
    total = sum(widths)
    col_widths = [max(16 * mm, avail_width * w / total) for w in widths]
    scale = avail_width / sum(col_widths)
    col_widths = [w * scale for w in col_widths]

    data = [[Paragraph(html.escape(c), styles["cellhead" if i == 0 else "cell"])
             for c in row] for i, row in enumerate(rows)]

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#666666")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


def build_pdf(records: list[dict]):
    register_fonts()
    styles = make_styles()

    doc = BaseDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=20 * mm, rightMargin=15 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Vazirlar Mahkamasining 11.05.2026 yildagi 234-son qarori",
        author="Oʻzbekiston Respublikasi Vazirlar Mahkamasi",
        subject="Atrof-muhitga taʼsirni baholashning yangi mexanizmlarini "
                "joriy qilish chora-tadbirlari toʻgʻrisida",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(doc.leftMargin, 9 * mm, "lex.uz/uz/docs/-8193120")
        canvas.drawRightString(A4[0] - doc.rightMargin, 9 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])

    story = []
    for rec in records:
        if rec["type"] == "table":
            story.append(Spacer(1, 4))
            story.append(build_table(rec, styles, doc.width))
            story.append(Spacer(1, 6))
            continue
        style = styles[STYLE_FOR_KIND.get(rec["kind"], "body")]
        story.append(Paragraph(html.escape(rec["text"]).replace("\n", "<br/>"), style))

    doc.build(story)


def main():
    page = fetch()
    records = add_structure(parse(page))

    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    build_pdf(records)

    chars = sum(len(r["text"]) for r in records)
    tables = sum(1 for r in records if r["type"] == "table")
    ilovalar = sorted({r["ilova"] for r in records if r["ilova"]},
                      key=lambda s: int(s.split("-")[0]))
    meta = records[0].get("meta", {}) if records else {}
    print(f"act     : {meta.get('raqam','?')}-son, {meta.get('sana','?')}, "
          f"kuchga kirish {meta.get('kuchga_kirish','?')}")
    print(f"nodes   : {len(records)}")
    print(f"chars   : {chars:,}")
    print(f"tables  : {tables}")
    print(f"ilovalar: {', '.join(ilovalar)}")
    print(f"pdf     : {PDF_PATH}")
    print(f"jsonl   : {JSONL_PATH}")


if __name__ == "__main__":
    sys.exit(main())
