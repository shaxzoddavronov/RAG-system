#!/usr/bin/env python3
"""Generate eval/gold.jsonl deterministically from the built index.

Answerable questions are derived from real chunks so the expected retrieval
target is never guesswork. Unanswerable ones are deliberately adjacent -- they
use the document's own vocabulary (ekologik, ekspertiza, muddat) but ask about
things Resolution 234 does not regulate. Easy off-topic questions would make
the refusal metric look better than it is.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings

OUT = Path(__file__).resolve().parent.parent / "eval" / "gold.jsonl"
random.seed(234)

# Plausible but absent: adjacent legal topics, other statutes, invented details.
UNANSWERABLE = [
    "Ekologik ekspertizani oʻtkazmaganlik uchun jarima miqdori qancha?",
    "Jinoyat kodeksining 196-moddasida qanday jazo belgilangan?",
    "Atrof-muhitni ifloslantirgan korxona rahbari qamoq jazosiga tortiladimi?",
    "Ekologiya qoʻmitasi raisining ismi kim?",
    "Ushbu qaror Qozogʻiston Respublikasida ham amal qiladimi?",
    "Davlat ekologik ekspertizasi uchun toʻlov qaysi bankka oʻtkaziladi?",
    "Ekolog-ekspertlarning oylik ish haqi qancha?",
    "Malaka sertifikati uchun davlat boji necha soʻm?",
    "Toshkent shahrida nechta ekologik ekspertiza markazi bor?",
    "Ushbu qaror qaysi deputat tomonidan taklif qilingan?",
    "Atrof-muhitga taʼsirni baholash boʻyicha xalqaro ISO standarti raqami qanday?",
    "Korxona bankrot boʻlsa ekologik majburiyatlar kimga oʻtadi?",
    "Ekologik sugʻurta shartnomasi qanday tuziladi?",
    "Soliq kodeksida ekologik soliq stavkasi qancha?",
    "Ushbu qarorga qarshi sudga shikoyat qilish muddati necha kun?",
    "Chet el investorlari uchun alohida imtiyozlar bormi?",
    "Ekologik ekspertiza xulosasi qaysi tillarda beriladi?",
    "Qaror matni Oliy Majlis tomonidan tasdiqlanganmi?",
    "Ekologiya vazirligining manzili qayerda joylashgan?",
    "Ushbu qaror necha nusxada chop etilgan?",
    "Atom elektr stansiyasi qurilishi uchun qanday maxsus tartib bor?",
    "Ekologik auditorlik faoliyati qanday litsenziyalanadi?",
    "Qaror boʻyicha davlat byudjetidan qancha mablagʻ ajratilgan?",
    "Ekspertiza natijalari qaysi axborot tizimida eʼlon qilinadi?",
    "Ushbu hujjat qaysi xalqaro konvensiyaga asoslangan?",
]

# Hand-written. Anchored by a distinctive phrase rather than a node id: a
# mistyped id silently points at the wrong chunk and quietly corrupts the
# metrics, whereas a phrase that matches nothing fails loudly at build time.
PROSE = [
    ("Malaka sertifikati necha yil muddatga beriladi?",
     "Malaka sertifikati uch yil muddatga beriladi"),
    # Reported by a user as a wrong refusal. The fact was indexed and retrieved
    # correctly; the model mistyped one morpheme while transcribing its quote
    # and gate 3 threw the whole answer away. Kept as a regression.
    ("Ekologik ekspertiza obyektlari necha toifaga boʻlinadi?",
     "uch toifaga ajratiladi"),
    ("Mazkur qaror qachon kuchga kiradi?",
     "rasmiy eʼlon qilingan kundan eʼtiboran uch oy"),
    ("Jamoat ekologik ekspertizasi xulosasi nusxalari necha yil saqlanadi?",
     "xulosasi nusxalari tashabbuskor va obyektda kamida besh yil"),
    ("Reyestrga kiritish toʻgʻrisidagi ariza toʻlovlimi yoki bepulmi?",
     "Reyestrga kiritish toʻgʻrisidagi ariza bepul"),
    ("Loyihani ishlab chiquvchining xodimi qanchadan keyin malaka oshirishi shart?",
     "har ikki yilda bir marta malaka oshirish kurslarida"),
    ("Jamoatchilik eshituvi bayonnomalari nusxalari qancha muddat saqlanadi?",
     "bayonnomalarining nusxalari tashabbuskorda kamida bir yil"),
    ("Strategik ekologik baholash har bir obyekt boʻyicha necha marta oʻtkaziladi?",
     "obyekti boʻyicha bir marta oʻtkaziladi"),
    ("Reyestr maʼlumotlari Ekologiya qoʻmitasida qancha saqlanadi?",
     "Ekologiya qoʻmitasida besh yil mobaynida saqlanadi"),
]


def table_questions(chunks, count=22):
    rows = [c for c in chunks
            if c["kind"] == "TABLE_STD2" and c.get("ilova") == "1-ilova"
            and c.get("extra", {}).get("Ekspertizani oʻtkazish muddati (ish kuni)")]
    picked = random.sample(rows, min(count, len(rows)))
    out = []
    for chunk in picked:
        subject = re.sub(r"^.*?\d+-band:\s*", "", chunk["text"]).split(". Eksperti")[0]
        subject = subject.strip().rstrip(".")
        if len(subject) < 25 or len(subject) > 170:
            continue
        out.append({
            "question": f"{subject} uchun davlat ekologik ekspertizasi muddati necha ish kuni?",
            "answerable": True,
            "expect_chunk": chunk["chunk_id"],
            "expect_node": chunk["node_id"],
            "category": "table",
        })
    return out


def main() -> int:
    chunks = [json.loads(l) for l in (settings.index_dir / "chunks.jsonl").open(encoding="utf-8")]
    by_node = {}
    for c in chunks:
        by_node.setdefault(c["node_id"], c)

    items = []
    missing = []
    for question, anchor in PROSE:
        match = next((c for c in chunks if anchor in c["text"]), None)
        if match is None:
            missing.append(anchor)
            continue
        items.append({"question": question, "answerable": True,
                      "expect_chunk": match["chunk_id"],
                      "expect_node": match["node_id"], "category": "prose"})
    if missing:
        print("anchors matched nothing -- fix these before trusting the metrics:",
              file=sys.stderr)
        for anchor in missing:
            print(f"  {anchor!r}", file=sys.stderr)
        return 1

    items += table_questions(chunks)

    # Prose bands that state a duration or count: unambiguous retrieval targets.
    pool = [c for c in chunks
            if c["kind"] != "TABLE_STD2" and 90 < len(c["text"]) < 400
            and re.search(r"\b(kun|yil|oy|marta|foiz)\b", c["text"])
            and c["node_id"] not in {n for _, n in PROSE}]
    for chunk in random.sample(pool, min(25, len(pool))):
        head = re.sub(r"^\d+\.\s*", "", chunk["text"])
        subject = " ".join(head.split()[:9]).rstrip(".,;:")
        items.append({
            "question": f"{subject} boʻyicha qarorda nima belgilangan?",
            "answerable": True, "expect_chunk": chunk["chunk_id"],
            "expect_node": chunk["node_id"], "category": "prose-auto",
        })

    items += [{"question": q, "answerable": False, "expect_chunk": None,
               "expect_node": None, "category": "unanswerable"} for q in UNANSWERABLE]

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    answerable = sum(1 for i in items if i["answerable"])
    print(f"total       : {len(items)}")
    print(f"  answerable: {answerable}")
    print(f"  refusable : {len(items) - answerable}")
    print(f"written to  : {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
