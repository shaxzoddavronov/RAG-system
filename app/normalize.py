"""Uzbek text normalization, shared by the corpus builder, the index and queries.

Uzbek Latin writes oʻ/gʻ with U+02BB and the glottal stop with U+02BC, but real
users type ASCII apostrophes or typographic quotes, and lex.uz is not consistent
either. Documents and queries must pass through the same function or BM25 silently
stops matching. Everything here is deliberately dependency-free.
"""
import re
import unicodedata

# Apostrophes are ambiguous in Uzbek Latin: a typed ' means the letter ʻ (U+02BB)
# in oʻ/gʻ, but the glottal stop ʼ (U+02BC) everywhere else. Resolve by context,
# otherwise "g'alaba" normalizes to gʼalaba and never matches the corpus.
ANY_APOSTROPHE = re.compile(r"[\u2018\u2019`\u00b4\u02b9'\u02bb\u02bc]")
_OG_APOSTROPHE = re.compile(r"([oOgG])[\u02bb\u02bc]")

# Uzbek Cyrillic -> Latin. Longest keys first so digraphs win.
CYRILLIC_TO_LATIN = {
    "ё": "yo", "ж": "j", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
    "ю": "yu", "я": "ya", "ў": "oʻ", "ғ": "gʻ", "қ": "q", "ҳ": "h",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x",
    "ъ": "ʼ", "ь": "", "э": "e",
}

# Abbreviations and synonyms used in this act. Expanding them at query time is
# cheap and matters a lot: the document always spells the terms out in full.
GLOSSARY = {
    "dee": "davlat ekologik ekspertizasi",
    "atb": "atrof-muhitga taʼsirni baholash",
    "seb": "strategik ekologik baholash",
    "bhm": "bazaviy hisoblash miqdori",
    "jee": "jamoat ekologik ekspertizasi",
}

_WS = re.compile(r"[ \t ]+")
_PUNCT = re.compile(r"[^\w\sʻʼ]", re.UNICODE)


def normalize(text: str) -> str:
    """Canonical form: NFC, context-correct apostrophes, collapsed whitespace.

    Case is preserved -- this is the form used for display and for the verbatim
    quote check in gates.py.
    """
    text = unicodedata.normalize("NFC", text)
    text = ANY_APOSTROPHE.sub("\u02bc", text)      # default to the glottal stop
    text = _OG_APOSTROPHE.sub("\\1ʻ", text)   # ...but oʻ/gʻ take the letter
    text = text.replace("\xa0", " ")
    return _WS.sub(" ", text).strip()


def fold(text: str) -> str:
    """Aggressive form for lexical matching.

    Both apostrophes collapse to one character here. Users are inconsistent about
    the oʻ/taʼ distinction even when typing real Uzbek, so BM25 must not be able
    to miss on it.
    """
    text = normalize(text).lower()
    text = text.replace("\u02bb", "\u02bc")
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def to_latin(text: str) -> str:
    """Transliterate Uzbek Cyrillic to Latin. Latin input passes through unchanged."""
    out = []
    for ch in normalize(text):
        lower = ch.lower()
        if lower in CYRILLIC_TO_LATIN:
            mapped = CYRILLIC_TO_LATIN[lower]
            out.append(mapped.capitalize() if ch.isupper() and mapped else mapped)
        else:
            out.append(ch)
    return "".join(out)


def has_cyrillic(text: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in text)


def expand_query(query: str) -> list[str]:
    """Query plus its transliteration and any glossary expansions, deduplicated.

    Returned in priority order; the first entry is always the canonical query.
    """
    variants = [normalize(query)]
    if has_cyrillic(query):
        variants.append(to_latin(query))

    folded = fold(variants[-1])
    expanded = folded
    for abbr, full in GLOSSARY.items():
        if re.search(rf"\b{abbr}\b", folded):
            expanded = re.sub(rf"\b{abbr}\b", full, expanded)
    if expanded != folded:
        variants.append(expanded)

    seen, unique = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            unique.append(v)
    return unique
