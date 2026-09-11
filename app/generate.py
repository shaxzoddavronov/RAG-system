"""Ollama generation with a forced JSON shape.

Two things here are load-bearing:

  * num_ctx is always sent. Ollama's default context is small and it truncates
    the prompt silently -- the retrieved passages vanish and the model answers
    from memory. That is the single most common hidden cause of hallucination
    in an Ollama RAG stack.
  * The response is constrained to a JSON schema, so "I could not find it" is a
    boolean field we can act on rather than prose we would have to pattern-match.
"""
from __future__ import annotations

import json
import re

import httpx

from app.config import Settings, settings as default_settings

SYSTEM_PROMPT = """Siz Oʻzbekiston Respublikasi Vazirlar Mahkamasining 2026-yil \
11-maydagi 234-son qarori boʻyicha savollarga javob beruvchi yordamchisiz.

QATʼIY QOIDALAR:
1. Faqat quyida berilgan MANBA BLOKLARI asosida javob bering. Oʻz bilimingizdan, \
boshqa qonun yoki hujjatlardan foydalanish QATʼIYAN taqiqlanadi.
2. Agar bloklarda savolga javob boʻlmasa yoki javob toʻliq boʻlmasa, \
"sufficient" maydonini false qilib belgilang va "answer" maydonini boʻsh qoldiring.
2a. DIQQAT: bloklar savol mavzusiga oid boʻlishi, lekin aynan soʻralgan \
maʼlumotni (masalan, aniq summa, bank nomi, shaxs ismi, sana yoki raqam) \
oʻz ichiga olmasligi mumkin. Bunday holatda ham "sufficient" = false. \
Savolga toʻgʻridan-toʻgʻri javob bermaydigan, faqat mavzuga yaqin \
maʼlumotni javob sifatida BERMANG.
3. "citations" maydoniga siz foydalangan bloklarning identifikatorini AYNAN \
kvadrat qavs ichidagi koʻrinishda yozing, masalan: "-8205881#0". Identifikator \
"#" belgisi va raqam bilan tugaydi — ularni TUSHIRIB QOLDIRMANG. Hech qanday \
qoʻshimcha soʻz yoki belgi qoʻshmang.
4. "quotes" maydoniga javobingizni tasdiqlovchi jumlalarni bloklardan SOʻZMA-SOʻZ \
koʻchiring. Birorta ham soʻzni oʻzgartirmang va "..." bilan qisqartirmang.
4a. Iqtiboslar QISQA boʻlsin — javobni tasdiqlash uchun yetarli boʻlgan eng \
kichik qismni oling (odatda bir necha soʻz). Uzun jumlani toʻliq koʻchirishda \
xato qilish ehtimoli yuqori.
5. Javobni oʻzbek tilida, aniq va qisqa yozing. Raqamlar, muddatlar va toʻlov \
miqdorlarini blokdagidek aynan keltiring."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sufficient": {"type": "boolean"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "quotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "sufficient", "citations", "quotes"],
}


def build_prompt(question: str, blocks: list[dict]) -> str:
    parts = ["MANBA BLOKLARI:", ""]
    for block in blocks:
        parts.append(f"[{block['cite_id']}] {block['path']}")
        parts.append(block["text"])
        parts.append("")
    parts.append(f"SAVOL: {question}")
    return "\n".join(parts)


_CITATION_NOISE = re.compile(r"^[\s\[\(]*(?:node_?id\s*[:=]\s*)?|[\s\]\),.]*$",
                             re.IGNORECASE)


def _clean_citation(value) -> str:
    """Strip decoration a model may wrap around an id ("node_id: -820...", "[-820...]").

    The ids are ours, so being liberal in what we accept here costs nothing --
    gate 4 still checks the result against the ids we actually retrieved.
    """
    return _CITATION_NOISE.sub("", str(value)).strip()


class OllamaError(RuntimeError):
    """Transport-level failure: Ollama unreachable, timed out, HTTP error."""


class OllamaOutputError(OllamaError):
    """Ollama replied, but the body was not usable JSON.

    Usually means generation hit num_predict mid-object. Treated as "no
    trustworthy answer" rather than as an outage, so the caller refuses
    instead of returning 503.
    """


class Generator:
    def __init__(self, settings: Settings = default_settings):
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.ollama_host.rstrip("/"),
                timeout=self.settings.request_timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def reachable(self) -> tuple[bool, bool]:
        """(ollama responds, requested model is pulled)"""
        try:
            client = await self.client()
            resp = await client.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
        except Exception:
            return False, False
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        wanted = self.settings.ollama_model
        present = any(n == wanted or n.split(":")[0] == wanted.split(":")[0] for n in names)
        return True, present

    async def generate(self, question: str, blocks: list[dict],
                       temperature: float | None = None,
                       seed: int | None = None) -> dict:
        payload = {
            "model": self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, blocks)},
            ],
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "think": self.settings.think,
            "options": {
                "temperature": self.settings.temperature if temperature is None else temperature,
                "seed": self.settings.seed if seed is None else seed,
                "num_ctx": self.settings.num_ctx,
                "num_predict": self.settings.num_predict,
            },
        }
        client = await self.client()
        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        content = resp.json().get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaOutputError(
                f"model did not return usable JSON (likely truncated at "
                f"num_predict={self.settings.num_predict}): {content[:120]}") from exc

        return {
            "answer": str(parsed.get("answer", "")).strip(),
            "sufficient": bool(parsed.get("sufficient", False)),
            "citations": [_clean_citation(c) for c in parsed.get("citations", []) or []],
            "quotes": [str(q) for q in parsed.get("quotes", []) or []],
        }
