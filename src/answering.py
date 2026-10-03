"""Ответы Qwen строго по отобранным фрагментам, с проверкой обоснованности."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .clients import chat_json
from .config import Service
from .retrieval import Hit

INSUFFICIENT = "insufficient_evidence"

# Цитата — подтверждение ответа, а не выгрузка документа.
MAX_QUOTE_CHARS = 240

SYSTEM_PROMPT = (
    "Ты финансовый аналитик. Ты отвечаешь ИСКЛЮЧИТЕЛЬНО по предоставленным фрагментам документа.\n"
    "Жёсткие правила:\n"
    "1. Запрещено использовать любые знания вне фрагментов: отраслевые ориентиры, оценки, "
    "предположения о причинах, которых нет в тексте.\n"
    "2. Запрещено вычислять новые числа. Используй только те числа, которые дословно "
    "присутствуют во фрагментах.\n"
    "3. Каждый ответ обязан ссылаться минимум на один chunk_id и его страницу.\n"
    f"4. Если во фрагментах нет данных для ответа, верни sufficient=false и answer=\"{INSUFFICIENT}\".\n"
    "5. Отвечай на русском языке. Возвращай только JSON без markdown-обёртки."
)

USER_TEMPLATE = """Фрагменты документа:

{context}

Вопрос: {question}

Верни JSON строго такой структуры:
{{
  "sufficient": true | false,
  "answer": "2-4 предложения по существу вопроса, с числами из фрагментов",
  "metrics": [
    {{"name": "название показателя", "prev": "значение за ранний период",
      "current": "значение за поздний период", "change": "изменение как указано во фрагменте"}}
  ],
  "citations": [{{"chunk_id": "идентификатор фрагмента", "page": номер_страницы}}],
  "quote": "дословная цитата из фрагмента, подтверждающая ответ"
}}"""

# Числовые токены: 1 325 / 27,5% / (143) / 2,5x / -38.
# Пробел допускается ТОЛЬКО как разделитель разрядов между группами из трёх цифр,
# иначе числа из соседних строк таблицы склеиваются в одно бессмысленное значение.
_NUMBER = re.compile(
    r"[-+(]?\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?\)?(?:\s?(?:%|x|х))?", re.IGNORECASE
)


def _normalize_number(token: str) -> str:
    t = token.strip().lower()
    t = re.sub(r"[\s\u00a0]+", "", t)
    t = t.replace("(", "").replace(")", "")
    t = t.lstrip("+-")
    t = t.rstrip("%xх")
    t = t.replace(",", ".")
    if t.endswith("."):
        t = t[:-1]
    try:
        value = float(t)
    except ValueError:
        return ""
    return f"{value:g}"


def numbers_in(text: str) -> set[str]:
    found = {_normalize_number(m.group(0)) for m in _NUMBER.finditer(text)}
    return {n for n in found if n}


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            f"[chunk_id={hit.chunk.chunk_id} | страница {hit.chunk.page}]\n{hit.chunk.text}"
        )
    return "\n\n".join(blocks)


@dataclass
class Answer:
    question: str
    key: str
    raw: dict
    context_chunk_ids: list[str]
    context_pages: list[int]
    grounding: dict

    @property
    def sufficient(self) -> bool:
        return bool(self.raw.get("sufficient")) and self.raw.get("answer") != INSUFFICIENT

    @property
    def text(self) -> str:
        return str(self.raw.get("answer", "")).strip()

    @property
    def citations(self) -> list[dict]:
        cites = self.raw.get("citations") or []
        return [c for c in cites if isinstance(c, dict)]

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "question": self.question,
            "sufficient": self.sufficient,
            "answer": self.text,
            "metrics": self.raw.get("metrics") or [],
            "citations": self.citations,
            "quote": str(self.raw.get("quote", "")).strip()[:MAX_QUOTE_CHARS],
            "context_chunk_ids": self.context_chunk_ids,
            "context_pages": self.context_pages,
            "grounding": self.grounding,
        }


def _check_grounding(raw: dict, hits: list[Hit]) -> dict:
    context_text = " ".join(h.chunk.text for h in hits)
    valid_ids = {h.chunk.chunk_id: h.chunk.page for h in hits}

    answer_text = str(raw.get("answer", ""))
    metrics_text = " ".join(
        f"{m.get('name','')} {m.get('prev','')} {m.get('current','')} {m.get('change','')}"
        for m in (raw.get("metrics") or [])
        if isinstance(m, dict)
    )
    checked_text = f"{answer_text} {metrics_text}"

    context_numbers = numbers_in(context_text)
    # Номера страниц документа допустимы как ссылки на источник.
    allowed = context_numbers | {str(p) for p in sorted({h.chunk.page for h in hits})}
    unsupported = sorted(n for n in numbers_in(checked_text) if n not in allowed)

    citations = [c for c in (raw.get("citations") or []) if isinstance(c, dict)]
    bad_citations = []
    for cite in citations:
        cid = str(cite.get("chunk_id", ""))
        if cid not in valid_ids:
            bad_citations.append({"chunk_id": cid, "reason": "chunk_id отсутствует в контексте"})
            continue
        page = cite.get("page")
        if page is not None and int(page) != valid_ids[cid]:
            bad_citations.append(
                {"chunk_id": cid, "reason": f"страница {page} не совпадает с {valid_ids[cid]}"}
            )

    quote = str(raw.get("quote", "")).strip()
    normalized_context = re.sub(r"\s+", " ", context_text).lower()
    quote_found = bool(quote) and re.sub(r"\s+", " ", quote).lower() in normalized_context

    return {
        "has_citation": bool(citations),
        "citations_valid": not bad_citations,
        "invalid_citations": bad_citations,
        "quote_found_in_context": quote_found,
        "unsupported_numbers": unsupported,
        "numbers_grounded": not unsupported,
    }


def answer_question(service: Service, key: str, question: str, hits: list[Hit]) -> Answer:
    raw = chat_json(
        service,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                context=format_context(hits), question=question)},
        ],
        max_tokens=1100,
    )
    return Answer(
        question=question,
        key=key,
        raw=raw,
        context_chunk_ids=[h.chunk.chunk_id for h in hits],
        context_pages=sorted({h.chunk.page for h in hits}),
        grounding=_check_grounding(raw, hits),
    )
