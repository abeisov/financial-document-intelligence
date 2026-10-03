"""Нормализация результатов RAGFlow, внешний reranking и grounded-ответ Qwen."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

from .answering import INSUFFICIENT, numbers_in
from .clients import ServiceError, chat_json, rerank
from .config import Settings

ALLOWED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".txt", ".md", ".csv",
    ".xls", ".xlsx", ".ppt", ".pptx", ".png", ".jpg", ".jpeg",
}
#: Vercel ограничивает тело запроса к функции 4,5 МБ, поэтому по умолчанию 4 МБ.
#: Для локального запуска лимит поднимается переменной MAX_UPLOAD_MB.
DEFAULT_MAX_UPLOAD_MB = 4


def max_upload_mb() -> int:
    raw = os.getenv("MAX_UPLOAD_MB", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_UPLOAD_MB
    return value if value > 0 else DEFAULT_MAX_UPLOAD_MB


def max_upload_bytes() -> int:
    return max_upload_mb() * 1024 * 1024


@dataclass(frozen=True)
class Source:
    source_id: str
    text: str
    document: str
    page: str
    ragflow_score: float
    rerank_score: float | None = None

    def public(self, *, include_excerpt: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_id": self.source_id,
            "document": self.document,
            "page": self.page,
            "ragflow_score": round(self.ragflow_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
        }
        if include_excerpt:
            result["excerpt"] = re.sub(r"\s+", " ", self.text).strip()[:420]
        return result


def validate_upload(filename: str, content: bytes) -> str:
    safe_name = Path(filename or "").name
    if not safe_name:
        raise ValueError("Файл должен иметь имя")
    if Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Неподдерживаемый формат. Разрешены: {allowed}")
    if not content:
        raise ValueError("Файл пуст")
    limit = max_upload_bytes()
    if len(content) > limit:
        raise ValueError(f"Файл больше {max_upload_mb()} МБ")
    return safe_name


def _text(row: dict[str, Any]) -> str:
    for key in ("content_with_weight", "content", "text", "content_ltks"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _page_label(row: dict[str, Any]) -> str:
    raw = row.get("page_num_int", row.get("page_num", row.get("page")))
    if raw in (None, ""):
        positions = row.get("positions")
        if isinstance(positions, list) and positions:
            first = positions[0]
            if isinstance(first, list) and first:
                raw = first[0]
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw in (None, ""):
        return "не указана"
    return str(raw)


def normalize_ragflow_chunks(rows: list[dict[str, Any]]) -> list[Source]:
    sources: list[Source] = []
    seen: set[str] = set()
    for position, row in enumerate(rows, start=1):
        text = _text(row)
        if not text:
            continue
        source_id = str(row.get("id") or row.get("chunk_id") or f"ragflow-{position}")
        if source_id in seen:
            continue
        seen.add(source_id)
        score = row.get("similarity", row.get("score", row.get("vector_similarity", 0.0)))
        try:
            numeric_score = float(score or 0.0)
        except (TypeError, ValueError):
            numeric_score = 0.0
        sources.append(Source(
            source_id=source_id,
            text=text,
            document=str(
                row.get("document_name")
                or row.get("document_keyword")
                or row.get("docnm_kwd")
                or "Загруженный документ"
            ),
            page=_page_label(row),
            ragflow_score=numeric_score,
        ))
    return sources


def select_sources(settings: Settings, question: str, sources: list[Source], top_n: int = 5) -> list[Source]:
    if not sources:
        return []
    ranked = rerank(
        settings.reranker,
        question,
        [source.text for source in sources],
        top_n=min(top_n, len(sources)),
    )
    selected: list[Source] = []
    for row in ranked[:top_n]:
        idx = row["index"]
        if idx < 0 or idx >= len(sources):
            raise ServiceError("reranker: индекс результата выходит за границы списка")
        original = sources[idx]
        selected.append(Source(
            source_id=original.source_id,
            text=original.text,
            document=original.document,
            page=original.page,
            ragflow_score=original.ragflow_score,
            rerank_score=row["score"],
        ))
    return selected


SYSTEM_PROMPT = f"""Ты финансовый аналитик. Отвечай только по переданным источникам.
Запрещено придумывать числа, причины, факты и рекомендации, отсутствующие в источниках.
Отделяй факт из документа от интерпретации. Каждый существенный вывод должен иметь source_id.
Если данных недостаточно, верни sufficient=false и answer=\"{INSUFFICIENT}\".
Не давай инвестиционных или кредитных рекомендаций. Отвечай на русском языке.
Верни только JSON без markdown."""


def _context(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[source_id={source.source_id} | документ={source.document} | страница={source.page}]\n{source.text}"
        for source in sources
    )


def _quote_fingerprint(value: str) -> str:
    plain = unescape(re.sub(r"<[^>]+>", " ", value))
    return re.sub(r"[^\w]+", "", plain.casefold())


def answer_from_sources(settings: Settings, question: str, sources: list[Source]) -> dict[str, Any]:
    if not sources:
        return {
            "sufficient": False,
            "answer": INSUFFICIENT,
            "citations": [],
            "sources": [],
            "grounding": {"citations_valid": True, "numbers_grounded": True},
        }
    raw = chat_json(
        settings.qwen,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Источники:

{_context(sources)}

Вопрос: {question}

Верни JSON:
{{
  "sufficient": true | false,
  "answer": "краткий ответ по источникам",
  "observed_facts": ["факт из документа"],
  "interpretation": ["осторожная интерпретация, если она обоснована"],
  "citations": [{{"source_id": "точный ID источника", "quote": "короткая цитата"}}]
}}""",
            },
        ],
        max_tokens=1200,
    )

    valid = {source.source_id: source for source in sources}
    citations = [item for item in (raw.get("citations") or []) if isinstance(item, dict)]
    invalid_ids: list[str] = []
    quote_errors: list[str] = []
    clean_citations: list[dict[str, str]] = []
    for citation in citations:
        source_id = str(citation.get("source_id") or "")
        quote = re.sub(r"\s+", " ", str(citation.get("quote") or "")).strip()[:240]
        source = valid.get(source_id)
        if source is None:
            invalid_ids.append(source_id)
            continue
        normalized_text = _quote_fingerprint(source.text)
        if quote and _quote_fingerprint(quote) not in normalized_text:
            quote_errors.append(source_id)
        clean_citations.append({
            "source_id": source_id,
            "document": source.document,
            "page": source.page,
            "quote": quote,
        })

    checked_text = " ".join([
        str(raw.get("answer") or ""),
        " ".join(str(x) for x in (raw.get("observed_facts") or [])),
        " ".join(str(x) for x in (raw.get("interpretation") or [])),
    ])
    context_numbers = numbers_in(" ".join(source.text for source in sources))
    unsupported_numbers = sorted(numbers_in(checked_text) - context_numbers)
    sufficient = bool(raw.get("sufficient")) and str(raw.get("answer")) != INSUFFICIENT
    if sufficient and not clean_citations:
        sufficient = False

    return {
        "sufficient": sufficient,
        "answer": str(raw.get("answer") or INSUFFICIENT).strip(),
        "observed_facts": [str(x) for x in (raw.get("observed_facts") or [])],
        "interpretation": [str(x) for x in (raw.get("interpretation") or [])],
        "citations": clean_citations,
        "sources": [source.public() for source in sources],
        "grounding": {
            "citations_valid": not invalid_ids,
            "invalid_source_ids": invalid_ids,
            "quotes_found": not quote_errors,
            "quote_errors": quote_errors,
            "numbers_grounded": not unsupported_numbers,
            "unsupported_numbers": unsupported_numbers,
        },
    }
