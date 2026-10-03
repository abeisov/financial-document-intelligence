"""Извлечение текста из цифрового PDF с сохранением номеров страниц."""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from pypdf import PdfReader

# Минимум символов на странице, ниже которого страница считается несловесной
# (скан/изображение) и требует отдельной OCR-проверки.
DIGITAL_PAGE_MIN_CHARS = 40


@dataclass(frozen=True)
class Page:
    page: int  # нумерация с 1, как видит пользователь
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


def _normalize(text: str) -> str:
    text = text.replace(" ", " ").replace("‑", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def extract_pages(pdf_path: pathlib.Path) -> list[Page]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(Page(page=index, text=_normalize(page.extract_text() or "")))
    return pages


def digital_report(pages: list[Page]) -> dict:
    empty = [p.page for p in pages if p.chars < DIGITAL_PAGE_MIN_CHARS]
    return {
        "pages_total": len(pages),
        "chars_total": sum(p.chars for p in pages),
        "chars_per_page": {str(p.page): p.chars for p in pages},
        "pages_without_text_layer": empty,
        "is_digital": not empty,
    }
