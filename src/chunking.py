"""Разделение постраничного текста на фрагменты с сохранением страницы."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .extract import Page

TARGET_CHARS = 700
OVERLAP_CHARS = 140


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split_page(text: str, target: int, overlap: int) -> list[str]:
    """Режет по строкам, не разрывая строку таблицы посередине."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    parts: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) + 1 > target and buffer:
            parts.append("\n".join(buffer))
            # перекрытие: переносим хвостовые строки в следующий фрагмент
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(buffer):
                if tail_size + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            buffer = list(tail)
            size = tail_size
        buffer.append(line)
        size += len(line) + 1
    if buffer:
        parts.append("\n".join(buffer))
    return parts


def build_chunks(pages: list[Page], target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for order, part in enumerate(_split_page(page.text, target, overlap)):
            chunks.append(Chunk(chunk_id=f"p{page.page}_c{order}", page=page.page, text=part))
    return chunks
