"""Оффлайн-проверки web-цепочки без обращений к внешним API."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.web_analysis import (
    Source,
    answer_from_sources,
    normalize_ragflow_chunks,
    select_sources,
    validate_upload,
)


def test_validate_upload_accepts_supported_document_and_strips_path():
    assert validate_upload("../../report.pdf", b"content") == "report.pdf"


@pytest.mark.parametrize("filename", ["report.exe", "archive.zip", "no-extension"])
def test_validate_upload_rejects_unsupported_extensions(filename):
    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        validate_upload(filename, b"content")


def test_validate_upload_rejects_empty_file():
    with pytest.raises(ValueError, match="пуст"):
        validate_upload("report.pdf", b"")


def test_normalize_ragflow_chunks_uses_realistic_schema_and_deduplicates():
    rows = [
        {
            "id": "chunk-1",
            "content_with_weight": "Выручка выросла до 6 145 млн тенге.",
            "document_keyword": "report.pdf",
            "positions": [[2, 62, 517, 105, 287]],
            "similarity": 0.91,
        },
        {
            "id": "chunk-1",
            "content": "Дубликат",
            "document_name": "report.pdf",
            "page_num": 2,
        },
        {"id": "empty", "content": ""},
    ]

    sources = normalize_ragflow_chunks(rows)

    assert len(sources) == 1
    assert sources[0].source_id == "chunk-1"
    assert sources[0].document == "report.pdf"
    assert sources[0].page == "2"
    assert sources[0].ragflow_score == pytest.approx(0.91)


def test_select_sources_applies_external_reranker(monkeypatch):
    sources = [
        Source("a", "погода", "report.pdf", "1", 0.8),
        Source("b", "выручка выросла", "report.pdf", "2", 0.7),
    ]
    monkeypatch.setattr(
        "src.web_analysis.rerank",
        lambda *_args, **_kwargs: [
            {"index": 1, "score": 0.99},
            {"index": 0, "score": 0.12},
        ],
    )

    selected = select_sources(
        SimpleNamespace(reranker=object()),
        "Как изменилась выручка?",
        sources,
        top_n=2,
    )

    assert [source.source_id for source in selected] == ["b", "a"]
    assert selected[0].rerank_score == pytest.approx(0.99)


def test_answer_requires_valid_citation_and_grounded_numbers(monkeypatch):
    source = Source(
        "chunk-2",
        "<table><tr><td>Выручка выросла с 4 820 до 6 145 млн тенге.</td></tr></table>",
        "report.pdf",
        "2",
        0.9,
        0.98,
    )
    monkeypatch.setattr(
        "src.web_analysis.chat_json",
        lambda *_args, **_kwargs: {
            "sufficient": True,
            "answer": "Выручка выросла с 4 820 до 6 145 млн тенге.",
            "observed_facts": ["Выручка достигла 6 145 млн тенге."],
            "interpretation": ["Наблюдается рост выручки."],
            "citations": [
                {
                    "source_id": "chunk-2",
                    "quote": "Выручка выросла с 4 820 до 6 145 млн тенге.",
                }
            ],
        },
    )

    result = answer_from_sources(
        SimpleNamespace(qwen=object()),
        "Как изменилась выручка?",
        [source],
    )

    assert result["sufficient"] is True
    assert result["grounding"]["citations_valid"] is True
    assert result["grounding"]["quotes_found"] is True
    assert result["grounding"]["numbers_grounded"] is True


def test_answer_marks_invented_number(monkeypatch):
    source = Source("chunk-2", "Выручка 6 145 млн тенге.", "report.pdf", "2", 0.9)
    monkeypatch.setattr(
        "src.web_analysis.chat_json",
        lambda *_args, **_kwargs: {
            "sufficient": True,
            "answer": "Выручка составила 9 999 млн тенге.",
            "observed_facts": [],
            "interpretation": [],
            "citations": [{"source_id": "chunk-2", "quote": "Выручка 6 145 млн тенге."}],
        },
    )

    result = answer_from_sources(
        SimpleNamespace(qwen=object()),
        "Выручка?",
        [source],
    )

    assert result["grounding"]["numbers_grounded"] is False
    assert "9999" in result["grounding"]["unsupported_numbers"]
