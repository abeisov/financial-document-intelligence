"""Тесты цепочки. Оффлайн-часть запускается всегда, live-часть — по RUN_LIVE=1."""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from src.answering import _check_grounding, numbers_in
from src.chunking import build_chunks
from src.config import ARTIFACTS, DATA
from src.extract import Page, digital_report, extract_pages
from src.retrieval import Hit, cosine

PDF = DATA / "test_financial_report.pdf"
GT = DATA / "ground_truth.json"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def pages() -> list[Page]:
    if not PDF.exists():
        pytest.skip("нет фикстуры: python scripts/make_test_pdf.py")
    return extract_pages(PDF)


@pytest.fixture(scope="module")
def ground_truth() -> dict:
    if not GT.exists():
        pytest.skip("нет ground_truth.json")
    return json.loads(GT.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ извлечение
def test_pdf_is_digital_and_within_size_limits(pages):
    report = digital_report(pages)
    assert 3 <= report["pages_total"] <= 10
    assert report["is_digital"], report["pages_without_text_layer"]


def test_known_values_land_on_expected_pages(pages, ground_truth):
    by_page = {p.page: p.text.replace(" ", " ") for p in pages}
    for key, check in ground_truth["checks"].items():
        for value in check["values"].values():
            assert value in by_page[check["page"]], f"{key}: {value} не на стр. {check['page']}"


# ------------------------------------------------------------------ chunking
def test_chunks_keep_page_and_unique_id(pages):
    chunks = build_chunks(pages)
    assert chunks
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.page for c in chunks} == {p.page for p in pages if p.text}
    for chunk in chunks:
        assert chunk.chunk_id.startswith(f"p{chunk.page}_c")
        assert chunk.text.strip()


def test_chunk_overlap_does_not_lose_lines(pages):
    page = pages[1]
    joined = "\n".join(c.text for c in build_chunks([page]))
    for line in page.text.splitlines():
        if line.strip():
            assert line in joined


# ------------------------------------------------------------------ числа
def test_table_numbers_are_not_merged_across_lines():
    assert numbers_in("Выручка\n6 145\n4 820\n+27,5%") == {"6145", "4820", "27.5"}


def test_number_formats():
    assert numbers_in("поток (143) при 2,5x и 1 325") == {"143", "2.5", "1325"}


# ------------------------------------------------------------------ grounding
def _hit(chunk_id: str, page: int, text: str) -> Hit:
    from src.chunking import Chunk

    return Hit(chunk=Chunk(chunk_id=chunk_id, page=page, text=text), embedding_score=1.0, embedding_rank=1)


def test_grounding_flags_invented_number():
    hits = [_hit("p2_c0", 2, "Выручка\n6 145\n4 820")]
    result = _check_grounding(
        {"answer": "Выручка выросла до 9 999 млн тенге", "citations": [{"chunk_id": "p2_c0", "page": 2}]},
        hits,
    )
    assert result["numbers_grounded"] is False
    assert "9999" in result["unsupported_numbers"]


def test_grounding_accepts_numbers_from_context():
    hits = [_hit("p2_c0", 2, "Выручка\n6 145\n4 820\n+27,5%")]
    result = _check_grounding(
        {
            "answer": "Выручка выросла с 4 820 до 6 145 млн тенге (+27,5%).",
            "citations": [{"chunk_id": "p2_c0", "page": 2}],
            "quote": "Выручка 6 145 4 820",
        },
        hits,
    )
    assert result["numbers_grounded"] is True
    assert result["has_citation"] and result["citations_valid"]


def test_grounding_rejects_unknown_chunk_id():
    hits = [_hit("p2_c0", 2, "Выручка 6 145")]
    result = _check_grounding({"answer": "6 145", "citations": [{"chunk_id": "p9_c9", "page": 9}]}, hits)
    assert result["citations_valid"] is False


def test_grounding_rejects_page_mismatch():
    hits = [_hit("p2_c0", 2, "Выручка 6 145")]
    result = _check_grounding({"answer": "6 145", "citations": [{"chunk_id": "p2_c0", "page": 4}]}, hits)
    assert result["citations_valid"] is False


# ------------------------------------------------------------------ поиск
def test_cosine_bounds():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ------------------------------------------------------------------ live
@pytest.mark.live
@pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="нужен RUN_LIVE=1 и доступ к API")
def test_full_pipeline_passes():
    from src.pipeline import run

    report = run(PDF)
    failed = report["summary"]["failed_stages"]
    assert report["summary"]["overall"] == "PASS", failed
    assert (ARTIFACTS / "analysis.json").exists()
    assert (ARTIFACTS / "test_report.json").exists()

    analysis = json.loads((ARTIFACTS / "analysis.json").read_text(encoding="utf-8"))
    assert len(analysis["answers"]) == 4
    for answer in analysis["answers"]:
        assert answer["citations"], answer["key"]
        assert any(c.get("page") or c.get("chunk_id") for c in answer["citations"])
        assert answer["grounding"]["numbers_grounded"], answer["grounding"]["unsupported_numbers"]


@pytest.mark.live
@pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="нужен RUN_LIVE=1 и доступ к API")
def test_preflight_reports_all_four_services():
    from src.config import load_settings
    from src.pipeline import stage_preflight

    stage, results = stage_preflight(load_settings())
    assert stage.status == "PASS", stage.failures
    assert set(results) == {"qwen", "embedder", "reranker", "ragflow"}
    assert results["embedder"]["dim"] > 0
