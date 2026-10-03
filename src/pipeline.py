"""Оркестрация CLI-теста: preflight -> извлечение -> chunking -> embedding ->
retrieval -> reranking -> ответ Qwen -> артефакты и PASS/FAIL."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import time
from dataclasses import dataclass, field

from . import clients
from .answering import INSUFFICIENT, Answer, answer_question
from .chunking import build_chunks
from .clients import ServiceError
from .config import ARTIFACTS, DATA, Settings, collect_secrets, load_settings
from .extract import digital_report, extract_pages
from .retrieval import Index, apply_rerank

QUESTIONS = {
    "revenue": "Как изменилась выручка?",
    "margin": "Как изменилась маржинальность?",
    "debt": "Что произошло с долгом?",
    "cash_flow": "Что произошло с операционным денежным потоком?",
}

MIN_PAGES, MAX_PAGES = 3, 10
TOP_K = 8
TOP_N = 3


@dataclass
class Stage:
    name: str
    title: str
    status: str = "SKIP"
    details: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def finish(self) -> "Stage":
        self.status = "FAIL" if self.failures else "PASS"
        return self

    def to_dict(self) -> dict:
        return {
            "stage": self.name,
            "title": self.title,
            "status": self.status,
            "details": self.details,
            "failures": self.failures,
        }


def _norm(text: str) -> str:
    return re.sub(r"[\s ]+", " ", text).lower()


# --------------------------------------------------------------------------- этап 1
def stage_preflight(settings: Settings) -> tuple[Stage, dict]:
    stage = Stage("preflight", "Проверка Qwen, embedder, reranker и RAGFlow API")
    results: dict[str, dict] = {}

    for service in (settings.qwen, settings.embedder, settings.reranker):
        if not service.configured:
            stage.failures.append(f"{service.name}: не задан URL или API-ключ в .env")
            results[service.name] = {"ok": False, "error": "не сконфигурирован"}

    def record(name: str, fn) -> None:
        started = time.perf_counter()
        try:
            info = fn()
            results[name] = {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000), **info}
        except ServiceError as exc:
            results[name] = {"ok": False, "error": str(exc)}
            stage.failures.append(str(exc))

    if settings.qwen.configured:
        record("qwen", lambda: {
            "model": settings.qwen.model,
            "probe_reply": clients.chat(
                settings.qwen,
                [{"role": "user", "content": "Ответь ровно одним словом: ok"}],
                max_tokens=16,
            )[:40],
        })
    if settings.embedder.configured:
        def probe_embedder() -> dict:
            vec = clients.embed(settings.embedder, ["проверка доступности эмбеддера"])[0]
            return {"model": settings.embedder.model, "dim": len(vec),
                    "numeric_vector": all(isinstance(x, float) for x in vec)}
        record("embedder", probe_embedder)
    if settings.reranker.configured:
        def probe_reranker() -> dict:
            ranked = clients.rerank(
                settings.reranker,
                "как изменилась выручка",
                ["Сегодня в Астане ветрено", "Выручка выросла с 4 820 до 6 145 млн тенге"],
            )
            return {"model": settings.reranker.model, "top_index": ranked[0]["index"],
                    "orders_correctly": ranked[0]["index"] == 1}
        record("reranker", probe_reranker)
    if settings.ragflow_url and settings.ragflow_api_key:
        record("ragflow", lambda: clients.ragflow_ping(settings))
    else:
        stage.failures.append("ragflow: не задан URL или API-ключ в .env")
        results["ragflow"] = {"ok": False, "error": "не сконфигурирован"}

    if results.get("embedder", {}).get("ok") and not results["embedder"].get("numeric_vector"):
        stage.failures.append("embedder: вектор содержит нечисловые значения")
    if results.get("reranker", {}).get("ok") and not results["reranker"].get("orders_correctly"):
        stage.failures.append("reranker: не поднял релевантный фрагмент на первое место")

    stage.details = {"services": results, "endpoints": settings.public()}
    return stage.finish(), results


# --------------------------------------------------------------------------- этап 2
def stage_extraction(pdf_path: pathlib.Path, ground_truth: dict) -> tuple[Stage, list]:
    stage = Stage("extraction", "Извлечение текста с сохранением номеров страниц")
    pages = extract_pages(pdf_path)
    report = digital_report(pages)

    if not MIN_PAGES <= report["pages_total"] <= MAX_PAGES:
        stage.failures.append(
            f"в документе {report['pages_total']} страниц, требуется {MIN_PAGES}-{MAX_PAGES}"
        )
    if not report["is_digital"]:
        stage.failures.append(
            f"страницы без текстового слоя: {report['pages_without_text_layer']} "
            "(нужна отдельная OCR-проверка)"
        )

    by_page = {p.page: _norm(p.text) for p in pages}
    value_checks = []
    for key, check in ground_truth["checks"].items():
        page = check["page"]
        for value in check["values"].values():
            found = _norm(str(value)) in by_page.get(page, "")
            value_checks.append({"check": key, "page": page, "value": value, "found_on_page": found})
            if not found:
                stage.failures.append(f"{key}: значение «{value}» не найдено на странице {page}")

    periods_ok = all(
        any(period in text for text in by_page.values()) for period in ground_truth["periods"]
    )
    if not periods_ok:
        stage.failures.append(f"не найдены оба периода {ground_truth['periods']}")

    stage.details = {**report, "periods_found": periods_ok, "known_values": value_checks}
    return stage.finish(), pages


# --------------------------------------------------------------------------- этап 3
def stage_chunking(pages: list) -> tuple[Stage, list]:
    stage = Stage("chunking", "Разделение текста на фрагменты с сохранением страницы")
    chunks = build_chunks(pages)

    if not chunks:
        stage.failures.append("фрагменты не созданы")
    if len({c.chunk_id for c in chunks}) != len(chunks):
        stage.failures.append("chunk_id не уникальны")
    pages_with_text = {p.page for p in pages if p.text}
    covered = {c.page for c in chunks}
    missing = sorted(pages_with_text - covered)
    if missing:
        stage.failures.append(f"страницы без фрагментов: {missing}")
    if any(c.page < 1 for c in chunks):
        stage.failures.append("встречен фрагмент без корректного номера страницы")

    stage.details = {
        "chunks_total": len(chunks),
        "pages_covered": sorted(covered),
        "chars_min": min((len(c.text) for c in chunks), default=0),
        "chars_max": max((len(c.text) for c in chunks), default=0),
        "chunks_per_page": {
            str(p): sum(1 for c in chunks if c.page == p) for p in sorted(covered)
        },
    }
    return stage.finish(), chunks


# --------------------------------------------------------------------------- этапы 4-6
def stage_retrieval(settings: Settings, chunks: list, ground_truth: dict) -> tuple[Stage, Stage, dict]:
    embedding_stage = Stage("embedding_search", "Индексация и embedding-поиск")
    rerank_stage = Stage("reranking", "Reranking найденных фрагментов")

    index = Index(settings.embedder, chunks)
    if index.dim <= 0:
        embedding_stage.failures.append("embedder вернул пустые векторы")
    if len(index.vectors) != len(chunks):
        embedding_stage.failures.append("число векторов не совпадает с числом фрагментов")

    retrieved: dict[str, dict] = {}
    emb_rows, rr_rows = [], []

    for key, question in QUESTIONS.items():
        expected_page = ground_truth["checks"][key]["page"]
        hits = index.search(question, top_k=TOP_K)
        if not hits:
            embedding_stage.failures.append(f"{key}: поиск не вернул фрагментов")
            continue
        emb_pages = [h.chunk.page for h in hits]
        if expected_page not in emb_pages:
            embedding_stage.failures.append(
                f"{key}: страница {expected_page} не попала в top-{TOP_K} embedding-поиска"
            )
        emb_top_page = hits[0].chunk.page

        top = apply_rerank(settings.reranker, question, hits, top_n=TOP_N)
        if not top:
            rerank_stage.failures.append(f"{key}: reranker не вернул фрагментов")
            continue
        if top[0].chunk.page != expected_page:
            rerank_stage.failures.append(
                f"{key}: после reranking на первом месте страница {top[0].chunk.page}, "
                f"ожидалась {expected_page}"
            )

        retrieved[key] = {"question": question, "hits": hits, "top": top}
        emb_rows.append({
            "key": key, "expected_page": expected_page,
            "top_page": emb_top_page, "top_k_pages": emb_pages,
            "expected_in_top_k": expected_page in emb_pages,
            "top_score": round(hits[0].embedding_score, 6),
        })
        rr_rows.append({
            "key": key, "expected_page": expected_page,
            "top_page": top[0].chunk.page,
            "top_chunk_id": top[0].chunk.chunk_id,
            "top_score": round(top[0].rerank_score or 0.0, 6),
            "rank_before_rerank": top[0].embedding_rank,
            "promoted_by_rerank": top[0].embedding_rank > 1,
            "selected": [h.to_dict() for h in top],
        })

    embedding_stage.details = {
        "vector_dim": index.dim,
        "vectors_indexed": len(index.vectors),
        "top_k": TOP_K,
        "queries": emb_rows,
    }
    rerank_stage.details = {"top_n": TOP_N, "queries": rr_rows}
    return embedding_stage.finish(), rerank_stage.finish(), retrieved


# --------------------------------------------------------------------------- этап 7
def stage_answers(settings: Settings, retrieved: dict, ground_truth: dict) -> tuple[Stage, Stage, Stage, list[Answer]]:
    answer_stage = Stage("qwen_answers", "Ответы Qwen по отобранным фрагментам")
    citation_stage = Stage("citations", "Каждый ответ содержит страницу или chunk_id")
    grounding_stage = Stage("no_outside_knowledge", "Отсутствие фактов вне документа")

    answers: list[Answer] = []
    answer_rows, citation_rows = [], []

    for key, question in QUESTIONS.items():
        bundle = retrieved.get(key)
        if not bundle:
            answer_stage.failures.append(f"{key}: нет отобранных фрагментов для ответа")
            continue
        try:
            answer = answer_question(settings.qwen, key, question, bundle["top"])
        except ServiceError as exc:
            answer_stage.failures.append(f"{key}: {exc}")
            continue
        answers.append(answer)

        check = ground_truth["checks"][key]
        answer_norm = _norm(answer.text + " " + json.dumps(answer.raw.get("metrics") or [], ensure_ascii=False))
        missing_all = [v for v in check["must_contain_all"] if _norm(v) not in answer_norm]
        any_ok = (not check["must_contain_any"]) or any(
            _norm(v) in answer_norm for v in check["must_contain_any"]
        )

        if not answer.sufficient:
            answer_stage.failures.append(f"{key}: модель вернула {INSUFFICIENT} при наличии данных")
        if missing_all:
            answer_stage.failures.append(f"{key}: в ответе нет обязательных значений {missing_all}")
        if not any_ok:
            answer_stage.failures.append(
                f"{key}: в ответе нет ни одного из значений {check['must_contain_any']}"
            )

        cited_pages = sorted({int(c["page"]) for c in answer.citations if c.get("page") is not None})
        cited_ids = sorted({str(c["chunk_id"]) for c in answer.citations if c.get("chunk_id")})
        if not (cited_pages or cited_ids):
            citation_stage.failures.append(f"{key}: ответ не содержит ни страницы, ни chunk_id")
        if not answer.grounding["citations_valid"]:
            citation_stage.failures.append(
                f"{key}: некорректные ссылки {answer.grounding['invalid_citations']}"
            )
        if check["page"] not in cited_pages:
            citation_stage.failures.append(
                f"{key}: ожидалась ссылка на страницу {check['page']}, получены {cited_pages}"
            )

        if not answer.grounding["numbers_grounded"]:
            grounding_stage.failures.append(
                f"{key}: числа вне контекста {answer.grounding['unsupported_numbers']}"
            )

        answer_rows.append({
            "key": key, "sufficient": answer.sufficient,
            "must_contain_all_ok": not missing_all, "must_contain_any_ok": any_ok,
            "expected_direction": check["direction"],
        })
        citation_rows.append({
            "key": key, "cited_pages": cited_pages, "cited_chunk_ids": cited_ids,
            "expected_page": check["page"],
            "quote_found_in_context": answer.grounding["quote_found_in_context"],
        })

    # Негативный контроль: факта нет в документе -> insufficient_evidence
    control = ground_truth["absent_fact_control"]
    control_row = {"question": control["question"], "expected": control["expected"]}
    any_bundle = next(iter(retrieved.values()), None)
    if any_bundle:
        try:
            ctrl = answer_question(settings.qwen, "absent_fact_control", control["question"], any_bundle["top"])
            control_row.update({
                "sufficient": ctrl.sufficient,
                "answer": ctrl.text[:200],
                "unsupported_numbers": ctrl.grounding["unsupported_numbers"],
                "passed": not ctrl.sufficient,
            })
            if ctrl.sufficient:
                grounding_stage.failures.append(
                    "негативный контроль: модель ответила на вопрос, данных для которого нет в документе"
                )
        except ServiceError as exc:
            control_row.update({"passed": False, "error": str(exc)})
            grounding_stage.failures.append(f"негативный контроль: {exc}")
    else:
        control_row.update({"passed": False, "error": "нет контекста"})
        grounding_stage.failures.append("негативный контроль: нет контекста")

    answer_stage.details = {"model": settings.qwen.model, "answers": answer_rows}
    citation_stage.details = {"answers": citation_rows}
    grounding_stage.details = {
        "absent_fact_control": control_row,
        "unsupported_numbers_by_answer": {
            a.key: a.grounding["unsupported_numbers"] for a in answers
        },
    }
    return answer_stage.finish(), citation_stage.finish(), grounding_stage.finish(), answers


# --------------------------------------------------------------------------- этап 8
def _verbatim_coverage(pages: list, blob: str, window: int, step: int) -> tuple[float, list[int]]:
    """Доля документа, попавшая в артефакты дословно, и страницы с совпадениями."""
    total = matched = 0
    hit_pages: list[int] = []
    for page in pages:
        norm_page = _norm(page.text)
        if len(norm_page) < window:
            continue
        page_hit = False
        for start in range(0, len(norm_page) - window + 1, step):
            total += 1
            if norm_page[start : start + window] in blob:
                matched += 1
                page_hit = True
        if page_hit:
            hit_pages.append(page.page)
    return (matched / total if total else 0.0), hit_pages


# Цитаты допустимы, выгрузка документа — нет.
MAX_VERBATIM_SPAN = 400      # ни один дословный отрывок не длиннее, символов
MAX_DOC_COVERAGE = 0.25      # не более четверти документа дословно


def stage_artifact_safety(payloads: dict[str, dict], pages: list) -> Stage:
    stage = Stage("artifact_safety", "В артефактах нет секретов и полного документа")
    blob = json.dumps(payloads, ensure_ascii=False)
    normalized_blob = _norm(blob)

    leaked = [s for s in collect_secrets() if s in blob]
    if leaked:
        stage.failures.append(f"в артефактах обнаружено секретных значений: {len(leaked)}")

    coverage, covered_pages = _verbatim_coverage(pages, normalized_blob, window=120, step=60)
    if coverage > MAX_DOC_COVERAGE:
        stage.failures.append(
            f"дословно воспроизведено {coverage:.0%} документа при лимите {MAX_DOC_COVERAGE:.0%}"
        )

    long_span, long_pages = _verbatim_coverage(pages, normalized_blob, window=MAX_VERBATIM_SPAN, step=MAX_VERBATIM_SPAN // 4)
    if long_pages:
        stage.failures.append(
            f"дословный отрывок длиннее {MAX_VERBATIM_SPAN} символов: страницы {long_pages}"
        )

    stage.details = {
        "secret_values_checked": len(collect_secrets()),
        "secret_values_leaked": len(leaked),
        "verbatim_coverage": round(coverage, 4),
        "verbatim_coverage_limit": MAX_DOC_COVERAGE,
        "pages_with_verbatim_fragments": covered_pages,
        "pages_with_long_spans": long_pages,
        "artifact_bytes": {name: len(json.dumps(p, ensure_ascii=False)) for name, p in payloads.items()},
    }
    return stage.finish()


# --------------------------------------------------------------------------- run
def run(pdf_path: pathlib.Path, env_path: pathlib.Path | None = None) -> dict:
    started = dt.datetime.now(dt.timezone.utc)
    settings = load_settings(env_path)
    ground_truth = json.loads((DATA / "ground_truth.json").read_text(encoding="utf-8"))

    stages: list[Stage] = []
    analysis: dict = {
        "generated_at": started.isoformat(timespec="seconds"),
        "document": {"file": pdf_path.name},
        "questions": QUESTIONS,
        "answers": [],
        "retrieval": [],
    }

    preflight, _ = stage_preflight(settings)
    stages.append(preflight)
    pages: list = []

    if preflight.status == "PASS":
        extraction, pages = stage_extraction(pdf_path, ground_truth)
        stages.append(extraction)
        analysis["document"].update({
            "pages": len(pages),
            "chars_total": sum(p.chars for p in pages),
            "is_digital": extraction.details.get("is_digital"),
        })

        chunking, chunks = stage_chunking(pages)
        stages.append(chunking)

        if chunks:
            embedding, reranking, retrieved = stage_retrieval(settings, chunks, ground_truth)
            stages += [embedding, reranking]
            analysis["retrieval"] = [
                {
                    "key": key,
                    "question": bundle["question"],
                    "embedding_top_k": [h.to_dict() for h in bundle["hits"]],
                    "selected_after_rerank": [h.to_dict() for h in bundle["top"]],
                }
                for key, bundle in retrieved.items()
            ]
            answer_stage, citation_stage, grounding_stage, answers = stage_answers(
                settings, retrieved, ground_truth
            )
            stages += [answer_stage, citation_stage, grounding_stage]
            analysis["answers"] = [a.to_dict() for a in answers]

    report = {
        "generated_at": started.isoformat(timespec="seconds"),
        "document": analysis["document"],
        "endpoints": settings.public(),
        "stages": [s.to_dict() for s in stages],
    }
    safety = stage_artifact_safety({"analysis": analysis, "test_report": report}, pages)
    stages.append(safety)
    report["stages"].append(safety.to_dict())

    failed = [s.name for s in stages if s.status == "FAIL"]
    report["summary"] = {
        "stages_total": len(stages),
        "passed": sum(1 for s in stages if s.status == "PASS"),
        "failed": len(failed),
        "failed_stages": failed,
        "overall": "FAIL" if failed else "PASS",
        "duration_sec": round((dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 1),
    }
    analysis["overall"] = report["summary"]["overall"]

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS / "test_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
