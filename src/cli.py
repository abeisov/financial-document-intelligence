"""CLI-вход: python -m src.cli --pdf data/test_financial_report.pdf"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .config import ARTIFACTS, DATA
from .pipeline import run

MARK = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}


def _print_report(report: dict) -> None:
    doc = report["document"]
    print()
    print("=" * 78)
    print(f"  Документ: {doc.get('file')}  |  страниц: {doc.get('pages', '-')}  "
          f"|  цифровой: {doc.get('is_digital', '-')}")
    print("=" * 78)

    for stage in report["stages"]:
        print(f"[{MARK[stage['status']]}] {stage['stage']:<22} {stage['title']}")
        for failure in stage["failures"]:
            print(f"        ! {failure}")

    analysis = json.loads((ARTIFACTS / "analysis.json").read_text(encoding="utf-8"))
    if analysis.get("answers"):
        print("-" * 78)
        print("  ОТВЕТЫ (только по документу)")
        print("-" * 78)
        for item in analysis["answers"]:
            cites = ", ".join(
                f"стр. {c.get('page')} / {c.get('chunk_id')}" for c in item["citations"]
            ) or "нет ссылки"
            print(f"\n• {item['question']}")
            print(f"  {item['answer']}")
            for metric in item["metrics"]:
                print(f"    - {metric.get('name')}: {metric.get('prev')} -> "
                      f"{metric.get('current')} ({metric.get('change')})")
            print(f"  источник: {cites}")

    summary = report["summary"]
    print()
    print("=" * 78)
    print(f"  ИТОГ: {summary['overall']}  |  этапов: {summary['stages_total']}  "
          f"|  PASS: {summary['passed']}  FAIL: {summary['failed']}  "
          f"|  {summary['duration_sec']} c")
    if summary["failed_stages"]:
        print(f"  Провалены: {', '.join(summary['failed_stages'])}")
    print(f"  Артефакты: {ARTIFACTS / 'analysis.json'}")
    print(f"             {ARTIFACTS / 'test_report.json'}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="financial-doc-test",
        description="CLI-тест цепочки: извлечение -> chunking -> embedding -> rerank -> Qwen",
    )
    parser.add_argument("--pdf", type=pathlib.Path,
                        default=DATA / "test_financial_report.pdf",
                        help="путь к цифровому PDF на 3-10 страниц")
    parser.add_argument("--env", type=pathlib.Path, default=None, help="путь к .env")
    parser.add_argument("--json", action="store_true", help="вывести отчёт в JSON")
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        print(f"Файл не найден: {args.pdf}", file=sys.stderr)
        print("Сгенерируйте фикстуру: python scripts/make_test_pdf.py", file=sys.stderr)
        return 2

    report = run(args.pdf, args.env)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0 if report["summary"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
