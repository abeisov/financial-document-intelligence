"""Точка входа для Vercel: платформа ищет ASGI-приложение с именем app.

Логика целиком остаётся в src/web.py, здесь только импорт, чтобы локальный
запуск (uvicorn src.web:app) и размещение на Vercel использовали один код.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.web import app  # noqa: E402

__all__ = ["app"]
