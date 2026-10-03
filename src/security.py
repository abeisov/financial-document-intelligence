"""Минимальная защита публичного размещения: HTTP Basic.

Пароль берётся только из переменной окружения APP_PASSWORD. Если переменная не
задана, защита выключена — это режим локальной разработки. На Vercel переменная
задаётся в Environment Variables, в код и в браузер секрет не попадает.
"""
from __future__ import annotations

import base64
import binascii
import os
import secrets

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

#: эти пути остаются открытыми, чтобы внешний мониторинг видел статус сервиса
PUBLIC_PATHS = frozenset({"/api/health"})
REALM = "Financial Document Intelligence"


def auth_enabled() -> bool:
    return bool(os.getenv("APP_PASSWORD", "").strip())


def _expected() -> tuple[str, str]:
    return os.getenv("APP_USER", "admin").strip() or "admin", os.getenv("APP_PASSWORD", "").strip()


def check_credentials(header: str | None) -> bool:
    """Сравнение в постоянном времени, чтобы заголовок нельзя было подобрать по таймингу."""
    user, password = _expected()
    if not password:
        return True
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1].strip()).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, IndexError):
        return False
    got_user, _, got_password = decoded.partition(":")
    return secrets.compare_digest(got_user, user) and secrets.compare_digest(got_password, password)


def unauthorized() -> Response:
    return JSONResponse(
        {"detail": "Требуется вход"},
        status_code=401,
        headers={"WWW-Authenticate": f'Basic realm="{REALM}", charset="UTF-8"'},
    )


async def basic_auth_middleware(request: Request, call_next):
    if not auth_enabled() or request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    if check_credentials(request.headers.get("authorization")):
        return await call_next(request)
    return unauthorized()
