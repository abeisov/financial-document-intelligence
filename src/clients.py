"""HTTP-клиенты для Qwen, embedder, reranker и RAGFlow."""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from .config import Service, Settings

TIMEOUT = 120


class ServiceError(RuntimeError):
    pass


def _post(service: Service, payload: dict, url: str | None = None) -> dict:
    try:
        resp = requests.post(
            url or service.url,
            headers={
                "Authorization": f"Bearer {service.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:  # сетевой сбой
        raise ServiceError(f"{service.name}: сетевая ошибка ({type(exc).__name__})") from None
    if resp.status_code != 200:
        raise ServiceError(f"{service.name}: HTTP {resp.status_code} {resp.text[:200]}")
    try:
        return resp.json()
    except ValueError:
        raise ServiceError(f"{service.name}: ответ не является JSON") from None


# --------------------------------------------------------------------------- embedder
def embed(service: Service, texts: list[str], batch_size: int = 16) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        data = _post(service, {"model": service.model, "input": batch})
        rows = data.get("data") or []
        if len(rows) != len(batch):
            raise ServiceError(f"embedder: получено {len(rows)} векторов вместо {len(batch)}")
        for row in rows:
            vec = row.get("embedding")
            if not isinstance(vec, list) or not vec or not all(isinstance(x, (int, float)) for x in vec):
                raise ServiceError("embedder: вектор пуст или содержит нечисловые значения")
            vectors.append([float(x) for x in vec])
    return vectors


# --------------------------------------------------------------------------- reranker
def rerank(service: Service, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    payload: dict[str, Any] = {"model": service.model, "query": query, "documents": documents}
    if top_n:
        payload["top_n"] = top_n
    data = _post(service, payload)
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ServiceError("reranker: пустой результат")
    out = []
    for item in results:
        idx = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if not isinstance(idx, int) or not isinstance(score, (int, float)):
            raise ServiceError("reranker: некорректная структура результата")
        out.append({"index": idx, "score": float(score)})
    return sorted(out, key=lambda r: r["score"], reverse=True)


# --------------------------------------------------------------------------- qwen
def chat(
    service: Service,
    messages: list[dict],
    *,
    max_tokens: int = 900,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    retries: int = 2,
) -> str:
    payload = {
        "model": service.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            data = _post(service, payload)
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            if content.strip():
                return content.strip()
            last = ServiceError("qwen: пустой content")
        except ServiceError as exc:
            last = exc
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise last or ServiceError("qwen: не удалось получить ответ")


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def chat_json(service: Service, messages: list[dict], **kwargs) -> dict:
    """Ответ модели, приведённый к JSON-объекту."""
    raw = chat(service, messages, **kwargs)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(cleaned)
        if not match:
            raise ServiceError(f"qwen: ответ не является JSON: {cleaned[:160]}") from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            raise ServiceError(f"qwen: не удалось разобрать JSON: {cleaned[:160]}") from None


# --------------------------------------------------------------------------- ragflow
def ragflow_ping(settings: Settings) -> dict:
    url = f"{settings.ragflow_url}/api/v1/datasets"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {settings.ragflow_api_key}"},
            params={"page": 1, "page_size": 1},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ServiceError(f"ragflow: сетевая ошибка ({type(exc).__name__})") from None
    if resp.status_code != 200:
        raise ServiceError(f"ragflow: HTTP {resp.status_code} {resp.text[:200]}")
    body = resp.json()
    if body.get("code") != 0:
        raise ServiceError(f"ragflow: code={body.get('code')} {str(body.get('message'))[:160]}")
    return {"datasets_total": body.get("total_datasets", 0)}
