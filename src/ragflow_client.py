"""Минимальный безопасный клиент RAGFlow HTTP API для web-MVP."""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Any

import requests

from .clients import ServiceError, TIMEOUT


def _safe_progress_message(value: Any) -> str:
    message = str(value or "").strip()
    message = re.sub(
        r"(?i)\bauthorization\s*[:=]\s*(?:bearer\s+)?\S+",
        "Authorization: [REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)\b(api[-_ ]?key|password)\s*[:=]\s*\S+",
        r"\1: [REDACTED]",
        message,
    )
    message = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"\b[0-9a-f]{24,}\b", "[ID]", message, flags=re.IGNORECASE)
    return message[-800:]


@dataclass(frozen=True)
class RagflowDocument:
    id: str
    name: str
    run: str
    progress: float
    chunk_count: int
    token_count: int
    message: str

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "RagflowDocument":
        run_raw = raw.get("run", raw.get("status", "UNSTART"))
        run_names = {
            0: "UNSTART",
            1: "RUNNING",
            2: "CANCEL",
            3: "DONE",
            4: "FAIL",
            5: "SCHEDULED",
        }
        try:
            run = run_names.get(int(run_raw), str(run_raw))
        except (TypeError, ValueError):
            run = str(run_raw)
        return cls(
            id=str(raw.get("id") or ""),
            name=str(raw.get("name") or raw.get("display_name") or "Документ"),
            run=run,
            progress=float(raw.get("progress") or 0.0),
            chunk_count=int(raw.get("chunk_count") or 0),
            token_count=int(raw.get("token_count") or 0),
            message=_safe_progress_message(
                raw.get("progress_msg") or raw.get("message") or ""
            ),
        )

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "run": self.run,
            "progress": self.progress,
            "chunk_count": self.chunk_count,
            "token_count": self.token_count,
            "message": self.message,
        }


class RagflowClient:
    """Поддерживает только операции, необходимые текущему MVP."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    ) -> Any:
        if not self.configured:
            raise ServiceError("ragflow: URL или API-ключ не настроен")
        url = f"{self.base_url}/api/v1{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if files is None:
            headers["Content-Type"] = "application/json"
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=payload if files is None else None,
                files=files,
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ServiceError(
                f"ragflow: сетевая ошибка ({type(exc).__name__})"
            ) from None
        if response.status_code != 200:
            safe_body = response.text[:240].replace(self.api_key, "[REDACTED]")
            raise ServiceError(f"ragflow: HTTP {response.status_code} {safe_body}")
        try:
            body = response.json()
        except ValueError:
            raise ServiceError("ragflow: ответ не является JSON") from None
        if not isinstance(body, dict) or body.get("code") != 0:
            message = str(body.get("message") if isinstance(body, dict) else "unknown")
            message = message.replace(self.api_key, "[REDACTED]")
            raise ServiceError(f"ragflow: API error {message[:240]}")
        return body.get("data")

    def list_datasets(self, name: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"page": 1, "page_size": 100}
        if name:
            params["name"] = name
        data = self._request("GET", "/datasets", params=params)
        rows = data if isinstance(data, list) else []
        return [
            {
                "id": str(row.get("id") or ""),
                "name": str(row.get("name") or "Без названия"),
                "document_count": int(row.get("document_count") or 0),
                "chunk_count": int(row.get("chunk_count") or 0),
                "chunk_method": str(row.get("chunk_method") or row.get("parser_id") or ""),
            }
            for row in rows
            if isinstance(row, dict) and row.get("id")
        ]

    def get_or_create_dataset(
        self,
        name: str,
        *,
        chunk_method: str = "naive",
    ) -> tuple[dict[str, Any], bool]:
        # Shared-инсталляция может найти чужой Dataset при серверной фильтрации
        # по name и вернуть permission error. Фильтруем доступный список локально.
        existing = self.list_datasets()
        exact = next((row for row in existing if row["name"] == name), None)
        if exact:
            return exact, False
        variant_pattern = re.compile(rf"^{re.escape(name)}-[0-9a-f]{{6}}$")
        variant = next(
            (row for row in existing if variant_pattern.fullmatch(row["name"])),
            None,
        )
        if variant:
            return variant, False

        def create(dataset_name: str) -> Any:
            return self._request(
                "POST",
                "/datasets",
                payload={
                    "name": dataset_name,
                    "description": "Финансовые документы, загруженные через Financial Document Intelligence",
                    "permission": "me",
                    "chunk_method": chunk_method,
                },
            )

        created_name = name
        try:
            data = create(created_name)
        except ServiceError as exc:
            if "lacks permission for dataset" not in str(exc).lower():
                raise
            created_name = f"{name}-{secrets.token_hex(3)}"
            try:
                data = create(created_name)
            except ServiceError as retry_exc:
                if "lacks permission for dataset" not in str(retry_exc).lower():
                    raise
                raise ServiceError(
                    "ragflow: API-ключ не имеет права создавать Dataset; "
                    "создайте Dataset в кабинете RAGFlow или запросите это право"
                ) from None
        if not isinstance(data, dict) or not data.get("id"):
            raise ServiceError("ragflow: Dataset создан, но API не вернул его ID")
        return {
            "id": str(data["id"]),
            "name": str(data.get("name") or created_name),
            "document_count": int(data.get("document_count") or 0),
            "chunk_count": int(data.get("chunk_count") or 0),
            "chunk_method": str(data.get("chunk_method") or chunk_method),
        }, True

    def upload_document(
        self,
        dataset_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> list[RagflowDocument]:
        data = self._request(
            "POST",
            f"/datasets/{dataset_id}/documents",
            files=[("file", (filename, content, content_type))],
        )
        if not isinstance(data, list) or not data:
            raise ServiceError("ragflow: загрузка не вернула документ")
        documents = [RagflowDocument.from_api(row) for row in data if isinstance(row, dict)]
        if not documents or any(not item.id for item in documents):
            raise ServiceError("ragflow: в ответе загрузки отсутствует document_id")
        return documents

    def start_parsing(self, dataset_id: str, document_ids: list[str]) -> None:
        self._request(
            "POST",
            f"/datasets/{dataset_id}/chunks",
            payload={"document_ids": document_ids},
        )

    def get_document(self, dataset_id: str, document_id: str) -> RagflowDocument:
        data = self._request(
            "GET",
            f"/datasets/{dataset_id}/documents",
            params={"id": document_id, "page": 1, "page_size": 1},
        )
        rows = data.get("docs", []) if isinstance(data, dict) else []
        if not rows:
            raise ServiceError("ragflow: документ не найден")
        return RagflowDocument.from_api(rows[0])

    def retrieve(
        self,
        dataset_id: str,
        question: str,
        *,
        document_ids: list[str] | None = None,
        page_size: int = 12,
    ) -> list[dict[str, Any]]:
        data = self._request(
            "POST",
            "/retrieval",
            payload={
                "question": question,
                "dataset_ids": [dataset_id],
                "document_ids": document_ids or [],
                "page": 1,
                "page_size": page_size,
                "top_k": max(64, page_size * 8),
                "similarity_threshold": 0.15,
                "vector_similarity_weight": 0.35,
                "keyword": True,
            },
        )
        chunks = data.get("chunks", []) if isinstance(data, dict) else []
        return [row for row in chunks if isinstance(row, dict)]
