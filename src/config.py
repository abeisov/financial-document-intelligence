"""Загрузка конфигурации из .env. Секреты никогда не попадают в вывод и артефакты."""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"

SECRET_SUFFIXES = ("_API_KEY", "_PASSWORD", "_TOKEN", "_SECRET")


@dataclass(frozen=True)
class Service:
    name: str
    url: str
    model: str
    api_key: str

    @property
    def configured(self) -> bool:
        return bool(self.url and self.api_key)

    def public(self) -> dict:
        """Безопасное для логов представление: без ключа."""
        return {"service": self.name, "url": self.url, "model": self.model}


@dataclass(frozen=True)
class Settings:
    qwen: Service
    embedder: Service
    reranker: Service
    ragflow_url: str
    ragflow_api_key: str
    ragflow_dataset_name: str
    ragflow_chunk_method: str

    def public(self) -> dict:
        return {
            "qwen": self.qwen.public(),
            "embedder": self.embedder.public(),
            "reranker": self.reranker.public(),
            "ragflow": {
                "service": "ragflow",
                "url": self.ragflow_url,
                "dataset_name": self.ragflow_dataset_name,
                "chunk_method": self.ragflow_chunk_method,
            },
        }


def _service(name: str, prefix: str) -> Service:
    return Service(
        name=name,
        url=os.getenv(f"{prefix}_URL", "").strip(),
        model=os.getenv(f"{prefix}_MODEL", "").strip(),
        api_key=os.getenv(f"{prefix}_API_KEY", "").strip(),
    )


def load_settings(env_path: pathlib.Path | None = None) -> Settings:
    load_dotenv(env_path or ROOT / ".env", override=False)
    return Settings(
        qwen=_service("qwen", "QWEN"),
        embedder=_service("embedder", "EMBEDDER"),
        reranker=_service("reranker", "RERANKER"),
        ragflow_url=os.getenv("RAGFLOW_URL", "").strip().rstrip("/"),
        ragflow_api_key=os.getenv("RAGFLOW_API_KEY", "").strip(),
        ragflow_dataset_name=os.getenv(
            "RAGFLOW_DATASET_NAME", "financial-documents"
        ).strip() or "financial-documents",
        ragflow_chunk_method=os.getenv("RAGFLOW_CHUNK_METHOD", "naive").strip() or "naive",
    )


def collect_secrets() -> list[str]:
    """Значения, которые обязаны отсутствовать в артефактах."""
    out = []
    for key, value in os.environ.items():
        if any(key.endswith(s) for s in SECRET_SUFFIXES) and len(value.strip()) >= 8:
            out.append(value.strip())
    return out
