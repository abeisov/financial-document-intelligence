"""Локальный web-MVP: RAGFlow upload/retrieval -> reranker -> Qwen."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .clients import ServiceError
from .config import ROOT, load_settings
from .ragflow_client import RagflowClient
from .security import auth_enabled, basic_auth_middleware
from .web_analysis import (
    answer_from_sources,
    max_upload_mb,
    normalize_ragflow_chunks,
    select_sources,
    validate_upload,
)

STATIC = ROOT / "web" / "static"

app = FastAPI(
    title="Financial Document Intelligence",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url=None,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.middleware("http")(basic_auth_middleware)


class DatasetCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)


class AskRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    document_ids: list[str] = Field(default_factory=list, max_length=20)
    question: str = Field(min_length=3, max_length=1000)


def _dependencies():
    settings = load_settings()
    ragflow = RagflowClient(settings.ragflow_url, settings.ragflow_api_key)
    return settings, ragflow


def _service_error(exc: ServiceError) -> HTTPException:
    return HTTPException(status_code=502, detail=str(exc))


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    settings, ragflow = _dependencies()
    return {
        "status": "ok",
        "services": {
            "ragflow": ragflow.configured,
            "qwen": settings.qwen.configured,
            "embedder": settings.embedder.configured,
            "reranker": settings.reranker.configured,
        },
        "defaults": {
            "dataset_name": settings.ragflow_dataset_name,
            "chunk_method": settings.ragflow_chunk_method,
        },
        "limits": {"max_upload_mb": max_upload_mb()},
        "auth": {"enabled": auth_enabled()},
    }


@app.get("/api/datasets")
def list_datasets() -> dict:
    _, ragflow = _dependencies()
    try:
        return {"datasets": ragflow.list_datasets()}
    except ServiceError as exc:
        raise _service_error(exc) from None


@app.post("/api/datasets")
def create_dataset(request: DatasetCreate) -> dict:
    settings, ragflow = _dependencies()
    try:
        dataset, created = ragflow.get_or_create_dataset(
            request.name.strip(),
            chunk_method=settings.ragflow_chunk_method,
        )
        return {"dataset": dataset, "created": created}
    except ServiceError as exc:
        raise _service_error(exc) from None


@app.post("/api/documents")
async def upload_document(
    dataset_id: Annotated[str, Form(min_length=1)],
    file: Annotated[UploadFile, File()],
) -> dict:
    _, ragflow = _dependencies()
    content = await file.read()
    try:
        safe_name = validate_upload(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        documents = ragflow.upload_document(
            dataset_id,
            safe_name,
            content,
            file.content_type or "application/octet-stream",
        )
        ragflow.start_parsing(dataset_id, [item.id for item in documents])
        return {
            "dataset_id": dataset_id,
            "documents": [item.public() for item in documents],
            "parsing_started": True,
        }
    except ServiceError as exc:
        raise _service_error(exc) from None
    finally:
        await file.close()


@app.get("/api/documents/{dataset_id}/{document_id}")
def document_status(dataset_id: str, document_id: str) -> dict:
    _, ragflow = _dependencies()
    try:
        return {"document": ragflow.get_document(dataset_id, document_id).public()}
    except ServiceError as exc:
        raise _service_error(exc) from None


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    settings, ragflow = _dependencies()
    if not settings.qwen.configured or not settings.reranker.configured:
        raise HTTPException(
            status_code=503,
            detail="Qwen или Reranker не настроен в .env",
        )
    try:
        raw_chunks = ragflow.retrieve(
            request.dataset_id,
            request.question,
            document_ids=request.document_ids,
            page_size=12,
        )
        normalized = normalize_ragflow_chunks(raw_chunks)
        selected = select_sources(settings, request.question, normalized, top_n=5)
        result = answer_from_sources(settings, request.question, selected)
        return {
            "question": request.question,
            "retrieved": len(normalized),
            "selected_after_rerank": len(selected),
            **result,
        }
    except ServiceError as exc:
        raise _service_error(exc) from None
