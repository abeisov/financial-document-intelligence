"""Smoke-тесты локального HTTP-интерфейса без внешних запросов."""
from fastapi.testclient import TestClient

from src.web import app

client = TestClient(app)


def test_home_page_is_available():
    response = client.get("/")
    assert response.status_code == 200
    assert "FinScope" in response.text


def test_health_exposes_only_configuration_flags():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body["services"]) == {"ragflow", "qwen", "embedder", "reranker"}
    assert all(isinstance(value, bool) for value in body["services"].values())
    serialized = response.text.lower()
    assert "api_key" not in serialized
    assert "password" not in serialized


def test_upload_rejects_unknown_format_before_ragflow_call():
    response = client.post(
        "/api/documents",
        data={"dataset_id": "test-dataset"},
        files={"file": ("payload.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Неподдерживаемый формат" in response.json()["detail"]
