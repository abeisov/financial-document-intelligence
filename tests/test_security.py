"""Проверка минимальной защиты публичного размещения."""
import base64

import pytest
from fastapi.testclient import TestClient

from src.web import app

client = TestClient(app)


def _basic(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_open_when_password_not_set(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert client.get("/").status_code == 200


def test_requires_password_when_configured(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret-value")
    response = client.get("/")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic")
    assert "s3cret-value" not in response.text


def test_accepts_correct_password(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret-value")
    monkeypatch.setenv("APP_USER", "analyst")
    assert client.get("/", headers=_basic("analyst", "s3cret-value")).status_code == 200


def test_rejects_wrong_password(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret-value")
    monkeypatch.setenv("APP_USER", "analyst")
    assert client.get("/", headers=_basic("analyst", "wrong")).status_code == 401
    assert client.get("/", headers={"Authorization": "Basic ???"}).status_code == 401


def test_health_stays_public_for_monitoring(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret-value")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "s3cret-value" not in response.text


@pytest.mark.parametrize("value,expected", [("", 4), ("8", 8), ("abc", 4), ("0", 4)])
def test_upload_limit_is_configurable(monkeypatch, value, expected):
    from src import web_analysis

    monkeypatch.setenv("MAX_UPLOAD_MB", value)
    assert web_analysis.max_upload_mb() == expected


def test_upload_rejects_file_above_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
    response = client.post(
        "/api/documents",
        data={"dataset_id": "test-dataset"},
        files={"file": ("report.pdf", big, "application/pdf")},
    )
    assert response.status_code == 400
    assert "больше 1 МБ" in response.json()["detail"]
