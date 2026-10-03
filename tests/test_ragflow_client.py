"""Проверки совместимости с вариантами ответа RAGFlow."""
from src.clients import ServiceError
from src.ragflow_client import RagflowClient, RagflowDocument


def test_numeric_document_status_is_normalized():
    document = RagflowDocument.from_api(
        {
            "id": "doc-1",
            "name": "report.pdf",
            "run": 3,
            "progress": 1,
            "progress_msg": "Done",
        }
    )
    assert document.run == "DONE"
    assert document.progress == 1.0
    assert document.message == "Done"


def test_document_message_redacts_credentials_and_internal_ids():
    document = RagflowDocument.from_api(
        {
            "id": "doc-1",
            "progress_msg": (
                "Authorization: Bearer secret-value "
                "user=221200e0ace611f1b110c35aea857303"
            ),
        }
    )
    assert "secret-value" not in document.message
    assert "221200e0ace611f1b110c35aea857303" not in document.message


def test_dataset_name_collision_gets_private_suffix(monkeypatch):
    client = RagflowClient("https://ragflow.example", "secret-key")
    post_count = 0

    def fake_request(method, path, **kwargs):
        nonlocal post_count
        if method == "GET":
            return []
        post_count += 1
        if post_count == 1:
            raise ServiceError(
                "ragflow: API error User 'internal-id' lacks permission for dataset 'financial-documents'"
            )
        return {"id": "dataset-1", "name": kwargs["payload"]["name"]}

    monkeypatch.setattr(client, "_request", fake_request)

    dataset, created = client.get_or_create_dataset("financial-documents")

    assert created is True
    assert dataset["id"] == "dataset-1"
    assert dataset["name"].startswith("financial-documents-")
    assert len(dataset["name"].rsplit("-", 1)[1]) == 6


def test_persistent_create_permission_error_is_sanitized(monkeypatch):
    client = RagflowClient("https://ragflow.example", "secret-key")

    def fake_request(method, path, **kwargs):
        if method == "GET":
            return []
        raise ServiceError(
            "ragflow: API error User 'internal-id' lacks permission for dataset 'private'"
        )

    monkeypatch.setattr(client, "_request", fake_request)

    try:
        client.get_or_create_dataset("financial-documents")
    except ServiceError as exc:
        message = str(exc)
    else:
        raise AssertionError("Ожидалась ошибка прав доступа")

    assert "API-ключ не имеет права" in message
    assert "internal-id" not in message
