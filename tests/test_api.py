import uuid

import pytest
from fastapi.testclient import TestClient

from legal_rag.api.main import CORPUS_STORE, app
from legal_rag.schemas import ArticleRecord


@pytest.fixture(autouse=True)
def setup_corpus():
    CORPUS_STORE[147] = ArticleRecord(
        article_number=147,
        book="Obligations",
        chapter="Contracts",
        section="Effects",
        topic="General",
        text_ar="العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله.",
        text_en="The contract makes the law of the parties.",
        is_repealed=False,
        source_page=34,
        citation="Egyptian Civil Code, Article 147",
    )
    yield
    CORPUS_STORE.clear()


def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_metadata():
    client = TestClient(app)
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["total_articles"] == 1


def test_api_ask_valid():
    client = TestClient(app)
    response = client.post("/ask", json={"question": "ما هو العقد؟", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data
    assert "Egyptian Civil Code, Article 147" in data["sources"]
    assert "X-Request-ID" in response.headers


def test_request_id_preserves_valid_incoming_id():
    client = TestClient(app)
    request_id = "trace-123.ab:456_test"
    response = client.get("/metadata", headers={"X-Request-ID": request_id})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id
    assert float(response.headers["X-Response-Time-MS"]) >= 0


def test_request_id_is_generated_when_missing():
    client = TestClient(app)
    response = client.get("/metadata")

    assert response.status_code == 200
    generated_id = response.headers["X-Request-ID"]
    assert str(uuid.UUID(generated_id)) == generated_id
    assert float(response.headers["X-Response-Time-MS"]) >= 0


@pytest.mark.parametrize("request_id", ["bad id", "x" * 129, "contains/slash"])
def test_request_id_is_replaced_when_malformed(request_id):
    client = TestClient(app)
    response = client.get("/metadata", headers={"X-Request-ID": request_id})

    assert response.status_code == 200
    generated_id = response.headers["X-Request-ID"]
    assert generated_id != request_id
    assert str(uuid.UUID(generated_id)) == generated_id


def test_request_id_headers_are_present_on_validation_error():
    client = TestClient(app)
    response = client.post(
        "/ask",
        json={"question": ""},
        headers={"X-Request-ID": "validation-test-1"},
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "validation-test-1"
    assert float(response.headers["X-Response-Time-MS"]) >= 0
    assert "detail" in response.json()


def test_api_ask_validation_error():
    client = TestClient(app)
    # Empty question triggers 422
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_api_health_unhealthy():
    CORPUS_STORE.clear()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
