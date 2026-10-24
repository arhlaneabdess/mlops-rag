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


def test_api_correlation_id_valid():
    client = TestClient(app)
    custom_id = "custom-trace-id-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers["X-Request-ID"] == custom_id
    assert "X-Response-Time-MS" in response.headers


def test_api_correlation_id_missing():
    client = TestClient(app)
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) >= 32  # Valid generated UUID4


def test_api_correlation_id_sanitization_disallowed_and_overlong():
    client = TestClient(app)
    # Disallowed characters (XSS injection attempt)
    bad_id = "<script>alert(1)</script>"
    res_bad = client.get("/health", headers={"X-Request-ID": bad_id})
    assert res_bad.headers["X-Request-ID"] != bad_id
    assert len(res_bad.headers["X-Request-ID"]) >= 32  # Generated fresh UUID

    # Overlong value (> 64 chars)
    overlong_id = "a" * 128
    res_long = client.get("/health", headers={"X-Request-ID": overlong_id})
    assert res_long.headers["X-Request-ID"] != overlong_id

    # Whitespace only
    res_space = client.get("/health", headers={"X-Request-ID": "   "})
    assert res_space.headers["X-Request-ID"] != "   "


def test_api_correlation_id_on_validation_error():
    client = TestClient(app)
    custom_id = "trace-err-99"
    # Empty question triggers 422 validation error
    response = client.post("/ask", json={"question": ""}, headers={"X-Request-ID": custom_id})
    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == custom_id
    assert "X-Response-Time-MS" in response.headers
