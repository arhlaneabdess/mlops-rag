from unittest.mock import MagicMock

from legal_rag.tracing import TracedLegalRAG


def test_traced_legal_rag_flow():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "chunk_id": "art_147_p1",
            "document": "العقد شريعة المتعاقدين",
            "metadata": {
                "citation": "Egyptian Civil Code, Article 147",
                "topic": "العقود",
                "chapter": "مصادر الالتزام",
            },
            "score": 0.85,
        },
        {
            "chunk_id": "art_148_p1",
            "document": "يجب تنفيذ العقد طبقاً لما اشتمل عليه وبحسن نية",
            "metadata": {
                "citation": "Egyptian Civil Code, Article 148",
                "topic": "العقود",
                "chapter": "آثار العقد",
            },
            "score": 0.70,
        },
    ]

    pipeline = TracedLegalRAG(retriever=mock_retriever)
    result = pipeline.query("ما هو حكم العقد؟", user_id="test_user", correlation_id="test_corr_01")

    assert "Egyptian Civil Code, Article 147" in result["sources"]
    assert "العقد شريعة المتعاقدين" in result["answer"]
    assert result["correlation_id"] == "test_corr_01"


def test_traced_legal_rag_empty_candidates():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    pipeline = TracedLegalRAG(retriever=mock_retriever)
    result = pipeline.query("سؤال غير موجود")
    assert result["sources"] == []
    assert "لم يتم العثور" in result["answer"]
