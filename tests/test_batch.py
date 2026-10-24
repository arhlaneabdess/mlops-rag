import json
from unittest.mock import MagicMock

import pytest

from legal_rag.batch import BatchReindexer


def test_batch_reindexer_flow(tmp_path):
    mock_retriever = MagicMock()
    mock_retriever.count.return_value = 100

    reindexer = BatchReindexer(retriever=mock_retriever)

    # Create dummy amendment file
    amendment_data = [
        {
            "article_number": 1150,
            "book": "BOOK I",
            "chapter": "CHAPTER I",
            "section": "Section I",
            "topic": "Electronic Commerce",
            "text_ar": "(1) تسري على العقود الإلكترونية ذات الأحكام المقررة للعقود المحررة كتابة.",
            "text_en": "Provisions apply to electronic contracts.",
            "source_page": 170,
            "citation": "Egyptian Civil Code, Article 1150 (Added 2026)",
        }
    ]
    file_path = tmp_path / "amendment.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(amendment_data, f)

    result = reindexer.reindex_amendments(file_path)

    assert result["articles_processed"] == 1
    assert result["chunks_indexed"] >= 1
    assert result["total_collection_count"] == 100
    assert mock_retriever.index_chunks.called


def test_batch_reindexer_missing_file(tmp_path):
    reindexer = BatchReindexer()
    with pytest.raises(FileNotFoundError):
        reindexer.reindex_amendments(tmp_path / "non_existent.json")
