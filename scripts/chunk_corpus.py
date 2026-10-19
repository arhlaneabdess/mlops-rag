import json

from legal_rag.chunker import LegalArticleChunker
from legal_rag.config import settings
from legal_rag.schemas import ArticleRecord


def main():
    print(f"[*] Reading articles from {settings.PROCESSED_JSON_PATH}...")
    if not settings.PROCESSED_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Missing {settings.PROCESSED_JSON_PATH}. Run parse_civil_code first."
        )

    with open(settings.PROCESSED_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = [ArticleRecord(**item) for item in data]
    chunker = LegalArticleChunker()

    all_chunks = []
    for art in articles:
        chunks = chunker.chunk_article(art)
        all_chunks.extend(chunks)

    settings.CHUNKS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_chunks], f, ensure_ascii=False, indent=2)

    print(
        f"[✓] Created {len(all_chunks)} chunks from {len(articles)} articles. Saved to {settings.CHUNKS_JSON_PATH}"
    )


if __name__ == "__main__":
    main()
