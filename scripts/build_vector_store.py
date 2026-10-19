import json

from legal_rag.config import settings
from legal_rag.retriever import LegalRetriever
from legal_rag.schemas import ArticleChunk


def main():
    print(f"[*] Reading chunked provisions from {settings.CHUNKS_JSON_PATH}...")
    if not settings.CHUNKS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Missing {settings.CHUNKS_JSON_PATH}. Run python -m scripts.chunk_corpus first."
        )

    with open(settings.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [ArticleChunk(**item) for item in data]
    print(f"[*] Loaded {len(chunks)} chunks.")

    print(f"[*] Initializing ChromaDB and {settings.EMBEDDING_MODEL_NAME}...")
    retriever = LegalRetriever()

    print(
        "[*] Generating embeddings and indexing into ChromaDB (this may take a couple of minutes)..."
    )
    retriever.index_chunks(chunks, batch_size=64)

    total_indexed = retriever.count()
    print(
        f"[✓] Successfully indexed {total_indexed} chunks in ChromaDB at {settings.CHROMA_DB_DIR}"
    )


if __name__ == "__main__":
    main()
