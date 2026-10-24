import os
import uuid
from typing import Any

# Universal Langfuse import handling across v2, v3, and fallback
try:
    from langfuse import Langfuse, observe
except ImportError:
    try:
        from langfuse import Langfuse
        from langfuse.decorators import observe
    except ImportError:

        def observe(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        Langfuse = None

from legal_rag.retriever import LegalRetriever
from legal_rag.text_normalizer import normalize_arabic

# Environment variables for Langfuse client
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3001")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-local-legal-rag")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-local-legal-rag")

try:
    langfuse_client = Langfuse() if Langfuse else None
except Exception:  # noqa: BLE001
    langfuse_client = None


class TracedLegalRAG:
    """
    Production Traced RAG Pipeline instrumenting traces, spans,
    generations, and scores matching Handbook pg. 53.
    """

    def __init__(self, retriever: LegalRetriever = None):
        self.retriever = retriever or LegalRetriever()

    @observe(name="retrieve")
    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        norm_q = normalize_arabic(query)
        hits = self.retriever.search(norm_q, top_k=top_k)
        return hits

    @observe(name="rerank")
    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_n: int = 3
    ) -> list[dict[str, Any]]:
        """
        Topic-aware hybrid re-ranking scoring both statutory body text
        and legal hierarchy metadata (topic, chapter, citation).
        """
        query_terms = set(query.split())
        scored = []
        for cand in candidates:
            doc_text = cand.get("document", "")
            meta = cand.get("metadata", {})
            topic = meta.get("topic", "")
            chapter = meta.get("chapter", "")
            citation = meta.get("citation", "")

            combined_legal_text = f"{topic} {chapter} {citation} {doc_text}"
            legal_terms = set(combined_legal_text.split())

            overlap = len(query_terms.intersection(legal_terms)) / max(len(query_terms), 1)
            dense_score = cand.get("score", 0.5)

            # 50% Dense Vector Score + 50% Legal Concept & Statutory Match
            score = (dense_score * 0.5) + (overlap * 0.5)
            c = dict(cand)
            c["rerank_score"] = round(score, 4)
            scored.append(c)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]

    @observe(as_type="generation", name="generate")
    def generate(self, question: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
        primary = hits[0]
        sources = list(dict.fromkeys(h["metadata"]["citation"] for h in hits))

        answer = (
            f"بناءً على أحكام القانون المدني المصري ({primary['metadata']['citation']}):\n"
            f"{primary['document']}\n\n"
            f"السند القانوني المعتمد: {', '.join(sources)}."
        )

        prompt_tokens = (len(question) + sum(len(h["document"]) for h in hits)) // 3
        completion_tokens = len(answer) // 3

        return {
            "answer": answer,
            "sources": sources,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    @observe(name="arabic_legal_rag_query")
    def query(
        self,
        question: str,
        user_id: str = "attorney_cairo_01",
        session_id: str = "session_contract_review",
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        corr_id = correlation_id or str(uuid.uuid4())

        # 1. Retrieve Step (Fetch top 10 candidates from ChromaDB)
        candidates = self.retrieve(question, top_k=10)
        if not candidates:
            return {
                "answer": "لم يتم العثور على مواد قانونية مطابقة.",
                "sources": [],
                "correlation_id": corr_id,
            }

        # 2. Rerank Step (Topic-aware pruning to top 3 precision articles)
        hits = self.rerank(question, candidates, top_n=3)

        # 3. Generate Step (Synthesize grounded legal consultation)
        result = self.generate(question, hits)
        result["correlation_id"] = corr_id
        result["user_id"] = user_id
        result["session_id"] = session_id

        return result
