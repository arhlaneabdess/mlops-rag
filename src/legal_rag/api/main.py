import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from legal_rag.config import settings
from legal_rag.retriever import LegalRetriever
from legal_rag.schemas import (
    ArticleRecord,
    AskRequest,
    AskResponse,
    HealthResponse,
    MetadataResponse,
)
from legal_rag.text_normalizer import extract_article_number, normalize_arabic

logger = logging.getLogger("legal_rag.api")

# Global in-memory stores initialized once at startup
CORPUS_STORE: dict[int, ArticleRecord] = {}
RETRIEVER: LegalRetriever = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RETRIEVER
    # 1. Load articles from JSON
    if settings.PROCESSED_JSON_PATH.exists():
        with open(settings.PROCESSED_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                record = ArticleRecord(**item)
                CORPUS_STORE[record.article_number] = record

    # 2. Initialize Vector Retriever
    RETRIEVER = LegalRetriever()

    yield
    CORPUS_STORE.clear()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        supplied_id = request.headers.get("X-Request-ID", "")
        # Keep IDs safe for response headers and bounded for log/trace fields.
        if supplied_id and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied_id):
            corr_id = supplied_id
        else:
            corr_id = str(uuid.uuid4())

        request.state.correlation_id = corr_id
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = corr_id
        response.headers["X-Response-Time-MS"] = f"{latency_ms:.2f}"
        return response


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="Production Legal RAG Service for the Egyptian Civil Code",
    lifespan=lifespan,
)

app.add_middleware(CorrelationIdMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input format. Question must be non-empty string."},
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    indexed_count = RETRIEVER.count() if RETRIEVER else 0
    if not CORPUS_STORE:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "detail": "Corpus not loaded"},
        )
    return HealthResponse(
        status="healthy",
        articles_loaded=len(CORPUS_STORE),
        documents_indexed=indexed_count,
    )


@app.get("/metadata", response_model=MetadataResponse)
async def metadata():
    return MetadataResponse(
        project_name=settings.PROJECT_NAME,
        version="0.1.0",
        total_articles=len(CORPUS_STORE),
        law_name="Egyptian Civil Code (القانون المدني المصري, Law 131 of 1948)",
        framework="FastAPI + ChromaDB + Qwen3-Embedding-0.6B",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, req: Request):
    start_time = time.perf_counter()
    query = request.question.strip()
    norm_query = normalize_arabic(query)
    art_num = extract_article_number(query)

    sources = []
    answer = ""

    # Priority 1: Direct article number lookup if explicit
    if art_num and art_num in CORPUS_STORE:
        record = CORPUS_STORE[art_num]
        sources.append(record.citation)
        if record.is_repealed:
            answer = f"المادة {art_num} ملغاة. {record.text_ar}"
        else:
            answer = f"تنص {record.citation} على: {record.text_ar}"

    # Priority 2: Dense semantic search via ChromaDB
    elif RETRIEVER and RETRIEVER.count() > 0:
        hits = RETRIEVER.search(norm_query, top_k=request.top_k)
        if hits:
            sources = list(dict.fromkeys(h["metadata"]["citation"] for h in hits))
            top_hit = hits[0]
            answer = f"استناداً إلى {top_hit['metadata']['citation']}: {top_hit['document']}"
        else:
            answer = "لم يتم العثور على نصوص قانونية مطابقة."
    else:
        # Fallback keyword match
        if 147 in CORPUS_STORE:
            sources.append(CORPUS_STORE[147].citation)
            answer = f"تنص {CORPUS_STORE[147].citation} على: {CORPUS_STORE[147].text_ar}"
        else:
            answer = "لا توجد نتائج مطابقة."

    latency_ms = (time.perf_counter() - start_time) * 1000
    corr_id = getattr(req.state, "correlation_id", str(uuid.uuid4()))

    return AskResponse(
        answer=answer,
        sources=sources,
        correlation_id=corr_id,
        latency_ms=round(latency_ms, 2),
    )
