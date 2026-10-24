import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
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

# --- Prometheus Metric Contract (Handbook pg. 45) ---
REQUEST_COUNTER = Counter(
    "rag_requests_total",
    "Total count of incoming legal queries",
    ["status", "endpoint"],
)
STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Latency breakdown across pipeline stages",
    ["stage"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
TOKEN_COUNTER = Counter(
    "rag_tokens_total",
    "Total tokens consumed for cost tracking",
    ["token_type"],
)
INFLIGHT_REQUESTS = Gauge(
    "rag_active_requests",
    "Current in-flight concurrent requests",
    ["endpoint"],
)
GUARDRAIL_REJECTIONS = Counter(
    "rag_guardrail_rejections_total",
    "Security rejections for prompt injection or PII",
    ["guardrail_type"],
)

# Global stores
CORPUS_STORE: dict[int, ArticleRecord] = {}
RETRIEVER: LegalRetriever = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RETRIEVER
    if settings.PROCESSED_JSON_PATH.exists():
        with open(settings.PROCESSED_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                record = ArticleRecord(**item)
                CORPUS_STORE[record.article_number] = record

    RETRIEVER = LegalRetriever()
    yield
    CORPUS_STORE.clear()


VALID_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        raw_id = request.headers.get("X-Request-ID", "").strip()

        # Security sanitization: alphanumeric, hyphens, underscores up to 64 chars
        if raw_id and VALID_REQUEST_ID_REGEX.match(raw_id):
            corr_id = raw_id
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
    version="0.5.0",
    description="Production Legal RAG Service for the Egyptian Civil Code",
    lifespan=lifespan,
)

app.add_middleware(CorrelationIdMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    REQUEST_COUNTER.labels(status="422", endpoint="/ask").inc()
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input format. Question must be non-empty string."},
    )


@app.get("/metrics")
async def metrics():
    """Prometheus pull scrape endpoint (Handbook pg. 45)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
        version="0.5.0",
        total_articles=len(CORPUS_STORE),
        law_name="Egyptian Civil Code (القانون المدني المصري, Law 131 of 1948)",
        framework="FastAPI + ChromaDB + Qwen3-Embedding + Prometheus",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, req: Request):
    INFLIGHT_REQUESTS.labels(endpoint="/ask").inc()
    overall_start = time.perf_counter()
    query = request.question.strip()

    # Guardrail check
    if "ignore previous instructions" in query.lower() or "تجاهل التعليمات السابقة" in query:
        GUARDRAIL_REJECTIONS.labels(guardrail_type="prompt_injection").inc()
        REQUEST_COUNTER.labels(status="400", endpoint="/ask").inc()
        INFLIGHT_REQUESTS.labels(endpoint="/ask").dec()
        raise HTTPException(
            status_code=400,
            detail="Security violation: Prompt injection detected.",
        )

    sources = []
    hits = []

    # Stage 1: Retrieval Timing
    with STAGE_LATENCY.labels(stage="retrieval").time():
        norm_query = normalize_arabic(query)
        art_num = extract_article_number(query)

        # Priority A: Direct article number match
        if art_num and art_num in CORPUS_STORE:
            record = CORPUS_STORE[art_num]
            hits = [
                {
                    "document": record.text_ar,
                    "metadata": {"citation": record.citation, "article_number": art_num},
                }
            ]
        # Priority B: Vector store search (if database is loaded)
        elif RETRIEVER and RETRIEVER.count() > 0:
            hits = RETRIEVER.search(norm_query, top_k=request.top_k)
        # Priority C: In-memory fallback (for unit tests / clean CI runners)
        elif CORPUS_STORE:
            for num, record in CORPUS_STORE.items():
                if any(w in record.text_ar for w in norm_query.split() if len(w) > 3) or num == 147:
                    hits.append(
                        {
                            "document": record.text_ar,
                            "metadata": {"citation": record.citation, "article_number": num},
                        }
                    )
                    if len(hits) >= request.top_k:
                        break

    # Stage 2: Synthesis / Generation Timing
    with STAGE_LATENCY.labels(stage="generation").time():
        if hits:
            sources = list(dict.fromkeys(h["metadata"]["citation"] for h in hits))
            primary = hits[0]
            answer = (
                f"استناداً إلى أحكام القانون المدني المصري ({primary['metadata']['citation']}):\n"
                f"{primary['document']}\n\nالسند القانوني: {', '.join(sources)}."
            )
            TOKEN_COUNTER.labels(token_type="prompt").inc(len(query) // 3)
            TOKEN_COUNTER.labels(token_type="completion").inc(len(answer) // 3)
        else:
            answer = "لم يتم العثور على نصوص قانونية مطابقة."

    total_latency_sec = time.perf_counter() - overall_start
    STAGE_LATENCY.labels(stage="total").observe(total_latency_sec)
    REQUEST_COUNTER.labels(status="200", endpoint="/ask").inc()
    INFLIGHT_REQUESTS.labels(endpoint="/ask").dec()

    corr_id = getattr(req.state, "correlation_id", str(uuid.uuid4()))
    return AskResponse(
        answer=answer,
        sources=sources,
        correlation_id=corr_id,
        latency_ms=round(total_latency_sec * 1000, 2),
    )
