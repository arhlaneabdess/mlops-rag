# Egyptian Civil Code Legal RAG API

A bilingual Arabic/English retrieval-augmented question-answering API for the Egyptian Civil Code. The FastAPI service returns an answer together with the legal article citation(s) used, and the repository includes a DVC pipeline for extracting, validating, chunking, and indexing the corpus.

## Quickstart

### Requirements

- Python 3.10 or newer
- Docker Engine and the Docker Compose plugin
- Git
- Python pip plus a virtual environment
- DVC with the S3 extra (`dvc[s3]`) for the corpus pipeline

The DVC remote is configured for the MinIO service in `docker/docker-compose.yml`. Start MinIO before running DVC. The first run may download model weights and build the vector index, so it can take several minutes and requires an internet connection. Docker must have enough memory for the embedding model.

### Run the service

From the repository root, run these steps in order:

1. Start the local services that provide DVC's S3-compatible storage:

   ```bash
   docker compose -f docker/docker-compose.yml up -d postgres minio
   ```

2. Install the development dependencies and reproduce the data pipeline:

   ```bash
   python -m venv .venv
   # Linux/macOS:
   source .venv/bin/activate
   # Windows PowerShell instead: .venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev]"
   dvc repro
   ```

3. Build and start the API:

   ```bash
   docker compose -f docker/docker-compose.yml up --build -d
   ```

### Make a request

Send a question in Arabic (or English) to the running API:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هو أثر العقد بين المتعاقدين؟", "top_k": 2}'
```

The response contains an answer, source citations, a correlation ID, and latency information. You can also check service health and metadata:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
```

> **Note:** The first-time setup depends on the configured DVC/MinIO storage and successful corpus processing. If `dvc repro` cannot access the configured remote, check that the MinIO container is healthy and that the remote credentials in `.dvc/config` match your local setup. Do not put real credentials in the repository.

## Repository layout

```text
mlops-rag/
├── .github/workflows/       # CI and continuous-training workflows
├── data/
│   ├── raw/                 # DVC-tracked source document
│   └── processed/           # Extracted corpus, chunks, and vector index
├── docker/
│   ├── Dockerfile           # Container image for the API
│   └── docker-compose.yml   # API, MLflow, PostgreSQL, and MinIO
├── infra/                   # Terraform infrastructure configuration
├── loadtest/                # Locust load-test scenarios
├── reports/                 # Module reports and evaluation artifacts
├── scripts/                 # Corpus parsing, chunking, and indexing
├── src/legal_rag/
│   ├── api/                 # FastAPI application
│   ├── config.py            # Application settings
│   ├── parser.py            # Source-document parsing
│   ├── retriever.py         # Vector retrieval
│   ├── schemas.py           # API request/response schemas
│   └── text_normalizer.py   # Arabic text normalization
├── tests/                   # API, corpus, and pipeline tests
├── dvc.yaml                 # Data pipeline stages
├── pyproject.toml           # Dependencies and tool configuration
└── README.md
```

## Development checks

Run the API tests and lint check from the repository root:

```bash
pytest
ruff check .
```

The test configuration enforces a minimum 70% coverage threshold. Run checks in your own environment and inspect the results before reporting them as passing.

## Stopping the services

```bash
docker compose -f docker/docker-compose.yml down
```
