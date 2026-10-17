# Module 3 Lab Report: Serving RAG Three Ways & Safe Release Strategies

## 1. Serving Vocabulary in Production
- **Latency vs. Throughput:** Latency is the turnaround time for a single legal query (measured from request arrival to the first token emitted, TTFT). Throughput is the total volume of requests or tokens served concurrently per second across the entire cluster.
- **Percentiles (p50 / p95 / p99):** Mean latency is deceptive in legal search because outliers (queries retrieving complex 5-paragraph articles) are masked by simple single-article lookups. Optimizing for p95/p99 ensures the slowest 5% of lawyers and citizens still receive responsive answers under our SLA.
- **Cold Start:** The time required to initialize the Python process, load `Qwen3-Embedding-0.6B` and `Qwen2.5-Instruct` into memory/VRAM, and warm up the ChromaDB HNSW graph before the first request can be served.
- **Batch Size & Micro-batching:** In web services, incoming user requests arrive individually (batch size = 1). Adaptive micro-batching dynamically groups queries arriving within a small time window (`max_latency_ms=20ms`) into a single tensor batch to maximize CPU/GPU compute efficiency without human-perceptible delay.

---

## 2. The Three Serving Failure Modes for Legal RAG

1. **Cost Blowout:** Deploying an unquantized 70B parameter model running 24/7 on dedicated multi-A100 instances to answer infrequent legal lookups that a quantized Qwen model on an optimized server handles at 1/20th the infrastructure cost.
2. **Latency SLA Breach:** Loading model weights or parsing legal PDF articles per-request inside the FastAPI handler rather than at application startup, causing p95 latency to explode from 400ms to over 15 seconds.
3. **Over-Engineering:** Implementing a full Kafka streaming cluster and distributed vector partitions for a low-traffic internal law firm application where an asynchronous BentoML web service and SQLite/ChromaDB satisfy all query needs.

---

## 3. Inference Pattern Decision Table (Legal Domain)

| Scenario | Inference Pattern | Driving SLA | Rationale |
| :--- | :--- | :--- | :--- |
| **(a) Citizen / Lawyer Interactive Chat** | Real-Time Web Service (BentoML + vLLM) | TTFT < 800ms, streaming tokens | User is actively waiting in browser/app; requires streaming tokens for high perceived responsiveness. |
| **(b) Weekly Re-indexing of Court Amendments** | Nightly Batch Job (Airflow / Script) | Completion within 4-hour window | High throughput, offline execution; reads raw text partitions and rebuilds embeddings at 50–100× lower cost per document than a live API. |
| **(c) Live Legislation Alert Stream** | Streaming Inference (Redis Streams / Consumer Group) | Event-to-decision < 2s | Ingests continuous legal gazette feeds, runs similarity against saved client alerts, and emits notifications asynchronously with at-least-once delivery. |

---

## 4. The 4-Layer Serving Stack Architecture

```text
Layer 4: Ingress & Reverse Proxy       --> Nginx (Port 80/443, SSL, Canary 90/10 Routing)
Layer 3: API & Protocol Layer          --> BentoML / FastAPI (REST, SSE Token Streaming, Request Validation)
Layer 2: Inference Engine / Worker     --> vLLM (PagedAttention, Continuous Batching) + ChromaDB Retriever
Layer 1: Hardware & Compute Layer      --> CPU / NVIDIA Tensor Cores (Optimized FP16 / INT8 Kernels)

## 5. Locust Load Test Results (50 Concurrent Users)

**Pass Condition Met:** HTML report saved to `reports/locust_report.html`.

| Endpoint | Requests | Failures | p50 | p95 | p99 | RPS | Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `POST /ask` | 57 | 0 (0.0%) | 6,000 ms | 46,000 ms | 52,000 ms | 0.92 | **CPU-bound (Transformer forward pass queuing)** |

### Bottleneck Diagnosis & Evidence
- **Utilization Evidence:** CPU utilization was pinned near 100% across all available cores while memory remained stable and below 2 GB.
- **Root Cause:** In the absence of an NVIDIA GPU, running dense 600M parameter embeddings on CPU causes incoming concurrent requests to wait in the server worker queue, driving p95 latency up from 6s to 46s.
- **Production Solution:** Shifting inference to a GPU with vLLM PagedAttention or applying INT8 quantization (Module 4) will dramatically reduce p95 latency under high concurrency.

## 7. Deployment & Release Strategy Comparison Table (Handbook pg. 30)

| Strategy | Traffic Mechanics | Risk Reduced | Infrastructure Cost | Real-World Company Example |
| :--- | :--- | :--- | :--- | :--- |
| **Canary Release** | Weighted reverse proxy splits live traffic (e.g. 90/10 -> 50/50 -> 100) | Limits blast radius of runtime latency/memory leaks to a small fraction of users | Low (runs both models side-by-side with shared capacity) | **Netflix** (Kayenta automated canary analysis) |
| **Blue/Green** | Router switches 100% of traffic from Environment Blue to Green instantaneously | Eliminates service downtime during version deployment | High (requires 2× identical production infrastructure) | **Amazon** (Elastic Beanstalk Blue/Green) |
| **A/B Testing** | User-segmented routing (e.g., hash of user ID) measuring business metrics | Business metric degradation (conversion, search click-through) | Moderate | **Booking.com** (continuous feature & model experimentation) |
| **Shadow Mode (Mirroring)** | Ingress proxy mirrors asynchronous copy of live requests to candidate offline | 100% zero risk to users (candidate response is discarded, stable response returned) | Moderate to High (must compute predictions twice for every request) | **Uber Michelangelo** (shadowing ride-pricing and ETA models) |

### Why Shadow Mode is the Safest Way to Validate Model Changes
In high-liability domains like law or medical diagnosis, a newly trained candidate model might pass synthetic offline evaluations but fail silently on real-world edge cases. Shadow mode duplicates live production traffic to the candidate without serving its answers to users. Engineers can evaluate candidate predictions, latency, and hallucination rates against real legal queries with **absolute zero risk** of user impact.

## 8. vLLM Serving Architecture: PagedAttention & Continuous Batching (Handbook pg. 29)

### 1. PagedAttention
Standard transformer generation allocates contiguous memory blocks for the Key-Value (KV) cache based on the maximum possible sequence length. This leads to 60–80% memory fragmentation and limits concurrency. PagedAttention divides the KV cache into non-contiguous virtual memory blocks (similar to paging in OS kernels), allowing dynamic allocation without memory waste and unlocking up to 4× higher concurrency.

### 2. Continuous (Iteration-Level) Batching
Traditional web batching waits for a batch of requests to arrive and processes them together until the longest sequence completes, causing shorter requests to stall. Continuous batching operates at the token-iteration level: as soon as one sequence emits an `<EOS>` end token, its slot is immediately freed and a new waiting request is inserted mid-flight.

### 3. Why a Naive FastAPI + Transformers Loop Cannot Compete
1. **The Python GIL:** Running eager autoregressive generation in FastAPI locks the Python Global Interpreter Lock, degrading API concurrency.
2. **Static Padding Overhead:** In a naive loop, variable-length legal queries must be padded to the longest sequence in the batch, wasting compute cycles on dummy pad tokens.
3. **No KV Cache Management:** Naive loops reallocate KV tensors per request, exhausting GPU VRAM rapidly under multi-user concurrency.