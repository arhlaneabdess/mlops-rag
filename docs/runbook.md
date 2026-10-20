# Legal RAG Production On-Call Runbook (3am-Proof)

**Service:** Egyptian Civil Code Legal RAG Service (`legal-rag-api` & `arabic_legal_rag_service`)  
**Repository:** `mlops-rag`  
**Escalation Lead:** On-Call Engineering Lead / Legal Compliance Officer

---

## 1. Alert Runbooks (1 Section per Prometheus Alert)

### ALERT 1: `LegalRAGServiceDown` (Severity: CRITICAL)

- **What it means:** The API container has stopped responding to health checks on port 8000.
- **First command to run:**

  ```bash
  docker ps -a --filter "name=legal-rag-api" && docker logs legal-rag-api --tail 50
  ```

- **How to tell a real problem from a transient blip:** If the container status is `Exited` or `Restarting`, inspect the traceback. If it is a memory allocation crash (OOM), check host RAM using `free -h`.
- **Immediate mitigation / rollback:**

  ```bash
  docker compose -f docker/docker-compose.yml up -d --force-recreate legal-rag-api
  ```

- **When to escalate:** If the container restarts and crashes again immediately due to corrupted ChromaDB storage or disk failure.

### ALERT 2: `P95LatencyAboveSLA` (Severity: CRITICAL)

- **What it means:** Tail request latency (p95) has exceeded the 5.0-second SLA for more than 2 minutes.
- **First command to run:**

  ```bash
  docker top legal-rag-api
  ```

- **How to diagnose:** Check whether incoming request concurrency is saturating CPU cores (compute-bound) or whether a large multi-paragraph article is causing generation bottlenecks.
- **Immediate mitigation / rollback:** Shift traffic 100% back to stable in under 20 milliseconds:

  ```bash
  ./serving/rollback.sh
  ```

- **When to escalate:** If latency remains above 5 seconds after rollback, indicating a host-level noisy neighbor or CPU throttling.

### ALERT 3: `HighErrorRate` (Severity: CRITICAL)

- **What it means:** More than 5% of incoming user queries are returning HTTP 5xx errors.
- **First command to run:**

  ```bash
  docker logs legal-rag-api --tail 100 | grep -i "error"
  ```

- **How to diagnose:** Determine whether the errors are database connection drops (PostgreSQL down) or Pydantic validation exceptions.
- **Immediate mitigation:** Restart the database backend:

  ```bash
  docker compose -f docker/docker-compose.yml restart postgres
  ```

- **When to escalate:** If the database is healthy but the application throws internal unhandled exceptions on standard Arabic inputs.

### ALERT 4: `PromptInjectionDetected` (Severity: WARNING)

- **What it means:** An adversarial prompt injection attempt (“تجاهل التعليمات” or system prompt reveal) was blocked by the input guardrail.
- **First command to run:** Open Langfuse at `http://localhost:3001` and search for the `prompt_injection` tag.
- **How to diagnose:** Check the user ID and IP address of the query session. If isolated, the guardrail successfully protected the system. If it comes from multiple IPs, a coordinated jailbreak attack may be underway.
- **When to escalate:** If an adversarial prompt successfully bypassed the guardrail and caused the LLM to output non-legal system instructions.

### ALERT 5: `HighTokenConsumptionRate` (Severity: WARNING)

- **What it means:** Token burn rate exceeded 500 tokens/sec for 5 minutes, risking a budget blowout.
- **First command to run:** Open Grafana at `http://localhost:3002` and select **Row 3 (Token Usage)**.
- **How to diagnose:** Check whether prompt tokens or completion tokens are spiking. If prompt tokens are high, verify that the re-ranker is properly pruning context to the top 2.
- **When to escalate:** If prompt token usage continues climbing unchecked under normal request volume.

### ALERT 6: `EmbeddingDriftDetected` (Severity: WARNING)

- **What it means:** User query vocabulary has drifted significantly from the 1948 Civil Code semantic centroid (cosine distance > 0.45).
- **First command to run:**

  ```bash
  python -c "import psycopg2, pprint; conn=psycopg2.connect(host='localhost',port=5432,user='mlflow',password='mlflowpassword',dbname='mlflowdb'); cur=conn.cursor(); cur.execute('SELECT dataset, value, threshold, computed_at FROM drift_metrics ORDER BY id DESC LIMIT 5;'); pprint.pprint(cur.fetchall())"
  ```

- **How to diagnose:** Check which dataset drifted (e.g., Family Law vs. Criminal Law). If users are asking about unindexed modern laws, the corpus needs new amendments ingested.
- **Remediation:** Ingest the new statutory amendment via the batch pipeline:

  ```bash
  python -m scripts.batch_reindex
  ```

- **When to escalate:** If drift persists across all user traffic, indicating a permanent semantic shift in client demographics.

## 2. The Symptom → Cause Diagnosis Table (Handbook Step 9/16)

| Observed Symptom | Candidate Causes | Metric That Discriminates |
|---|---|---|
| p95 latency spikes, throughput flat | CPU-bound transformer queuing vs. slow DB lookup vs. in-flight lock | Inspect `rag_stage_latency_seconds`: if retrieval is slow, suspect a ChromaDB lock; if generation is slow, suspect CPU compute. |
| Detections / citations missing | Embedding semantic drift vs. OCR parsing bug vs. over-pruned re-ranking | Compare cosine drift score in PostgreSQL vs. re-ranker candidate count. |
| High error rate after deploy | Missing environment variable vs. schema change vs. PostgreSQL connection | Inspect the `docker logs legal-rag-api` traceback. |
| RAM usage climbing continuously | Memory leak in tokenizer vs. ChromaDB cache bloat | Run a `psutil` RSS check; inspect container memory in Docker. |

## 3. Ten Monitoring Anti-Patterns to Avoid (Handbook Step 16.4)

1. **High-cardinality Prometheus labels:** Putting `user_id`, `session_id`, or `query_text` into Prometheus metrics can cause TSDB memory explosion and crash Prometheus. Keep these in Langfuse and structured logs.
2. **Storing low-frequency metrics in Prometheus:** Daily drift scores belong in PostgreSQL, not Prometheus.
3. **Alerting on statistical significance at high N:** A KS test will report \(p < 0.001\) on 100,000 queries with an irrelevant 0.1% change. Always alert on magnitude measures (cosine distance, Wasserstein distance).
4. **Alerts with no first action:** An alert that says “Drift detected” with no immediate triage command gets muted and ignored.
5. **Dashboards clicked together in the UI:** A dashboard created by manual UI clicks is lost on the next machine. Always provision dashboards as code from Git.
6. **Optimizing for mean latency:** Mean latency hides tail spikes. Always monitor p95 and p99.
7. **Trusting an uncalibrated LLM judge:** LLM judges have verbosity and position biases. Always calibrate against human labels first.
8. **Monitoring only HTTP status codes for LLMs:** A catastrophic legal hallucination can return HTTP 200. Monitor RAGAS faithfulness and groundedness guardrails.
9. **Auto-promoting models directly to production:** Legal and medical AI carries high liability. Staging can be automated; production must require human approval.
10. **Writing tests only after code:** Tests written post-facto assert whatever the code already does, catching zero regressions.

---