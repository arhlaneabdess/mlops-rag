# Module 5 Lab Report: Production Observability, Drift Detection & Guardrails

## 1. Raw Prometheus Exposition Format Annotation (Handbook pg. 45)

```text
# HELP rag_requests_total Total count of incoming legal queries
# TYPE rag_requests_total counter
rag_requests_total{endpoint="/ask",status="200"} 1.0
# HELP rag_requests_created Total count of incoming legal queries
# TYPE rag_requests_created gauge
rag_requests_created{endpoint="/ask",status="200"} 1.79243976499462e+09
# HELP rag_stage_latency_seconds Latency breakdown across pipeline stages
# TYPE rag_stage_latency_seconds histogram
rag_stage_latency_seconds_bucket{le="0.05",stage="retrieval"} 0.0
rag_stage_latency_seconds_bucket{le="0.1",stage="retrieval"} 0.0
rag_stage_latency_seconds_bucket{le="0.25",stage="retrieval"} 0.0
rag_stage_latency_seconds_bucket{le="0.5",stage="retrieval"} 0.0
rag_stage_latency_seconds_bucket{le="1.0",stage="retrieval"} 0.0
rag_stage_latency_seconds_bucket{le="2.5",stage="retrieval"} 0.0
rag_stage_latency_seconds_bucket{le="5.0",stage="retrieval"} 0.0
rag_stage_latency_seconds_bucket{le="10.0",stage="retrieval"} 1.0
rag_stage_latency_seconds_bucket{le="+Inf",stage="retrieval"} 1.0
rag_stage_latency_seconds_count{stage="retrieval"} 1.0
rag_stage_latency_seconds_sum{stage="retrieval"} 6.968303367999994
rag_stage_latency_seconds_bucket{le="0.05",stage="generation"} 1.0
rag_stage_latency_seconds_bucket{le="0.1",stage="generation"} 1.0
rag_stage_latency_seconds_bucket{le="0.25",stage="generation"} 1.0
rag_stage_latency_seconds_bucket{le="0.5",stage="generation"} 1.0
rag_stage_latency_seconds_bucket{le="1.0",stage="generation"} 1.0
rag_stage_latency_seconds_bucket{le="2.5",stage="generation"} 1.0
rag_stage_latency_seconds_bucket{le="5.0",stage="generation"} 1.0
rag_stage_latency_seconds_bucket{le="10.0",stage="generation"} 1.0
rag_stage_latency_seconds_bucket{le="+Inf",stage="generation"} 1.0
rag_stage_latency_seconds_count{stage="generation"} 1.0
rag_stage_latency_seconds_sum{stage="generation"} 8.018500000162021e-05
rag_stage_latency_seconds_bucket{le="0.05",stage="total"} 0.0
rag_stage_latency_seconds_bucket{le="0.1",stage="total"} 0.0
rag_stage_latency_seconds_bucket{le="0.25",stage="total"} 0.0
rag_stage_latency_seconds_bucket{le="0.5",stage="total"} 0.0
rag_stage_latency_seconds_bucket{le="1.0",stage="total"} 0.0
rag_stage_latency_seconds_bucket{le="2.5",stage="total"} 0.0
rag_stage_latency_seconds_bucket{le="5.0",stage="total"} 0.0
rag_stage_latency_seconds_bucket{le="10.0",stage="total"} 1.0
rag_stage_latency_seconds_bucket{le="+Inf",stage="total"} 1.0
rag_stage_latency_seconds_count{stage="total"} 1.0
rag_stage_latency_seconds_sum{stage="total"} 6.968700199000011
# HELP rag_stage_latency_seconds_created Latency breakdown across pipeline stages
# TYPE rag_stage_latency_seconds_created gauge
rag_stage_latency_seconds_created{stage="retrieval"} 1.7924397580258863e+09
rag_stage_latency_seconds_created{stage="generation"} 1.7924397649943388e+09
rag_stage_latency_seconds_created{stage="total"} 1.7924397649945698e+09
# HELP rag_tokens_total Total tokens consumed for cost tracking
# TYPE rag_tokens_total counter
rag_tokens_total{token_type="prompt"} 5.0
rag_tokens_total{token_type="completion"} 169.0
# HELP rag_tokens_created Total tokens consumed for cost tracking
# TYPE rag_tokens_created gauge
rag_tokens_created{token_type="prompt"} 1.7924397649945138e+09
rag_tokens_created{token_type="completion"} 1.7924397649945369e+09
# HELP rag_active_requests Current in-flight concurrent requests
# TYPE rag_active_requests gauge
rag_active_requests{endpoint="/ask"} 0.0
# HELP rag_guardrail_rejections_total Security rejections for prompt injection or PII
# TYPE rag_guardrail_rejections_total counter