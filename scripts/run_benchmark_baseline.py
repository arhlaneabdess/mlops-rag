from legal_rag.retriever import LegalRetriever
from optimization.benchmark import run_benchmark


def main():
    print("[*] Running Benchmark Harness for Row 0: Baseline (Unoptimized FP32)...")
    retriever = LegalRetriever()

    result = run_benchmark(
        search_fn=retriever.search,
        technique_name="Baseline (FP32 PyTorch)",
        hardware="CPU (x86_64)",
        warmup_iters=5,
        eval_iters=50,
    )

    print("\n" + "=" * 65)
    print("        ROW 0: UNOPTIMIZED BASELINE BENCHMARK")
    print("=" * 65)
    print(f"  Technique       : {result.technique}")
    print(f"  Hit Rate @ 3    : {result.hit_rate * 100:.1f}%")
    print(f"  Faithfulness    : {result.faithfulness:.4f}")
    print(f"  p50 Latency     : {result.p50_ms} ms")
    print(f"  p95 Latency     : {result.p95_ms} ms")
    print(f"  Throughput      : {result.throughput_qps} QPS")
    print(f"  Database Size   : {result.size_mb} MB")
    print(f"  Monthly Cost    : ${result.monthly_cost_1k_req_usd:,} / mo (for 1,000 req/s)")
    print("=" * 65)
    print("[✓] Baseline successfully recorded in optimization/results.csv")


if __name__ == "__main__":
    main()