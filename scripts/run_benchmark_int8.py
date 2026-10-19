import torch

from legal_rag.retriever import LegalRetriever
from optimization.benchmark import run_benchmark


class QuantizedLegalRetriever(LegalRetriever):
    """
    Subclass that applies PyTorch dynamic INT8 quantization to linear layers
    of the underlying transformer backbone.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[*] Applying Dynamic INT8 Quantization to linear layers...")
        # Access underlying HuggingFace transformer inside SentenceTransformer
        transformer_module = self.model._first_module().auto_model

        # Quantize torch.nn.Linear layers to torch.qint8
        self.quantized_backbone = torch.ao.quantization.quantize_dynamic(
            transformer_module, {torch.nn.Linear}, dtype=torch.qint8
        )
        # Point the sentence-transformer to use the quantized backbone
        self.model._first_module().auto_model = self.quantized_backbone
        print("[✓] Model weights successfully quantized to INT8.")


def main():
    print("[*] Initializing Quantized INT8 Legal Retriever...")
    quantized_retriever = QuantizedLegalRetriever()

    print("[*] Running Benchmark Harness for Row 1: INT8 Dynamic Quantization...")
    result = run_benchmark(
        search_fn=quantized_retriever.search,
        technique_name="PTQ INT8 Dynamic (PyTorch)",
        hardware="CPU (x86_64)",
        warmup_iters=5,
        eval_iters=50,
    )

    print("\n" + "=" * 65)
    print("      ROW 1: PTQ INT8 DYNAMIC QUANTIZATION BENCHMARK")
    print("=" * 65)
    print(f"  Technique       : {result.technique}")
    print(f"  Hit Rate @ 3    : {result.hit_rate * 100:.1f}%")
    print(f"  Faithfulness    : {result.faithfulness:.4f} (Constraint: >= 0.7540)")
    print(f"  p50 Latency     : {result.p50_ms} ms")
    print(f"  p95 Latency     : {result.p95_ms} ms")
    print(f"  Throughput      : {result.throughput_qps} QPS")
    print(f"  RAM Usage       : {result.peak_ram_mb} MB")
    print(f"  Monthly Cost    : ${result.monthly_cost_1k_req_usd:,} / mo (for 1,000 req/s)")
    print("=" * 65)
    print("[✓] Row 1 appended to optimization/results.csv")


if __name__ == "__main__":
    main()
