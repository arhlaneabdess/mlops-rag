from legal_rag.batch import BatchReindexer
from legal_rag.config import settings


def main():
    amendment_path = settings.DATA_DIR / "amendments" / "sample_amendment.json"
    print(f"[*] Starting Batch Re-indexing Pipeline for: {amendment_path}...")

    reindexer = BatchReindexer()
    initial_count = reindexer.retriever.count()
    print(f"[*] Pre-reindex collection count: {initial_count} chunks")

    results = reindexer.reindex_amendments(amendment_path)

    print("\n" + "=" * 60)
    print("           BATCH RE-INDEXING RUN REPORT")
    print("=" * 60)
    print(f"  Articles Processed : {results['articles_processed']}")
    print(f"  New Chunks Created : {results['chunks_indexed']}")
    print(f"  Wall-Clock Time    : {results['wall_clock_sec']}s")
    print(f"  Peak Added Memory  : {results['peak_memory_mb']} MB")
    print(f"  Batch Throughput   : {results['throughput_chunks_per_sec']} chunks/sec")
    print(f"  Estimated Cost/1M  : ${results['cost_per_million_predictions_usd']} USD")
    print(f"  New Database Total : {results['total_collection_count']} chunks")
    print("=" * 60)

    # Acceptance test: verify newly indexed amendment is immediately searchable
    print("\n[*] Validating immediate searchability of the new amendment...")
    hits = reindexer.retriever.search("ما هو حكم العقود الإلكترونية والتوقيع الرقمي؟", top_k=1)
    if hits and hits[0]["metadata"]["article_number"] == 1150:
        print(f"[✓] VERIFIED: Retrieved {hits[0]['metadata']['citation']}!")
        print(f"    Excerpt: {hits[0]['document'][:100]}...")
    else:
        print("[!] Verification failed to retrieve the new amendment.")


if __name__ == "__main__":
    main()
