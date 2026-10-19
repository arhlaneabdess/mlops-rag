from legal_rag.tracing import TracedLegalRAG


def main():
    print("[*] Initializing Traced Legal RAG Pipeline (Langfuse)...")
    traced_pipeline = TracedLegalRAG()

    question = "ما هو حكم العقد وهل يجوز تعديله بإرادة أحد الطرفين فقط؟"
    print(f"\n[*] Executing user query: '{question}'...")

    response = traced_pipeline.query(
        question=question,
        user_id="lawyer_cairo_102",
        session_id="contract_dispute_case_54",
        correlation_id="corr-m5-phase3-001",
    )

    print("\n" + "=" * 70)
    print("                    LANGFUSE TRACE SUMMARY")
    print("=" * 70)
    print("  Trace Name     : arabic_legal_rag_query")
    print("  User ID        : lawyer_cairo_102")
    print(f"  Correlation ID : {response['correlation_id']}")
    print("  Spans Executed : retrieve -> rerank -> generate")
    print(f"  Prompt Tokens  : {response.get('prompt_tokens', 0)} tokens")
    print(f"  Output Tokens  : {response.get('completion_tokens', 0)} tokens")
    print(f"  Citations      : {', '.join(response['sources'])}")
    print("=" * 70)
    print(f"\n[✓] Synthesized Answer:\n{response['answer']}")
    print("\n[✓] Trace successfully dispatched to Langfuse!")


if __name__ == "__main__":
    main()
