import os
from datetime import datetime, timezone

import numpy as np
import psycopg2

from legal_rag.retriever import LegalRetriever
from legal_rag.text_normalizer import normalize_arabic
from monitoring.simulate_drift import get_drift_scenarios

# PostgreSQL connection string to existing container
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_USER = os.getenv("POSTGRES_USER", "mlflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mlflowpassword")
DB_NAME = os.getenv("POSTGRES_DB", "mlflowdb")

# Threshold
COSINE_DRIFT_THRESHOLD = 0.45


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )


def initialize_drift_table():
    """Creates the drift_metrics table in PostgreSQL (Handbook pg. 51)."""
    ddl = """
    CREATE TABLE IF NOT EXISTS drift_metrics (
        id SERIAL PRIMARY KEY,
        computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        dataset TEXT NOT NULL,
        feature TEXT NOT NULL,
        metric TEXT NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        threshold DOUBLE PRECISION NOT NULL,
        drifted BOOLEAN NOT NULL,
        model_version TEXT
    );
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(ddl)
        conn.commit()
    print("[✓] PostgreSQL 'drift_metrics' table initialized.")


class EmbeddingDriftDetector:
    def __init__(self, retriever: LegalRetriever = None):
        self.retriever = retriever or LegalRetriever()
        # Compute baseline reference centroid from in-domain Civil Code queries
        scenarios = get_drift_scenarios()
        ref_queries = [normalize_arabic(q) for q in scenarios["control_civil_code"]]
        ref_embeddings = np.array(self.retriever.embed_texts(ref_queries))
        self.reference_centroid = np.mean(ref_embeddings, axis=0, keepdims=True)
        # Normalize centroid
        self.reference_centroid = self.reference_centroid / np.linalg.norm(self.reference_centroid)

    def compute_cosine_drift(self, queries: list[str]) -> float:
        """Computes mean cosine distance from the in-domain legal centroid."""
        if not queries:
            return 0.0
        cleaned = [normalize_arabic(q) for q in queries]
        embeddings = np.array(self.retriever.embed_texts(cleaned))
        # Normalize vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

        # Cosine similarity against reference centroid
        sims = np.dot(embeddings, self.reference_centroid.T).flatten()
        # Cosine distance = 1.0 - similarity
        distances = 1.0 - sims
        return float(np.mean(distances))

    def evaluate_and_record(self, dataset_name: str, queries: list[str]) -> dict:
        drift_distance = self.compute_cosine_drift(queries)
        is_drifted = drift_distance > COSINE_DRIFT_THRESHOLD

        record = {
            "computed_at": datetime.now(timezone.utc),
            "dataset": dataset_name,
            "feature": "query_embeddings",
            "metric": "cosine_distance",
            "value": round(drift_distance, 4),
            "threshold": COSINE_DRIFT_THRESHOLD,
            "drifted": is_drifted,
            "model_version": "v0.5.0",
        }

        # Insert record into PostgreSQL
        query = """
        INSERT INTO drift_metrics (computed_at, dataset, feature, metric, value, threshold, drifted, model_version)
        VALUES (%(computed_at)s, %(dataset)s, %(feature)s, %(metric)s, %(value)s, %(threshold)s, %(drifted)s, %(model_version)s)
        """
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(query, record)
            conn.commit()

        return record


def main():
    print("[*] Starting Legal Embedding Drift Detection Engine...")
    initialize_drift_table()

    detector = EmbeddingDriftDetector()
    scenarios = get_drift_scenarios()

    results = []
    print("\n" + "=" * 70)
    print("         EMBEDDING DRIFT EVALUATION & POSTGRESQL INSERTION")
    print("=" * 70)

    for name, queries in scenarios.items():
        res = detector.evaluate_and_record(name, queries)
        results.append(res)
        status = "DRIFT DETECTED" if res["drifted"] else "STABLE (NO DRIFT)"
        print(f"  Dataset: {name:<25} | Metric: {res['value']:.4f} | Status: {status}")

    print("=" * 70)

    # Acceptance Check (Handbook pg. 44): Control set MUST NOT fire
    control_res = next(r for r in results if r["dataset"] == "control_civil_code")
    assert not control_res["drifted"], "ACCEPTANCE FAILURE: Detector fired on undrifted control!"
    print("\n[✓] ACCEPTANCE GATE PASSED: Zero false alarms on in-domain control queries.")


if __name__ == "__main__":
    main()
