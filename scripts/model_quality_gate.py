import json
import sys

from legal_rag.config import settings

FAITHFULNESS_THRESHOLD = 0.75


def run_quality_gate() -> bool:
    print(
        f"[*] Checking RAG Model Quality Gate (Threshold: Faithfulness >= {FAITHFULNESS_THRESHOLD})..."
    )

    artifact_dir = settings.DATA_DIR / "eval_artifacts"
    if not artifact_dir.exists():
        print("[!] No evaluation artifacts directory found. Run scripts.run_rag_experiments first.")
        return False

    eval_files = list(artifact_dir.glob("*_eval.json"))
    if not eval_files:
        print("[!] No evaluation run artifacts found in data/eval_artifacts/.")
        return False

    # Find the best faithfulness score among logged runs
    best_score = -1.0
    best_file = None

    for fpath in eval_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                score = data.get("metrics", {}).get("faithfulness", 0.0)
                if score > best_score:
                    best_score = score
                    best_file = fpath.name
        except Exception as e:
            print(f"[!] Error reading {fpath.name}: {e}")

    print(f"[*] Best Evaluated Run: {best_file}")
    print(f"[*] Measured Faithfulness: {best_score:.4f} (Required >= {FAITHFULNESS_THRESHOLD})")

    if best_score >= FAITHFULNESS_THRESHOLD:
        print(f"[✓] QUALITY GATE PASSED: Faithfulness {best_score:.4f} >= {FAITHFULNESS_THRESHOLD}")
        return True
    else:
        print(
            f"[✘] QUALITY GATE FAILED: Faithfulness {best_score:.4f} is below the {FAITHFULNESS_THRESHOLD} baseline!"
        )
        return False


def main():
    passed = run_quality_gate()
    if not passed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
