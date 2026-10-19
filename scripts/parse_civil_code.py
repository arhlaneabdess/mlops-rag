import json

from legal_rag.config import settings
from legal_rag.parser import CivilCodeParser


def main():
    print(f"[*] Starting extraction from {settings.RAW_PDF_PATH}...")
    parser = CivilCodeParser(str(settings.RAW_PDF_PATH))
    records = parser.parse()

    settings.PROCESSED_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.PROCESSED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, ensure_ascii=False, indent=2)

    print(f"[✓] Extracted {len(records)} articles. Saved to {settings.PROCESSED_JSON_PATH}")


if __name__ == "__main__":
    main()
