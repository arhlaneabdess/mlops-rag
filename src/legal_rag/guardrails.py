import re

from prometheus_client import Counter

# Guardrail Telemetry Metric (Handbook pg. 55)
GUARDRAIL_DECISIONS = Counter(
    "guardrail_decisions_total",
    "Telemetry for input and output guardrail evaluations",
    ["guardrail", "decision"],
)

# 1. Egyptian National ID: 14 digits starting with 2 or 3
EGYPTIAN_NATIONAL_ID_REGEX = re.compile(
    r"\b[23]\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{7}\b"
)

# 2. Egyptian Mobile Phone Numbers: 010, 011, 012, 015 followed by 8 digits
EGYPTIAN_PHONE_REGEX = re.compile(r"(?:\+?20|0)?1[0125]\d{8}\b")

# 3. Adversarial Prompt Injection Patterns (Arabic & English)
INJECTION_PATTERNS = [
    re.compile(r"(?:ignore|disregard|forget)\s+.*(?:previous|all)?\s*instructions", re.IGNORECASE),
    re.compile(r"تجاهل\s+.*(?:التعليمات|الأوامر|الاوامر)\s*(?:السابقة)?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"تصرف\s+كأنك\s+بدون\s+قيود", re.IGNORECASE),
    re.compile(r"reveal\s+.*system\s+prompt", re.IGNORECASE),
    re.compile(r"ما\s+هي\s+.*(?:تعليماتك|أوامرك)\s*(?:السرية)?", re.IGNORECASE),
]


class LegalGuardrails:
    """
    Production Guardrail Suite for Arabic Legal RAG Applications.
    Handles Egyptian PII redaction, prompt injection defense, and citation verification.
    """

    @staticmethod
    def inspect_and_sanitize_input(query: str) -> tuple[bool, str, str]:
        # Check 1: Prompt Injection
        for pattern in INJECTION_PATTERNS:
            if pattern.search(query):
                GUARDRAIL_DECISIONS.labels(guardrail="prompt_injection", decision="blocked").inc()
                return False, query, "Security Violation: Prompt injection attempt detected."

        GUARDRAIL_DECISIONS.labels(guardrail="prompt_injection", decision="passed").inc()

        # Check 2: Egyptian PII Redaction
        sanitized = query
        pii_found = False

        if EGYPTIAN_NATIONAL_ID_REGEX.search(sanitized):
            sanitized = EGYPTIAN_NATIONAL_ID_REGEX.sub("[REDACTED_EGYPTIAN_NATIONAL_ID]", sanitized)
            pii_found = True

        if EGYPTIAN_PHONE_REGEX.search(sanitized):
            sanitized = EGYPTIAN_PHONE_REGEX.sub("[REDACTED_PHONE_NUMBER]", sanitized)
            pii_found = True

        if pii_found:
            GUARDRAIL_DECISIONS.labels(guardrail="pii_masking", decision="redacted").inc()
        else:
            GUARDRAIL_DECISIONS.labels(guardrail="pii_masking", decision="passed").inc()

        return True, sanitized, ""

    @staticmethod
    def verify_output_groundedness(answer: str, retrieved_sources: list[str]) -> tuple[bool, str]:
        cited_articles = re.findall(r"(?:المادة|مادة)\s*(\d+)", answer)
        retrieved_article_nums = [
            re.search(r"Article\s*(\d+)", s).group(1)
            for s in retrieved_sources
            if re.search(r"Article\s*(\d+)", s)
        ]

        hallucinations = [art for art in cited_articles if art not in retrieved_article_nums]

        if hallucinations:
            GUARDRAIL_DECISIONS.labels(guardrail="citation_hallucination", decision="blocked").inc()
            return (
                False,
                f"Hallucination Detected: Article(s) {', '.join(hallucinations)} cited but not in context.",
            )

        GUARDRAIL_DECISIONS.labels(guardrail="citation_hallucination", decision="passed").inc()
        return True, ""
