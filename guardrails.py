"""Guardrails: input validation, output safety, PII detection, hallucination flagging."""

import re


# --- Input Guardrails ---

BLOCKED_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all prior",
    r"system prompt",
    r"you are now",
    r"pretend you are",
    r"jailbreak",
]

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(\+1[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
}


def validate_input(query: str) -> dict:
    """Check user input for prompt injection and PII.

    Returns: {"safe": bool, "issues": list[str]}
    """
    issues = []
    lower = query.lower()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lower):
            issues.append(f"prompt_injection_detected: {pattern}")

    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, query):
            issues.append(f"pii_detected: {pii_type}")

    if len(query) > 5000:
        issues.append("query_too_long")

    if len(query.strip()) < 3:
        issues.append("query_too_short")

    return {"safe": len(issues) == 0, "issues": issues}


# --- Output Guardrails ---

def validate_output(answer: str, sources: list[dict]) -> dict:
    """Check LLM output for safety issues.

    Returns: {"safe": bool, "issues": list[str], "warnings": list[str]}
    """
    issues = []
    warnings = []

    # Check for PII leakage in output
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, answer):
            issues.append(f"pii_in_output: {pii_type}")

    # Check if answer references sources
    if sources and "[Source" not in answer and len(answer) > 100:
        warnings.append("no_source_citations")

    # Check for refusal patterns that might indicate the model went off-rails
    refusal_patterns = [
        r"as an ai",
        r"i cannot",
        r"i'm not able to",
        r"i don't have access",
    ]
    lower = answer.lower()
    for p in refusal_patterns:
        if re.search(p, lower):
            warnings.append(f"possible_refusal: {p}")
            break

    # Empty or suspiciously short answer
    if len(answer.strip()) < 10:
        issues.append("answer_too_short")

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }


def redact_pii(text: str) -> str:
    """Redact detected PII from text."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
    return text
