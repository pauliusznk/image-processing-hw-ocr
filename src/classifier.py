from __future__ import annotations

from typing import Tuple
from .llm import ollama_json

LABELS = ["email", "invoice", "news", "receipts"]

def _rule_based(text: str) -> Tuple[str, float]:
    t = (text or "").lower()

    # Email-ish
    if ("from:" in t and "to:" in t) or ("subject:" in t and "from:" in t):
        return "email", 0.65

    # Receipt-ish (check BEFORE invoice to avoid confusion)
    if "receipt" in t:
        return "receipts", 0.70
    # Receipts are shorter, simpler, often have store names at top
    if ("total" in t or "subtotal" in t) and ("cash" in t or "card" in t or "payment" in t):
        return "receipts", 0.60
    # Receipt-specific patterns
    if ("thank you" in t or "come again" in t) and "total" in t:
        return "receipts", 0.65

    # Invoice-ish (more formal, has specific invoice markers)
    if "invoice no" in t or "invoice number" in t or "invoice #" in t:
        return "invoice", 0.75
    if "bill to" in t or ("seller" in t and "buyer" in t):
        return "invoice", 0.70
    # Only invoice if explicit invoice keyword present
    if "invoice" in t and ("date of issue" in t or "due date" in t or "buyer" in t):
        return "invoice", 0.65

    # Generic money document (lower confidence)
    if "total" in t and ("eur" in t or "usd" in t or "$" in t or "€" in t):
        return "receipts", 0.45  # default to receipt for generic money docs

    # News/article-ish
    if "by " in t and ("published" in t or "updated" in t):
        return "news", 0.55
    if len(t.split()) > 150:
        return "news", 0.45

    return "news", 0.35

def classify_document(text: str, model: str = "phi3", use_llm: bool = True) -> Tuple[str, float, str]:
    """Return (label, confidence, method)."""
    if not use_llm:
        label, conf = _rule_based(text)
        return label, conf, "rules"

    prompt = f"""You are a strict document classifier.

Task:
1) Classify the document into exactly ONE label from: {LABELS}
2) Return STRICT JSON only.

IMPORTANT - Key differences:
- **invoice**: Formal business document with seller/buyer info, invoice number, VAT breakdown, payment terms, "Bill To", "Date of Issue". Usually multi-party (company to company/client).
- **receipts**: Simple proof of purchase from store/restaurant. Has store name, items purchased, total, payment method (cash/card). Usually says "Receipt" or "Thank you". Single-party transaction.
- **email**: Has email headers (From:, To:, Subject:, Date:, CC:).
- **news**: Article or news page with title, author, published date, long text content.

JSON schema:
{{
  "document_type": "email|invoice|news|receipts",
  "confidence": 0.0
}}

Document text:
{text}
"""

    obj, raw = ollama_json(prompt, model=model, temperature=0.0)
    if isinstance(obj, dict):
        dt = str(obj.get("document_type", "")).strip().lower()
        if dt in LABELS:
            try:
                conf = float(obj.get("confidence", 0.6))
            except Exception:
                conf = 0.6
            conf = max(0.0, min(1.0, conf))
            return dt, conf, "llm"

    # fallback
    label, conf = _rule_based(text)
    return label, conf, "rules_fallback"
