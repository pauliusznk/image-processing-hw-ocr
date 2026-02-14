from __future__ import annotations

from typing import Tuple
from .llm import ollama_json

LABELS = ["email", "invoice", "news", "receipts"]

def _rule_based(text: str) -> Tuple[str, float]:
    t = (text or "").lower()

    # Email-ish
    if ("from:" in t and "to:" in t) or ("subject:" in t and "from:" in t):
        return "email", 0.65

    # Invoice-ish
    if "invoice" in t or "vat" in t or "invoice no" in t or "bill to" in t:
        return "invoice", 0.65

    # Receipt-ish
    if "total" in t and ("eur" in t or "usd" in t or "cash" in t or "card" in t):
        return "receipt", 0.55
    if "receipt" in t:
        return "receipt", 0.65

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

JSON schema:
{{
  "document_type": "email|invoice|news|receipt",
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
