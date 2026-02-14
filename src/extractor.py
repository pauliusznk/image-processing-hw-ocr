from __future__ import annotations

import re
from typing import Dict, Any
from .llm import ollama_json

# ---- Simple regex helpers (fallbacks) ----

def _email_fallback(text: str) -> Dict[str, Any]:
    t = text or ""

    def grab(pat):
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    return {
        "document_type": "email",
        "fields": {
            "from": grab(r"^from:\s*(.+)$"),
            "to": grab(r"^to:\s*(.+)$"),
            "cc": grab(r"^cc:\s*(.+)$"),
            "subject": grab(r"^subject:\s*(.+)$"),
            "date": grab(r"^date:\s*(.+)$"),
        },
    }


def _invoice_fallback(text: str) -> Dict[str, Any]:
    t = text or ""

    def grab(pat):
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    return {
        "document_type": "invoice",
        "fields": {
            "invoice_number": grab(r"invoice\s*(?:no\.|number|#)\s*[:\-]?\s*([A-Z0-9\-]+)") or grab(r"(INV[\-\s]?[0-9A-Z]+)"),
            "date": grab(r"date\s*[:\-]?\s*([0-9]{4}[\-/\.][0-9]{2}[\-/\.][0-9]{2})") or grab(r"([0-9]{2}[\./-][0-9]{2}[\./-][0-9]{4})"),
            "seller": grab(r"^(?:seller|from)\s*[:\-]?\s*(.+)$"),
            "buyer": grab(r"^(?:bill\s*to|buyer|to)\s*[:\-]?\s*(.+)$"),
            "total_amount": grab(r"total\s*[:\-]?\s*([0-9]+[\.,][0-9]{2}\s*(?:eur|usd|gbp)?)"),
            "vat_amount": grab(r"vat\s*[:\-]?\s*([0-9]+[\.,][0-9]{2})"),
            "currency": grab(r"\b(EUR|USD|GBP)\b"),
        },
    }


def _receipt_fallback(text: str) -> Dict[str, Any]:
    t = text or ""

    def grab(pat):
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    store_guess = lines[0] if lines else None

    return {
        "document_type": "receipt",
        "fields": {
            "store": store_guess,
            "date": grab(r"([0-9]{4}[\-/\.][0-9]{2}[\-/\.][0-9]{2})") or grab(r"([0-9]{2}[\./-][0-9]{2}[\./-][0-9]{4})"),
            "total": grab(r"total\s*[:\-]?\s*([0-9]+[\.,][0-9]{2}\s*(?:eur|usd|gbp)?)"),
            "currency": grab(r"\b(EUR|USD|GBP)\b"),
            "payment_method": grab(r"\b(cash|card|visa|mastercard)\b"),
        },
    }


def _news_fallback(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    title = lines[0] if lines else None

    author = None
    for ln in lines[:10]:
        m = re.search(r"^by\s+(.+)$", ln, flags=re.IGNORECASE)
        if m:
            author = m.group(1).strip()
            break

    return {
        "document_type": "news",
        "fields": {
            "title": title,
            "author": author,
            "content": t[:8000] if t else None,  # kad nebūtų milžiniškas JSON
        },
    }


# ---- Dynamic LLM extraction ----

def extract_fields(text: str, doc_type: str, model: str = "phi3", use_llm: bool = True) -> Dict[str, Any]:
    """
    Extract structured fields.
    - No predefined schema: LLM returns dynamic `fields`.
    - Fallback regex extraction if LLM is disabled/unavailable or JSON parsing fails.
    """
    doc_type = (doc_type or "").strip().lower()

    if not use_llm:
        return _fallback(text, doc_type)

    prompt = f"""You extract structured information from OCR text.

Document type: {doc_type}

Return STRICT JSON only. No explanations. No code fences.

Output JSON format:
{{
  "document_type": "{doc_type}",
  "fields": {{
    "key1": "value1",
    "key2": null
  }}
}}

Rules:
- `fields` keys must be concise and snake_case (e.g., invoice_number, total_amount, from, subject, title, content).
- Extract only information present in the OCR text. Do not invent data.
- If a value is unknown/missing, use null.
- Prefer short values. For long text (news/articles), put it into `content` and optionally add `summary` (1-2 sentences).
- Keep `content` max ~8000 characters.

OCR text:
{text}
"""

    obj, _raw = ollama_json(prompt, model=model, temperature=0.0)
    if isinstance(obj, dict) and str(obj.get("document_type", "")).strip().lower() == doc_type:
        if isinstance(obj.get("fields"), dict) and obj["fields"]:
            # Optional: hard-limit content length if model ignored it
            content = obj["fields"].get("content")
            if isinstance(content, str) and len(content) > 8000:
                obj["fields"]["content"] = content[:8000]
            return obj

    return _fallback(text, doc_type)


def _fallback(text: str, doc_type: str) -> Dict[str, Any]:
    if doc_type == "email":
        return _email_fallback(text)
    if doc_type == "invoice":
        return _invoice_fallback(text)
    if doc_type == "receipt":
        return _receipt_fallback(text)
    return _news_fallback(text)