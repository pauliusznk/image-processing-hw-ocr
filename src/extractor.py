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
            "invoice_number": grab(r"(?:invoice\s*(?:no\.|number|#)\s*[:\-]?\s*([A-Z0-9\-]+))")
                              or grab(r"(?:\bno\b\s*[:\-]?\s*([A-Z0-9\-]{4,}))")
                              or grab(r"\b(INV[\-\s]?[0-9A-Z]+)\b"),
            "date": grab(r"(?:date(?:\s+of\s+issue)?\s*[:\-]?\s*([0-9]{4}[\-/\.][0-9]{2}[\-/\.][0-9]{2}))")
                    or grab(r"(?:date(?:\s+of\s+issue)?\s*[:\-]?\s*([0-9]{2}[\./-][0-9]{2}[\./-][0-9]{4}))")
                    or grab(r"([0-9]{4}[\-/\.][0-9]{2}[\-/\.][0-9]{2})")
                    or grab(r"([0-9]{2}[\./-][0-9]{2}[\./-][0-9]{4})"),
            "seller": grab(r"^(?:seller|from)\s*[:\-]?\s*(.+)$"),
            "buyer": grab(r"^(?:client|bill\s*to|buyer|to)\s*[:\-]?\s*(.+)$"),
            "total_amount": grab(r"\btotal\b\s*[:\-]?\s*([$€£]?\s*[0-9]+[\.,][0-9]{2}\s*(?:eur|usd|gbp)?)"),
            "vat_amount": grab(r"\bvat\b\s*[:\-]?\s*([$€£]?\s*[0-9]+[\.,][0-9]{2})"),
            "currency": grab(r"\b(EUR|USD|GBP)\b") or grab(r"([$€£])"),
        },
    }


def _receipts_fallback(text: str) -> Dict[str, Any]:
    t = text or ""

    def grab(pat):
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    store_guess = lines[0] if lines else None

    return {
        "document_type": "receipts",
        "fields": {
            "store": store_guess,
            "date": grab(r"([0-9]{4}[\-/\.][0-9]{2}[\-/\.][0-9]{2})")
                    or grab(r"([0-9]{2}[\./-][0-9]{2}[\./-][0-9]{4})"),
            "total": grab(r"\btotal\b\s*[:\-]?\s*([$€£]?\s*[0-9]+[\.,][0-9]{2}\s*(?:eur|usd|gbp)?)"),
            "currency": grab(r"\b(EUR|USD|GBP)\b") or grab(r"([$€£])"),
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
            "content": t[:8000] if t else None,
        },
    }


# ---- Text focusing (VERY important for small local models) ----

def _focus_text(text: str, doc_type: str) -> str:
    """
    Small local models (phi3, etc.) work better if we feed only relevant parts.
    """
    t = (text or "").strip()
    if not t:
        return ""

    lines = [ln.rstrip() for ln in t.splitlines()]

    # Common: top section contains header info
    top = "\n".join([ln for ln in lines[:70] if ln.strip()])

    def find_block_start(keywords):
        for i, ln in enumerate(lines):
            low = ln.lower()
            if any(k in low for k in keywords):
                return i
        return None

    if doc_type == "invoice":
        # bottom: summary/total blocks
        s = find_block_start(["summary", "total", "gross worth", "vat"])
        bottom = "\n".join([ln for ln in (lines[s:s+120] if s is not None else lines[-120:]) if ln.strip()])
        return top + "\n\n----\n\n" + bottom

    if doc_type == "receipts":
        # receiptss often have totals near bottom
        s = find_block_start(["total", "sum", "amount", "paid", "cash", "card"])
        bottom = "\n".join([ln for ln in (lines[s:s+100] if s is not None else lines[-100:]) if ln.strip()])
        return top + "\n\n----\n\n" + bottom

    if doc_type == "email":
        # email headers near top
        return "\n".join([ln for ln in lines[:120] if ln.strip()])

    # news: keep start + small body (avoid huge)
    body = "\n".join([ln for ln in lines[:250] if ln.strip()])
    return body[:9000]


# ---- Dynamic LLM extraction ----

def extract_fields(text: str, doc_type: str, model: str = "phi3", use_llm: bool = True) -> Dict[str, Any]:
    """
    Extract structured fields.
    - LLM returns dynamic `fields`.
    - Fallback regex extraction if LLM is disabled/unavailable or JSON parsing fails.
    """
    doc_type = (doc_type or "").strip().lower()

    if not use_llm:
        return _fallback(text, doc_type)

    focused = _focus_text(text, doc_type)

    # Strong, doc-type-specific guidance helps phi3 a lot
    hints = ""
    if doc_type == "invoice":
        hints = """
Invoice extraction hints:
- invoice_number: prefer patterns like "no: 123456" or "invoice no: XYZ".
- date: prefer "date of issue:" or a nearby "date:".
- total_amount: prefer line starting with "Total" near the end (Summary).
- currency: infer from symbol ($, €, £) or currency code (USD/EUR/GBP) if present.
- seller/buyer: prefer lines after "Seller:" and "Client:" (or "Bill to:").
"""
    elif doc_type == "receipts":
        hints = """
Receipt extraction hints:
- store: usually the first non-empty line.
- total: prefer the last "Total" line; include currency if present.
- date: may appear as DD/MM/YYYY or YYYY-MM-DD.
"""
    elif doc_type == "email":
        hints = """
Email extraction hints:
- from/to/cc/subject/date are typically in header lines like "From: ...".
"""
    else:
        hints = """
News extraction hints:
- title: first line (or the largest heading if present).
- author: line starting with "By ...".
- content: include up to 8000 chars.
- summary: optional 1-2 sentences.
"""

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

Rules (VERY IMPORTANT):
- Extract ONLY information explicitly present in the OCR text. Do NOT invent data.
- If a value is missing/unknown, use null.
- Keys must be concise snake_case (invoice_number, total_amount, from, subject, title, content, summary, etc.).
- Prefer short values (no long paragraphs) except `content` for news.
- Keep `content` max 8000 characters.
- For money, keep the numeric amount and include currency if possible (e.g., "504.69 USD" or "$ 504.69").

{hints}

OCR text:
{focused}
"""

    obj, _raw = ollama_json(prompt, model=model, temperature=0.0)
    if isinstance(obj, dict) and str(obj.get("document_type", "")).strip().lower() == doc_type:
        if isinstance(obj.get("fields"), dict) and obj["fields"]:
            # Hard limits
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
    if doc_type == "receipts":
        return _receipts_fallback(text)
    return _news_fallback(text)