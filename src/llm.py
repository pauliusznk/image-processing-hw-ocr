from __future__ import annotations

import json
import re
import requests
from typing import Any, Dict, Optional, Tuple

OLLAMA_URL = "http://localhost:11434/api/generate"

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from model output."""
    # common: model outputs code fences or extra commentary
    # find first '{' and last '}' and attempt parse progressively
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # try to fix trailing commas / minor issues
        blob2 = re.sub(r",\s*\}", "}", blob)
        blob2 = re.sub(r",\s*\]", "]", blob2)
        try:
            return json.loads(blob2)
        except json.JSONDecodeError:
            return None

def ollama_generate(prompt: str, model: str = "phi3", temperature: float = 0.0, timeout: int = 120) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.exceptions.RequestException:
        # Ollama not running / not installed / blocked
        return ""

def ollama_json(prompt: str, model: str = "phi3", temperature: float = 0.0) -> Tuple[Optional[Dict[str, Any]], str]:
    """Call Ollama and try to parse JSON. Returns (json_or_none, raw_text)."""
    raw = ollama_generate(prompt, model=model, temperature=temperature)
    return _extract_json(raw), raw
