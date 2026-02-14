from __future__ import annotations

from typing import Dict, Any, List
import cv2
import easyocr

# Reader sukūrimas užtrunka, todėl laikom globaliai
# Galima įdėti 'lt' jei reikia: ['en', 'lt']
_READER = None

def _get_reader(langs: List[str]):
    global _READER
    # Jei nori kelių kalbų (pvz en+lt), geriau inicijuoti su abiem iš karto.
    # Paprastumo dėlei: jei jau sukurtas, pernaudojam.
    if _READER is None:
        _READER = easyocr.Reader(langs, gpu=False)  # gpu=False kad veiktų visur
    return _READER

def ocr_image(image_path: str, lang: str = "en") -> Dict[str, Any]:
    """
    EasyOCR OCR:
    - text: sujungtas tekstas
    - boxes: word/line box'ai (x,y,w,h,text,conf)
    lang: 'en' arba 'en+lt' (mes suparsinsim)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # lang formatas: "eng+lit" buvo tesseract'ui.
    # EasyOCR naudoja trumpinius, pvz: 'en', 'lt'.
    # Čia paprasta logika: jei vartotojas duoda "en+lt" -> ['en','lt']
    if "+" in lang:
        langs = [x.strip() for x in lang.split("+") if x.strip()]
    else:
        langs = [lang.strip()]

    reader = _get_reader(langs)

    # detail=1 grąžina dėžutes ir confidence
    # paragraph=False kad būtų daugiau kontrolės
    results = reader.readtext(image_path, detail=1, paragraph=False)

    lines = []
    boxes = []
    for (bbox, text, conf) in results:
        if not text or not str(text).strip():
            continue
        lines.append(str(text).strip())

        # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        boxes.append({
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "text": str(text).strip(),
            "conf": float(conf),
        })

    full_text = "\n".join(lines)

    return {"engine": "easyocr", "text": full_text, "boxes": boxes}