from __future__ import annotations

import json
import os
from typing import List, Dict, Any
import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def ensure_dirs(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "json"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "annotated_images"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "metrics"), exist_ok=True)

def list_images(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMAGE_EXTS:
                out.append(os.path.join(root, fn))
    return sorted(out)

def save_json(data: Dict[str, Any], folder: str, filename: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def save_annotated_image(image_path: str, boxes: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        return
    for b in boxes:
        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
