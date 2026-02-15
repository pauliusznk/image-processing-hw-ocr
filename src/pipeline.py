import os
import time
from .ocr import ocr_image
from .classifier import classify_document
from .extractor import extract_fields
from .utils import save_json, save_annotated_image, ensure_dirs, get_timestamp_prefix

def process_image(
    image_path: str,
    outdir: str = "results",
    model: str = "phi3",
    use_llm: bool = True,
    tesseract_lang: str = "eng",
    annotate: bool = False,
):
    """Process a single image: OCR -> classify -> extract -> save outputs."""
    start_time = time.time()
    ensure_dirs(outdir)

    ocr_start = time.time()
    ocr = ocr_image(image_path, lang=tesseract_lang)
    ocr_time = time.time() - ocr_start
    text = ocr["text"]

    classify_start = time.time()
    doc_type, conf, method = classify_document(text, model=model, use_llm=use_llm)
    classify_time = time.time() - classify_start

    extract_start = time.time()
    data = extract_fields(text, doc_type, model=model, use_llm=use_llm)
    extract_time = time.time() - extract_start

    data["ocr_text"] = text
    # enrich
    data.setdefault("document_type", doc_type)
    data.setdefault("meta", {})

    total_time = time.time() - start_time

    data["meta"].update({
        "source_image": image_path,
        "classification_confidence": conf,
        "classification_method": method,
        "ocr_engine": ocr.get("engine", "tesseract"),
        "processing_time_seconds": round(total_time, 3),
        "ocr_time_seconds": round(ocr_time, 3),
        "classification_time_seconds": round(classify_time, 3),
        "extraction_time_seconds": round(extract_time, 3),
    })

    base = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = get_timestamp_prefix()
    json_filename = f"{timestamp}-{base}.json"
    json_path = save_json(data, os.path.join(outdir, "json"), json_filename)

    if annotate and ocr.get("boxes"):
        annotated_filename = f"{timestamp}-{base}_boxes.jpg"
        save_annotated_image(
            image_path=image_path,
            boxes=ocr["boxes"],
            out_path=os.path.join(outdir, "annotated_images", annotated_filename),
        )

    print(f"⏱️  Processing time: {total_time:.3f}s (OCR: {ocr_time:.3f}s, Classify: {classify_time:.3f}s, Extract: {extract_time:.3f}s)")

    return data
