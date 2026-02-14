import os
from .ocr import ocr_image
from .classifier import classify_document
from .extractor import extract_fields
from .utils import save_json, save_annotated_image, ensure_dirs

def process_image(
    image_path: str,
    outdir: str = "results",
    model: str = "phi3",
    use_llm: bool = True,
    tesseract_lang: str = "eng",
    annotate: bool = False,
):
    """Process a single image: OCR -> classify -> extract -> save outputs."""
    ensure_dirs(outdir)

    ocr = ocr_image(image_path, lang=tesseract_lang)
    text = ocr["text"]

    doc_type, conf, method = classify_document(text, model=model, use_llm=use_llm)

    data = extract_fields(text, doc_type, model=model, use_llm=use_llm)
    data["ocr_text"] = text
    # enrich
    data.setdefault("document_type", doc_type)
    data.setdefault("meta", {})
    data["meta"].update({
        "source_image": image_path,
        "classification_confidence": conf,
        "classification_method": method,
        "ocr_engine": ocr.get("engine", "tesseract"),
    })

    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = save_json(data, os.path.join(outdir, "json"), base + ".json")

    if annotate and ocr.get("boxes"):
        save_annotated_image(
            image_path=image_path,
            boxes=ocr["boxes"],
            out_path=os.path.join(outdir, "annotated_images", base + "_boxes.jpg"),
        )

    return data
