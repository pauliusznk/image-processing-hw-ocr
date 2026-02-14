#!/usr/bin/env python3
"""Run OCR + LLM document classification + field extraction.

Usage:
  python main.py path/to/image.jpg
  python main.py --batch dataset --limit 50
"""
import argparse
import os
import sys

# Allow importing modules from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.pipeline import process_image
from src.eval import run_batch

def parse_args():
    p = argparse.ArgumentParser(description="OCR + Local LLM document parser (email/invoice/news/receipt).")
    p.add_argument("image", nargs="?", help="Path to input image (.jpg/.png).")
    p.add_argument("--batch", type=str, default=None, help="Batch folder (e.g., dataset).")
    p.add_argument("--outdir", type=str, default="results", help="Output directory (default: results).")
    p.add_argument("--model", type=str, default="phi3", help="Ollama model name (default: phi3).")
    p.add_argument("--no-llm", action="store_true", help="Disable LLM; use rule-based fallback only.")
    p.add_argument("--limit", type=int, default=0, help="Limit number of images in batch (0 = no limit).")
    p.add_argument("--lang", type=str, default="en", help="Tesseract language(s), e.g. eng or eng+lit (default: eng).")
    p.add_argument("--annotate", action="store_true", help="Save annotated image with OCR boxes.")
    return p.parse_args()

def main():
    args = parse_args()

    if args.batch:
        if not os.path.isdir(args.batch):
            raise SystemExit(f"Batch folder not found: {args.batch}")
        run_batch(
            dataset_dir=args.batch,
            outdir=args.outdir,
            model=args.model,
            use_llm=not args.no_llm,
            limit=args.limit,
            tesseract_lang=args.lang,
            annotate=args.annotate,
        )
        return

    if not args.image:
        raise SystemExit("Provide an image path or use --batch. Example: python main.py dataset/email/1.jpg")

    if not os.path.exists(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    result = process_image(
        image_path=args.image,
        outdir=args.outdir,
        model=args.model,
        use_llm=not args.no_llm,
        tesseract_lang=args.lang,
        annotate=args.annotate,
    )
    print("\n=== RESULT JSON ===")
    print(result)

if __name__ == "__main__":
    main()
