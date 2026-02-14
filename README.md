# IMAGE-PROCESSING-HW-OCR

OCR + local LLM pipeline for **document type detection** and **structured field extraction** from images.

Supported document types:
- `email`
- `invoice`
- `news` (article/page)
- `receipt`

## What this repo produces

For each input image:
- OCR text (Tesseract)
- Predicted document type (LLM or rule-based fallback)
- Structured JSON output
- Optional annotated image with OCR boxes

Output folder (default: `results/`):
- `results/json/*.json`
- `results/annotated_images/*_boxes.jpg` (optional)
- `results/metrics/predictions.csv`, `summary.txt`, `confusion_matrix.png` (batch mode)

## Method overview

1. **OCR**: Tesseract OCR on lightly preprocessed image (adaptive threshold).
2. **Classification**: Local LLM via **Ollama** (recommended model: `phi3`) returns strict JSON:
   `{ "document_type": "...", "confidence": 0.0 }`
   If LLM is disabled/unavailable, the system falls back to simple keyword rules.
3. **Field extraction**: LLM returns strict JSON per document type with a small schema
   (invoice number, total, email subject, etc.). If JSON parsing fails, regex fallbacks are used.

## Setup

### 1) Python environment

```bash
pip install -r requirements.txt
```

### 2) Install Tesseract OCR (system dependency)

- **Windows**: install Tesseract and ensure `tesseract.exe` is in PATH.
- **macOS (Homebrew)**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`

Optional language packs:
- For Lithuanian: `tesseract-ocr-lit` (Linux) and then run with `--lang eng+lit`.

### 3) Local LLM via Ollama (recommended)

Install Ollama and pull a small model:

```bash
ollama pull phi3
```

Make sure Ollama is running (default server: `http://localhost:11434`).

If you **cannot** use an LLM, run with `--no-llm` (rule-based baseline).

## Run: single image

From the repository root:

```bash
python main.py dataset/email/example.jpg
```

Options:
- `--model phi3` (default)
- `--lang eng` or `--lang eng+lit`
- `--annotate` to save `*_boxes.jpg`
- `--no-llm` to disable local LLM and use rules only

Example:

```bash
python main.py dataset/invoice/inv1.jpg --model phi3 --annotate
```

## Run: batch evaluation

The dataset is expected in subfolders:

```
dataset/
  email/
  invoice/
  news/
  receipts/
```

Run:

```bash
python main.py --batch dataset --annotate
```

This generates:
- `results/metrics/predictions.csv`
- `results/metrics/summary.txt`
- `results/metrics/confusion_matrix.png`

## JSON output format

Example:

```json
{
  "document_type": "invoice",
  "fields": {
    "invoice_number": "INV-001",
    "date": "2024-01-12",
    "seller": "UAB Example",
    "buyer": "John Doe",
    "total_amount": "125.50 EUR",
    "vat_amount": null,
    "currency": "EUR"
  },
  "meta": {
    "source_image": "dataset/invoice/inv1.jpg",
    "classification_confidence": 0.82,
    "classification_method": "llm",
    "ocr_engine": "tesseract"
  }
}
```

## Known limitations

- OCR quality depends on scan quality (blur, rotation, shadows).
- Receipts/invoices can vary heavily across vendors; field extraction is best-effort.
- If the local LLM is not running, the pipeline falls back to simple rules/regex.

## Repo structure

```
.
├── main.py
├── src/
│   ├── ocr.py
│   ├── llm.py
│   ├── classifier.py
│   ├── extractor.py
│   ├── pipeline.py
│   ├── eval.py
│   └── utils.py
├── dataset/               # input samples (not required to commit full dataset)
├── results/               # generated outputs
├── requirements.txt
└── README.md
```
