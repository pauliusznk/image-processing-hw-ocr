from __future__ import annotations

import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from .pipeline import process_image
from .utils import list_images, ensure_dirs, get_timestamp_prefix
from .spinner import Spinner

LABELS = ["email", "invoice", "news", "receipts"]

def _true_label_from_path(path: str) -> str:
    # expects dataset/<label>/file.jpg
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p in LABELS:
            return p
    return "unknown"

def run_batch(
    dataset_dir: str,
    outdir: str = "results",
    model: str = "phi3",
    use_llm: bool = True,
    limit: int = 0,
    tesseract_lang: str = "eng",
    annotate: bool = False,
):
    batch_start_time = time.time()
    ensure_dirs(outdir)

    # Collect images per label
    label_images = {}
    for lab in LABELS:
        p = os.path.join(dataset_dir, lab)
        if os.path.isdir(p):
            label_images[lab] = list_images(p)

    # If no label folders found, fallback to flat directory
    if not label_images:
        images = list_images(dataset_dir)
    else:
        # Apply limit proportionally across labels
        images = []
        if limit and limit > 0:
            per_label = limit // len(label_images)
            remainder = limit % len(label_images)
            for idx, (lab, lab_imgs) in enumerate(label_images.items()):
                # Add extra image to first 'remainder' labels to distribute evenly
                take = per_label + (1 if idx < remainder else 0)
                images.extend(lab_imgs[:take])
                print(f"Taking {min(take, len(lab_imgs))}/{len(lab_imgs)} from {lab}")
        else:
            # No limit, take all
            for lab_imgs in label_images.values():
                images.extend(lab_imgs)

    print(f"\n🚀 Batch processing: {len(images)} images\n")
    rows = []
    for idx, img_path in enumerate(images, 1):
        # Progress spinner for each image
        spinner = Spinner(f"[{idx}/{len(images)}] Processing {os.path.basename(img_path)}")
        spinner.start()

        img_start = time.time()
        res = process_image(
            image_path=img_path,
            outdir=outdir,
            model=model,
            use_llm=use_llm,
            tesseract_lang=tesseract_lang,
            annotate=annotate,
            show_spinner=False,  # Disable inner spinner in batch mode
        )
        img_time = time.time() - img_start
        pred = res.get("document_type")
        true = _true_label_from_path(img_path)

        # Stop spinner with result
        spinner.stop(f"✓ [{idx}/{len(images)}] {os.path.basename(img_path)} → {pred} ({img_time:.2f}s)")

        rows.append({
            "image": img_path,
            "true_label": true,
            "pred_label": pred,
            "confidence": res.get("meta", {}).get("classification_confidence"),
            "method": res.get("meta", {}).get("classification_method"),
            "processing_time": res.get("meta", {}).get("processing_time_seconds", img_time),
        })

    # Generate metrics with spinner
    print()  # Add newline
    spinner = Spinner("📊 Generating metrics and confusion matrix")
    spinner.start()

    df = pd.DataFrame(rows)
    timestamp = get_timestamp_prefix()
    metrics_path = os.path.join(outdir, "metrics", f"{timestamp}-predictions.csv")
    df.to_csv(metrics_path, index=False)

    batch_total_time = time.time() - batch_start_time

    # accuracy (ignore unknown)
    df_known = df[df["true_label"].isin(LABELS)]
    acc = (df_known["true_label"] == df_known["pred_label"]).mean() if len(df_known) else 0.0

    # timing statistics
    avg_time = df["processing_time"].mean() if "processing_time" in df.columns else 0.0
    min_time = df["processing_time"].min() if "processing_time" in df.columns else 0.0
    max_time = df["processing_time"].max() if "processing_time" in df.columns else 0.0

    summary_path = os.path.join(outdir, "metrics", f"{timestamp}-summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Images: {len(df)}\n")
        f.write(f"Known-label images: {len(df_known)}\n")
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(f"\n=== Timing Statistics ===\n")
        f.write(f"Total batch time: {batch_total_time:.2f}s\n")
        f.write(f"Average per image: {avg_time:.3f}s\n")
        f.write(f"Min time: {min_time:.3f}s\n")
        f.write(f"Max time: {max_time:.3f}s\n")

    # confusion matrix plot
    if len(df_known):
        cm = pd.crosstab(df_known["true_label"], df_known["pred_label"], rownames=["true"], colnames=["pred"], dropna=False)
        for l in LABELS:
            if l not in cm.index:
                cm.loc[l] = 0
            if l not in cm.columns:
                cm[l] = 0
        cm = cm.loc[LABELS, LABELS]

        plt.figure()
        plt.imshow(cm.values)
        plt.xticks(range(len(LABELS)), LABELS, rotation=30)
        plt.yticks(range(len(LABELS)), LABELS)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                plt.text(j, i, str(cm.values[i, j]), ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (acc={acc:.3f})")
        plot_path = os.path.join(outdir, "metrics", f"{timestamp}-confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    spinner.stop(f"✓ Metrics generated (Accuracy: {acc:.3f})")

    print(f"\n=== Results ===")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {summary_path}")
    plot_path_check = os.path.join(outdir, "metrics", f"{timestamp}-confusion_matrix.png")
    if os.path.exists(plot_path_check):
        print(f"Saved: {plot_path_check}")
    print(f"\nAccuracy: {acc:.3f}")
    print(f"\n=== Timing ===")
    print(f"Total batch time: {batch_total_time:.2f}s")
    print(f"Average per image: {avg_time:.3f}s")
    print(f"Images processed: {len(df)}")
