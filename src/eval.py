from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

from .pipeline import process_image
from .utils import list_images, ensure_dirs

LABELS = ["email", "invoice", "news", "receipt"]

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
    ensure_dirs(outdir)
    images = []
    # prefer dataset/<label>/*
    for lab in LABELS:
        p = os.path.join(dataset_dir, lab)
        if os.path.isdir(p):
            images.extend(list_images(p))
    if not images:
        images = list_images(dataset_dir)

    if limit and limit > 0:
        images = images[:limit]

    rows = []
    for img_path in images:
        res = process_image(
            image_path=img_path,
            outdir=outdir,
            model=model,
            use_llm=use_llm,
            tesseract_lang=tesseract_lang,
            annotate=annotate,
        )
        pred = res.get("document_type")
        true = _true_label_from_path(img_path)
        rows.append({
            "image": img_path,
            "true_label": true,
            "pred_label": pred,
            "confidence": res.get("meta", {}).get("classification_confidence"),
            "method": res.get("meta", {}).get("classification_method"),
        })

    df = pd.DataFrame(rows)
    metrics_path = os.path.join(outdir, "metrics", "predictions.csv")
    df.to_csv(metrics_path, index=False)

    # accuracy (ignore unknown)
    df_known = df[df["true_label"].isin(LABELS)]
    acc = (df_known["true_label"] == df_known["pred_label"]).mean() if len(df_known) else 0.0

    summary_path = os.path.join(outdir, "metrics", "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Images: {len(df)}\n")
        f.write(f"Known-label images: {len(df_known)}\n")
        f.write(f"Accuracy: {acc:.3f}\n")

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
        plot_path = os.path.join(outdir, "metrics", "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"Saved: {metrics_path}")
    print(f"Saved: {summary_path}")
    if os.path.exists(os.path.join(outdir, "metrics", "confusion_matrix.png")):
        print(f"Saved: {os.path.join(outdir, 'metrics', 'confusion_matrix.png')}")
    print(f"Accuracy: {acc:.3f}")
