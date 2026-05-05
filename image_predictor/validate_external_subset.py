#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run EL severity prediction on a named subset from an Excel list and compute metrics.

Expected Excel columns:
  - 文件名: image basename without extension or with extension
  - 脱位程度: ground-truth label in {轻, 中, 重}

This helper matches requested filenames against a larger image pool, predicts only
the matched images, and exports merged predictions plus overall and per-class metrics.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms

from model import ResNet50Classifier


IMAGE_SIZE = 224
NUM_CLASSES = 3
LABEL_MAP = {0: "轻", 1: "中", 2: "重"}
LABEL_ORDER = ["轻", "中", "重"]
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".TIFF",
}
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def load_model(model_path: str, device: torch.device):
    model = ResNet50Classifier(
        num_classes=NUM_CLASSES, input_size=IMAGE_SIZE, pretrained=False
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def predict_image(model, image_path: str, transform, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
    pred_label = LABEL_MAP[pred_class]
    probs = {LABEL_MAP[i]: float(probabilities[i].item()) for i in range(NUM_CLASSES)}
    return pred_label, pred_class, probs


def normalize_filename_key(value: str) -> str:
    value = "" if value is None else str(value)
    value = value.strip()
    value = os.path.splitext(value)[0]
    value = value.replace("\u3000", " ")
    value = re.sub(r"\s+", "", value)
    return value.lower()


def build_image_index(image_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        key = normalize_filename_key(path.stem)
        index.setdefault(key, []).append(path)
    return index


def choose_match(request_name: str, candidates: List[Path]) -> Tuple[Path, str]:
    request_stem = os.path.splitext(str(request_name).strip())[0]
    request_key = normalize_filename_key(request_stem)
    exact_stem_matches = [
        path for path in candidates if normalize_filename_key(path.stem) == request_key
    ]
    if len(exact_stem_matches) == 1:
        return exact_stem_matches[0], "exact"
    if len(candidates) == 1:
        return candidates[0], "single_candidate"
    return sorted(candidates)[0], "ambiguous_first_sorted"


def compute_metrics(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
    metrics = [
        ("accuracy", accuracy_score(y_true, y_pred)),
        ("precision_macro", precision_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        ("recall_macro", recall_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        ("f1_macro", f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        ("precision_weighted", precision_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)),
        ("recall_weighted", recall_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)),
        ("f1_weighted", f1_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)),
        ("precision_micro", precision_score(y_true, y_pred, labels=LABEL_ORDER, average="micro", zero_division=0)),
        ("recall_micro", recall_score(y_true, y_pred, labels=LABEL_ORDER, average="micro", zero_division=0)),
        ("f1_micro", f1_score(y_true, y_pred, labels=LABEL_ORDER, average="micro", zero_division=0)),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def main():
    parser = argparse.ArgumentParser(
        description="Validate a named external photo subset for EL severity classification."
    )
    parser.add_argument("--list-xlsx", required=True, help="Excel file with 文件名 and 脱位程度")
    parser.add_argument("--image-dir", required=True, help="Folder containing the large external image pool")
    parser.add_argument("--model", required=True, help="Path to best_model.pth")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default=None, help="cuda or cpu; default auto-detect")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.list_xlsx)
    required_cols = {"文件名", "脱位程度"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    image_index = build_image_index(Path(args.image_dir))
    model = load_model(args.model, device)
    transform = get_transform()

    rows = []
    unmatched_rows = []
    for _, row in df.iterrows():
        requested_name = str(row["文件名"]).strip()
        truth_label = str(row["脱位程度"]).strip()
        key = normalize_filename_key(requested_name)
        candidates = image_index.get(key, [])
        if not candidates:
            unmatched_rows.append(
                {
                    "文件名": requested_name,
                    "脱位程度": truth_label,
                    "match_status": "not_found",
                }
            )
            continue

        matched_path, match_status = choose_match(requested_name, candidates)
        pred_label, pred_class, probs = predict_image(
            model, str(matched_path), transform, device
        )
        rows.append(
            {
                **row.to_dict(),
                "matched_path": str(matched_path),
                "match_status": match_status,
                "预测结果": pred_label,
                "pred_class": pred_class,
                "轻度概率": probs["轻"],
                "中度概率": probs["中"],
                "重度概率": probs["重"],
                "correct": pred_label == truth_label,
            }
        )

    results_df = pd.DataFrame(rows)
    unmatched_df = pd.DataFrame(unmatched_rows)

    results_xlsx = output_dir / "subset_predictions.xlsx"
    results_csv = output_dir / "subset_predictions.csv"
    unmatched_xlsx = output_dir / "subset_unmatched.xlsx"
    metrics_csv = output_dir / "overall_metrics.csv"
    class_report_csv = output_dir / "classification_report.csv"
    confusion_csv = output_dir / "confusion_matrix.csv"

    results_df.to_excel(results_xlsx, index=False)
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    unmatched_df.to_excel(unmatched_xlsx, index=False)

    if results_df.empty:
        raise RuntimeError("No matched images were available for prediction.")

    y_true = results_df["脱位程度"].astype(str).tolist()
    y_pred = results_df["预测结果"].astype(str).tolist()

    metrics_df = compute_metrics(y_true, y_pred)
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    report = classification_report(
        y_true, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        class_report_csv, encoding="utf-8-sig"
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(
        confusion_csv, encoding="utf-8-sig"
    )

    print(f"Matched cases: {len(results_df)}")
    print(f"Unmatched cases: {len(unmatched_df)}")
    print(metrics_df.to_string(index=False))
    print(f"Saved: {results_xlsx}")
    print(f"Saved: {metrics_csv}")


if __name__ == "__main__":
    main()
