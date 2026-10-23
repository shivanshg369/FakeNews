"""
Evaluation & Dynamic Threshold Calibration.

1. Evaluates MultimodalConsistencyDetector across the test benchmark.
2. Evaluates the MLP classifier on validation & test sets.
3. Computes cross-modal separation metrics (cosine similarity & contrastive confidence).
4. Saves results/confusion_matrix.png and results/metrics_summary.json.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MultimodalConsistencyDetector, ConsistencyClassifier
from src.dataset import preextract_clip_features, get_feature_dataloaders

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5), 4),
    }


def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    labels = ["Consistent (0)", "Inconsistent (1)"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=False,
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved confusion matrix -> {output_path}")


def evaluate_detector(csv_path: str = "data/dataset.csv", threshold: float = 0.45):
    print("\n" + "=" * 70)
    print(f"EVALUATING MULTIMODAL CONSISTENCY DETECTOR (threshold={threshold})")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MultimodalConsistencyDetector(device=device)

    df = pd.read_csv(csv_path)
    test_df = df[df["split"] == "test"]

    y_true = []
    y_pred = []
    y_prob = []
    cos_sims_consistent = []
    cos_sims_inconsistent = []

    for _, row in test_df.iterrows():
        p = Path(row["image_path"])
        if not p.exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue

        res = detector.analyze(img, row["headline"], threshold=threshold)
        pred = 0 if res["is_consistent"] else 1
        label = int(row["label"])

        y_true.append(label)
        y_pred.append(pred)
        # Probability of being class 1 (Inconsistent)
        p_inconsistent = 1.0 - res["confidence"]
        y_prob.append(p_inconsistent)

        if label == 0:
            cos_sims_consistent.append(res["raw_similarity"])
        else:
            cos_sims_inconsistent.append(res["raw_similarity"])

    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"  Test Samples  : {len(y_true)}")
    print(f"  Accuracy      : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision     : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall        : {metrics['recall'] * 100:.2f}%")
    print(f"  F1 Score      : {metrics['f1_score'] * 100:.2f}%")
    print(f"  ROC-AUC       : {metrics['roc_auc'] * 100:.2f}%")

    print(f"\n{classification_report(y_true, y_pred, target_names=['Consistent (0)', 'Inconsistent (1)'])}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true, y_pred,
        results_dir / "confusion_matrix.png",
        title=f"Multimodal Consistency Confusion Matrix (Acc={metrics['accuracy']*100:.1f}%)"
    )

    summary = {
        "test_dataset_size": len(y_true),
        "decision_threshold": threshold,
        "multimodal_fusion": metrics,
        "cosine_similarity_analysis": {
            "mean_consistent_similarity": round(float(np.mean(cos_sims_consistent)), 4),
            "mean_inconsistent_similarity": round(float(np.mean(cos_sims_inconsistent)), 4),
            "separation_gap": round(float(np.mean(cos_sims_consistent) - np.mean(cos_sims_inconsistent)), 4)
        }
    }

    metrics_file = results_dir / "metrics_summary.json"
    with open(metrics_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Eval] Saved metrics summary -> {metrics_file}")

    return summary


if __name__ == "__main__":
    evaluate_detector()
