"""
Evaluation & Dynamic Threshold Calibration.

1. Loads best checkpoint and evaluates on validation set.
2. Runs precision-recall / threshold sweep to find optimal threshold
   that maximises macro F1-score.
3. Saves calibrated threshold to saved_models/threshold_config.json.
4. Evaluates on test set using calibrated threshold.
5. Saves results/confusion_matrix.png and results/metrics_summary.json.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
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

from src.models import ConsistencyClassifier
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


def find_optimal_threshold(probs, targets, steps=200):
    """
    Sweeps over sigmoid probability thresholds to find the one
    that maximises macro F1-score.
    """
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.1, 0.9, steps):
        preds = (np.array(probs) >= t).astype(int)
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return round(float(best_thresh), 4), round(float(best_f1), 4)


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


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_probs, all_targets, all_cos = [], [], []
    for batch in dataloader:
        text_feats = batch["text_features"].to(device)
        img_feats = batch["image_features"].to(device)
        labels = batch["label"]

        out = model(text_feats, img_feats)
        all_probs.extend(out["probability"].cpu().flatten().tolist())
        all_cos.extend(out["cos_sim"].cpu().flatten().tolist())
        all_targets.extend(labels.tolist())

    return np.array(all_probs), np.array(all_targets), np.array(all_cos)


def evaluate_and_calibrate(
    checkpoint_path: str = "saved_models/best_model.pth",
    csv_path: str = "data/dataset.csv",
    save_dir: str = "saved_models",
    results_dir: str = "results",
):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    save_path = Path(save_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("EVALUATION & DYNAMIC THRESHOLD CALIBRATION")
    print("=" * 70)

    # 1. Load model
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    arch = checkpoint.get("architecture", {"input_dim": 1025, "hidden_dim": 256, "dropout": 0.3})
    model = ConsistencyClassifier(**arch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Eval] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # 2. Load cached features
    cache = preextract_clip_features(csv_path=csv_path)
    _, val_loader, test_loader = get_feature_dataloaders(cache, batch_size=64)

    # ========= PHASE A: Threshold Calibration on Validation Set =========
    print("\n--- PHASE A: Threshold Calibration (Validation Set) ---")
    val_probs, val_targets, val_cos = collect_predictions(model, val_loader, device)

    optimal_thresh, best_macro_f1 = find_optimal_threshold(val_probs, val_targets)
    print(f"  Optimal sigmoid threshold : {optimal_thresh}")
    print(f"  Best macro F1 at thresh   : {best_macro_f1}")

    # Cosine similarity stats
    match_cos = float(val_cos[val_targets == 0].mean()) if (val_targets == 0).any() else 0
    mismatch_cos = float(val_cos[val_targets == 1].mean()) if (val_targets == 1).any() else 0
    print(f"  Mean cos-sim (matched)    : {match_cos:.4f}")
    print(f"  Mean cos-sim (mismatched) : {mismatch_cos:.4f}")
    print(f"  Cos-sim separation gap    : {match_cos - mismatch_cos:.4f}")

    # Save threshold config
    threshold_config = {
        "optimal_threshold": optimal_thresh,
        "calibration_macro_f1": best_macro_f1,
        "calibrated_on": "validation_set",
        "cosine_analysis": {
            "mean_matched": round(match_cos, 4),
            "mean_mismatched": round(mismatch_cos, 4),
            "gap": round(match_cos - mismatch_cos, 4),
        },
    }
    thresh_file = save_path / "threshold_config.json"
    with open(thresh_file, "w") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"  Saved -> {thresh_file}")

    # ========= PHASE B: Final Test Set Evaluation =========
    print("\n--- PHASE B: Test Set Evaluation (Calibrated Threshold) ---")
    test_probs, test_targets, test_cos = collect_predictions(model, test_loader, device)

    # Using calibrated threshold
    test_preds_cal = (test_probs >= optimal_thresh).astype(int)
    # Using default 0.5
    test_preds_default = (test_probs >= 0.5).astype(int)

    metrics_cal = compute_metrics(test_targets, test_preds_cal, test_probs)
    metrics_default = compute_metrics(test_targets, test_preds_default, test_probs)

    print(f"\n  {'Metric':<12} | {'Calibrated (t={optimal_thresh})':>22} | {'Default (t=0.5)':>18}")
    print("  " + "-" * 60)
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"  {k:<12} | {metrics_cal[k]*100:>20.2f}% | {metrics_default[k]*100:>16.2f}%")

    print(f"\n{classification_report(test_targets, test_preds_cal, target_names=['Consistent', 'Inconsistent'])}")

    # Test cosine stats
    test_match_cos = float(test_cos[test_targets == 0].mean()) if (test_targets == 0).any() else 0
    test_mismatch_cos = float(test_cos[test_targets == 1].mean()) if (test_targets == 1).any() else 0

    # 3. Save confusion matrix
    plot_confusion_matrix(
        test_targets, test_preds_cal,
        results_path / "confusion_matrix.png",
        title=f"Test Confusion Matrix (threshold={optimal_thresh})"
    )

    # 4. Save metrics summary
    metrics_summary = {
        "calibrated_threshold": optimal_thresh,
        "test_dataset_size": len(test_targets),
        "test_metrics_calibrated": metrics_cal,
        "test_metrics_default_0.5": metrics_default,
        "cosine_similarity_analysis": {
            "test_mean_matched": round(test_match_cos, 4),
            "test_mean_mismatched": round(test_mismatch_cos, 4),
            "test_gap": round(test_match_cos - test_mismatch_cos, 4),
        },
    }
    metrics_file = results_path / "metrics_summary.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[Eval] Saved metrics -> {metrics_file}")

    print("\n" + "=" * 70)
    print("EVALUATION & CALIBRATION COMPLETE OK")
    print("=" * 70)

    return metrics_summary


if __name__ == "__main__":
    evaluate_and_calibrate()
