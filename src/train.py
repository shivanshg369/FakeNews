"""
Training loop for the Consistency Classifier (MLP head on frozen CLIP features).

Design:
  - Uses pre-extracted, cached CLIP features (text_features, image_features).
  - Trains only the MLP head (~263K params) with BCEWithLogitsLoss + AdamW.
  - 10 epochs, lr=1e-3.
  - Saves best checkpoint to saved_models/best_model.pth based on best val F1.
  - Prints per-epoch class distributions to detect collapse early.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ConsistencyClassifier
from src.dataset import generate_synthetic_benchmark, preextract_clip_features, get_feature_dataloaders

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    class_0_count = 0
    class_1_count = 0

    for batch in dataloader:
        text_feats = batch["text_features"].to(device)
        img_feats = batch["image_features"].to(device)
        labels = batch["label"].unsqueeze(1).to(device)  # (B, 1)

        optimizer.zero_grad()
        out = model(text_feats, img_feats)
        loss = criterion(out["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (out["probability"].detach() >= 0.5).long().cpu().flatten()
        targets = labels.long().cpu().flatten()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        class_0_count += (targets == 0).sum().item()
        class_1_count += (targets == 1).sum().item()

    n = len(all_targets)
    avg_loss = total_loss / n
    acc = accuracy_score(all_targets, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)
    return avg_loss, acc, f1, class_0_count, class_1_count


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    for batch in dataloader:
        text_feats = batch["text_features"].to(device)
        img_feats = batch["image_features"].to(device)
        labels = batch["label"].unsqueeze(1).to(device)

        out = model(text_feats, img_feats)
        loss = criterion(out["logits"], labels)
        total_loss += loss.item() * len(labels)

        probs = out["probability"].cpu().flatten()
        preds = (probs >= 0.5).long()
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.long().cpu().flatten().tolist())
        all_probs.extend(probs.tolist())

    n = len(all_targets)
    avg_loss = total_loss / n
    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)
    return avg_loss, acc, f1, prec, rec, all_probs, all_targets


def train_model(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    save_dir: str = "saved_models",
    results_dir: str = "results",
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("TRAINING: ConsistencyClassifier (MLP on frozen CLIP features)")
    print("=" * 70)
    print(f"  Device      : {device}")
    print(f"  Epochs      : {epochs}")
    print(f"  LR          : {lr}")
    print(f"  Batch Size  : {batch_size}")

    # 1. Regenerate dataset + extract features
    generate_synthetic_benchmark(data_dir="data", num_per_category=40, random_seed=42)

    # Force fresh feature extraction by deleting cache
    cache_file = Path("data/clip_cache/features.pt")
    if cache_file.exists():
        cache_file.unlink()
        print("[Train] Cleared stale feature cache.")

    cache = preextract_clip_features(csv_path="data/dataset.csv")
    train_loader, val_loader, test_loader = get_feature_dataloaders(cache, batch_size=batch_size)

    print(f"  Train : {len(train_loader.dataset)} samples")
    print(f"  Val   : {len(val_loader.dataset)} samples")
    print(f"  Test  : {len(test_loader.dataset)} samples")

    # 2. Initialise classifier
    model = ConsistencyClassifier(input_dim=1025, hidden_dim=256, dropout=0.3).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_epoch = 0
    history = []

    print("\n" + "-" * 85)
    print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | {'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>6} | {'Cl0':>4} {'Cl1':>4} | Status")
    print("-" * 85)

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, c0, c1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1, va_prec, va_rec, _, _ = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        status = ""
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            status = "★ BEST"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": va_f1,
                "val_acc": va_acc,
                "val_loss": va_loss,
                "architecture": {"input_dim": 1025, "hidden_dim": 256, "dropout": 0.3},
            }, save_path / "best_model.pth")

        print(f"{epoch:>3} | {tr_loss:>7.4f} | {tr_acc*100:>5.1f}% | {tr_f1*100:>5.1f}% | {va_loss:>7.4f} | {va_acc*100:>5.1f}% | {va_f1*100:>5.1f}% | {c0:>4} {c1:>4} | {status}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
            "class_0_train": c0, "class_1_train": c1,
            "time_sec": round(dt, 2),
        })

    total_time = time.time() - t_start
    print("-" * 85)
    print(f"Training complete in {total_time:.1f}s | Best epoch: {best_epoch} (Val F1: {best_val_f1*100:.1f}%)")

    # Save history
    with open(results_path / "training_history.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_f1": best_val_f1, "epochs": history}, f, indent=2)

    return {"best_epoch": best_epoch, "best_val_f1": best_val_f1}


if __name__ == "__main__":
    train_model(epochs=10, batch_size=32, lr=1e-3)
