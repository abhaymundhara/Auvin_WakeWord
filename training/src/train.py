from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .model import ConvWakeHead, spec_augment
from .paths import CLASSIFIER_PATH, CONFIG_PATH, FEATURES_DIR, LOGS_DIR


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_features():
    X = np.load(FEATURES_DIR / "X.npy")
    y = np.load(FEATURES_DIR / "y.npy")
    w = np.load(FEATURES_DIR / "w.npy")
    k = np.load(FEATURES_DIR / "k.npy")
    return X, y, w, k


def split_data(X, y, w, k, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - val_ratio))
    train_idx, val_idx = idx[:cut], idx[cut:]
    return (
        X[train_idx],
        y[train_idx],
        w[train_idx],
        k[train_idx],
        X[val_idx],
        y[val_idx],
        w[val_idx],
        k[val_idx],
    )


def compute_metrics(scores: np.ndarray, labels: np.ndarray, kinds: np.ndarray | None = None) -> dict:
    preds = scores >= 0.5
    pos = labels == 1
    neg = labels == 0
    recall = float((preds & pos).sum() / max(pos.sum(), 1))
    fp_rate = float((preds & neg).sum() / max(neg.sum(), 1))
    metrics = {
        "recall": recall,
        "fp_rate": fp_rate,
        "mean_pos_score": float(scores[pos].mean()) if pos.any() else 0.0,
        "mean_neg_score": float(scores[neg].mean()) if neg.any() else 0.0,
        "composite": recall - 5 * fp_rate,
    }
    if kinds is not None:
        hard = kinds == "hard_negative"
        random_neg = kinds == "random_negative"
        if hard.any():
            metrics["hard_negative_fpr"] = float((preds & hard).sum() / max(hard.sum(), 1))
        if random_neg.any():
            metrics["real_speech_fpr"] = float((preds & random_neg).sum() / max(random_neg.sum(), 1))
    return metrics


def train_epoch(model, loader, optimizer, device, pos_weight, use_aug=True):
    model.train()
    bce = nn.BCELoss(reduction="none")
    total_loss = 0.0
    for xb, yb, wb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        wb = wb.to(device)
        if use_aug:
            xb = spec_augment(xb)
        scores = model(xb)
        loss = bce(scores, yb) * wb * pos_weight
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, X, y, kinds, device):
    model.eval()
    bs = 2048
    scores_list = []
    for i in range(0, len(y), bs):
        xb = torch.from_numpy(X[i : i + bs]).to(device)
        scores = model(xb).cpu().numpy()
        scores_list.append(scores)
    scores = np.concatenate(scores_list)
    return compute_metrics(scores, y, kinds)


def balance_weights(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    pos_mask = y == 1
    neg_mask = y == 0
    pos_total = w[pos_mask].sum()
    neg_total = w[neg_mask].sum()
    out = w.copy()
    if pos_total > 0 and neg_total > 0:
        scale = neg_total / pos_total
        out[pos_mask] *= scale
    return out


def apply_training_weights(
    y: np.ndarray,
    w: np.ndarray,
    kinds: np.ndarray,
    hard_negative_weight: float,
) -> np.ndarray:
    """Apply configured weights while preserving featurization's class markers."""
    out = w.copy()
    hard_negative = kinds == "hard_negative"
    out[hard_negative] = hard_negative_weight
    return balance_weights(y, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Auvin wake word classifier")
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()

    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_cfg = config["training"]
    gates = config["gates"]
    epochs = args.epochs or train_cfg["epochs"]

    X, y, w, k = load_features()
    w = apply_training_weights(y, w, k, train_cfg["hard_negative_weight"])
    X_train, y_train, w_train, _, X_val, y_val, _, k_val = split_data(X, y, w, k)

    device = pick_device()
    print(f"Training on {device}")

    pos_weight = torch.tensor([1.0], device=device)

    # MPS smoke test
    model = ConvWakeHead().to(device)
    smoke_x = torch.randn(4, 16, 96, device=device)
    smoke_y = torch.randint(0, 2, (4,), device=device).float()
    out = model(smoke_x)
    loss = nn.functional.binary_cross_entropy(out, smoke_y)
    loss.backward()
    model.zero_grad(set_to_none=True)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(w_train),
    )
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "train.log"
    best_score = -1.0
    best_state = None
    patience = train_cfg["patience"]
    stale = 0

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
            metrics = evaluate(model, X_val, y_val, k_val, device)
            scheduler.step()
            line = f"epoch={epoch} loss={loss:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(line)
            log.write(line + "\n")
            log.flush()

            if metrics["composite"] > best_score:
                best_score = metrics["composite"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise SystemExit("Training failed to produce a model")

    model.load_state_dict(best_state)
    model.cpu()
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFIER_PATH.with_suffix(".pt"))

    final_metrics = evaluate(model, X_val, y_val, k_val, torch.device("cpu"))
    report = {"metrics": final_metrics, "gates": gates, "passed": True}
    for key, target in [
        ("recall", gates["recall_min"]),
        ("real_speech_fpr", gates["real_speech_fpr_max"]),
        ("hard_negative_fpr", gates["hard_negative_fpr_max"]),
        ("mean_pos_score", gates["mean_pos_score_min"]),
        ("mean_neg_score", gates["mean_neg_score_max"]),
    ]:
        val = final_metrics.get(key, 0)
        if key == "recall" or key == "mean_pos_score":
            ok = val >= target
        elif key == "mean_neg_score":
            ok = val <= target
        else:
            ok = val <= target
        report["passed"] = report["passed"] and ok

    with (LOGS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    if not report["passed"]:
        print("Warning: not all production gates passed. Model saved anyway for iteration.")


if __name__ == "__main__":
    main()
