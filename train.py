"""
Train a classifier on extracted CLIP+Whisper embeddings to predict TikTok folder categories.

Trains three approaches and picks the best:
  1. k-NN baseline (no training, just nearest neighbors)
  2. Logistic Regression
  3. Small MLP (2 hidden layers)

Outputs:
  - artifacts/model.pt (best model state dict)
  - artifacts/model_config.json (model metadata)
  - Prints confusion matrix and per-class accuracy
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, num_classes, device, epochs=100, lr=1e-3):
    input_dim = X_train.shape[1]
    model = MLP(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class-weighted loss to handle imbalance (soccer=82 vs funny=5)
    class_counts = np.bincount(y_train, minlength=num_classes).astype(float)
    class_counts = np.maximum(class_counts, 1.0)  # avoid div by zero
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes  # normalize
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(y_train).to(device),
    )
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    best_val_acc = 0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.FloatTensor(X_val).to(device))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = (val_preds == y_val).mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_acc


def evaluate(name, y_true, y_pred, label_names):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    # Header
    header = "        " + " ".join(f"{n[:6]:>6}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{label_names[i][:6]:>6}  " + " ".join(f"{v:>6}" for v in row)
        print(row_str)
    acc = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {acc:.1%}")
    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    data = torch.load(ARTIFACTS_DIR / "labeled_embeddings.pt", weights_only=False)
    X = data["features"].numpy()
    y = data["labels"].numpy()
    label_names = data["label_names"]
    num_classes = len(label_names)

    print(f"Loaded {len(X)} samples, {num_classes} classes: {label_names}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: {dict(zip(label_names, np.bincount(y)))}")

    # --- Cross-validation ---
    n_splits = min(5, min(np.bincount(y)))  # can't have more splits than smallest class
    n_splits = max(2, n_splits)
    print(f"\nUsing {n_splits}-fold stratified cross-validation")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {"knn": [], "logreg": [], "mlp": []}
    all_preds = {"knn": np.zeros_like(y), "logreg": np.zeros_like(y), "mlp": np.zeros_like(y)}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 1. k-NN
        k = min(5, len(X_train) - 1)
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(X_train, y_train)
        knn_preds = knn.predict(X_val)
        knn_acc = (knn_preds == y_val).mean()
        results["knn"].append(knn_acc)
        all_preds["knn"][val_idx] = knn_preds

        # 2. Logistic Regression
        lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_val)
        lr_acc = (lr_preds == y_val).mean()
        results["logreg"].append(lr_acc)
        all_preds["logreg"][val_idx] = lr_preds

        # 3. MLP
        mlp_model, mlp_acc = train_mlp(X_train, y_train, X_val, y_val, num_classes, device)
        mlp_preds = mlp_model(torch.FloatTensor(X_val).to(device)).argmax(dim=1).cpu().numpy()
        results["mlp"].append((mlp_preds == y_val).mean())
        all_preds["mlp"][val_idx] = mlp_preds

        print(f"  Fold {fold+1}: kNN={knn_acc:.1%}  LogReg={lr_acc:.1%}  MLP={results['mlp'][-1]:.1%}")

    # Summary
    print(f"\n{'='*60}")
    print("Cross-validation results (mean accuracy):")
    for name, accs in results.items():
        print(f"  {name:>8}: {np.mean(accs):.1%} (+/- {np.std(accs):.1%})")

    # Pick best model type
    mean_accs = {name: np.mean(accs) for name, accs in results.items()}
    best_name = max(mean_accs, key=mean_accs.get)
    print(f"\nBest model: {best_name} ({mean_accs[best_name]:.1%})")

    # Detailed report for best
    evaluate(f"Best Model ({best_name}) - Full CV Predictions", y, all_preds[best_name], label_names)

    # --- Retrain best model on ALL data ---
    print(f"\nRetraining {best_name} on all {len(X)} samples...")

    if best_name == "knn":
        k = min(5, len(X) - 1)
        final_model = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        final_model.fit(X, y)
        # Save as sklearn model
        import pickle
        with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
            pickle.dump(final_model, f)
        config = {"model_type": "knn", "k": k}

    elif best_name == "logreg":
        final_model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        final_model.fit(X, y)
        import pickle
        with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
            pickle.dump(final_model, f)
        config = {"model_type": "logreg"}

    else:  # mlp
        # Train on full data with a small hold-out for early stopping
        split = int(0.9 * len(X))
        perm = np.random.RandomState(42).permutation(len(X))
        X_t, X_v = X[perm[:split]], X[perm[split:]]
        y_t, y_v = y[perm[:split]], y[perm[split:]]
        final_model, _ = train_mlp(X_t, y_t, X_v, y_v, num_classes, device, epochs=200)
        torch.save(final_model.state_dict(), ARTIFACTS_DIR / "model.pt")
        config = {
            "model_type": "mlp",
            "input_dim": int(X.shape[1]),
            "num_classes": num_classes,
            "hidden_dim": 256,
        }

    config["label_names"] = label_names
    config["feature_dim"] = int(X.shape[1])
    config["best_cv_accuracy"] = float(mean_accs[best_name])

    with open(ARTIFACTS_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {ARTIFACTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
