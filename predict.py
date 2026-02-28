"""
Predict folder assignments for unsorted TikTok videos.

Uses the trained classifier to predict which folder each unsorted video belongs to.
Outputs a ranked prediction with confidence scores, and optionally moves/copies files.

Usage:
  python predict.py                    # Just print predictions
  python predict.py --move             # Actually move files into predicted folders
  python predict.py --threshold 0.5    # Only predict if confidence > threshold
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DATA_DIR = Path(__file__).parent / "data" / "Favorites" / "videos"


class MLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_model(config):
    model_type = config["model_type"]

    if model_type in ("knn", "logreg"):
        with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
            model = pickle.load(f)
        return model, model_type

    elif model_type == "mlp":
        model = MLP(config["input_dim"], config["num_classes"], config.get("hidden_dim", 256))
        model.load_state_dict(torch.load(ARTIFACTS_DIR / "model.pt", weights_only=True))
        model.eval()
        return model, model_type


def predict_sklearn(model, features):
    """Returns (predicted_labels, probabilities) for sklearn models."""
    probs = model.predict_proba(features)
    preds = probs.argmax(axis=1)
    return preds, probs


def predict_mlp(model, features, device="cpu"):
    """Returns (predicted_labels, probabilities) for MLP model."""
    model = model.to(device)
    with torch.no_grad():
        logits = model(torch.FloatTensor(features).to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return preds, probs


def main():
    parser = argparse.ArgumentParser(description="Predict folders for unsorted TikToks")
    parser.add_argument("--move", action="store_true", help="Actually move files to predicted folders")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum confidence to auto-assign (0-1)")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions per video")
    args = parser.parse_args()

    # Load config and model
    with open(ARTIFACTS_DIR / "model_config.json") as f:
        config = json.load(f)

    label_names = config["label_names"]
    model, model_type = load_model(config)

    # Load unlabeled embeddings
    unlabeled_data = torch.load(ARTIFACTS_DIR / "unlabeled_embeddings.pt", weights_only=False)
    features = unlabeled_data["features"].numpy()
    video_paths = unlabeled_data["video_paths"]

    print(f"Predicting folders for {len(video_paths)} unsorted videos")
    print(f"Model: {model_type} | Categories: {label_names}")
    print(f"Confidence threshold: {args.threshold}")
    print()

    # Predict
    if model_type in ("knn", "logreg"):
        preds, probs = predict_sklearn(model, features)
    else:
        preds, probs = predict_mlp(model, features)

    # Tally and display
    folder_counts = {name: 0 for name in label_names}
    skipped = 0
    assignments = []

    for i, (video_path, pred_idx) in enumerate(zip(video_paths, preds)):
        video_name = Path(video_path).name
        confidence = probs[i][pred_idx]
        predicted_folder = label_names[pred_idx]

        # Top-k predictions
        top_k_idx = np.argsort(probs[i])[::-1][:args.top_k]
        top_k = [(label_names[j], probs[i][j]) for j in top_k_idx]

        if confidence >= args.threshold:
            folder_counts[predicted_folder] += 1
            assignments.append((video_path, predicted_folder, confidence))
            status = "ASSIGN"
        else:
            skipped += 1
            status = "SKIP  "

        top_k_str = " | ".join(f"{name}: {conf:.0%}" for name, conf in top_k)
        print(f"  [{status}] {video_name[:40]:40s} → {predicted_folder:12s} ({confidence:.0%})  [{top_k_str}]")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for name in label_names:
        print(f"  {name:15s}: {folder_counts[name]:4d} videos")
    print(f"  {'SKIPPED':15s}: {skipped:4d} videos (below {args.threshold:.0%} threshold)")
    print(f"  {'TOTAL':15s}: {len(video_paths):4d} videos")

    # Move files if requested
    if args.move and assignments:
        print(f"\nMoving {len(assignments)} files...")
        for video_path, folder, conf in assignments:
            src = Path(video_path)
            dst_dir = DATA_DIR / folder
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / src.name
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"  Moved {src.name} → {folder}/")
            elif dst.exists():
                print(f"  Already exists: {folder}/{src.name}")
        print("Done moving files!")
    elif args.move:
        print("No files to move (all below threshold).")

    # Save predictions to JSON for review
    predictions = []
    for i, video_path in enumerate(video_paths):
        top_k_idx = np.argsort(probs[i])[::-1][:args.top_k]
        predictions.append({
            "video": Path(video_path).name,
            "predicted_folder": label_names[preds[i]],
            "confidence": float(probs[i][preds[i]]),
            "top_predictions": [
                {"folder": label_names[j], "confidence": float(probs[i][j])}
                for j in top_k_idx
            ],
        })

    with open(ARTIFACTS_DIR / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nFull predictions saved to {ARTIFACTS_DIR / 'predictions.json'}")


if __name__ == "__main__":
    main()
