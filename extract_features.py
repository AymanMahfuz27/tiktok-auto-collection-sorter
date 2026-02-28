"""
Extract multi-modal features from TikTok videos using CLIP (visual) and Whisper (audio).

For each video:
  1. Sample N frames uniformly → CLIP vision encoder → visual embedding (512-d)
  2. Extract audio → Whisper transcription → CLIP text encoder → text embedding (512-d)
  3. Concatenate → 1024-d feature vector

Saves:
  - embeddings.pt  (dict with 'features', 'labels', 'label_names', 'video_paths')
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

import torch
import clip
import cv2
import numpy as np
import whisper
from PIL import Image
from tqdm import tqdm


DATA_DIR = Path(__file__).parent / "data" / "Favorites" / "videos"
OUTPUT_DIR = Path(__file__).parent / "artifacts"
N_FRAMES = 5
CLIP_MODEL = "ViT-B/32"
WHISPER_MODEL = "base"  # small and fast, good enough for transcription


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models(device):
    print(f"Loading CLIP ({CLIP_MODEL})...")
    clip_model, clip_preprocess = clip.load(CLIP_MODEL, device=device)

    print(f"Loading Whisper ({WHISPER_MODEL})...")
    whisper_model = whisper.load_model(WHISPER_MODEL, device=device)

    return clip_model, clip_preprocess, whisper_model


def extract_visual_features(video_path, clip_model, preprocess, device, n_frames=N_FRAMES):
    """Sample n_frames uniformly from video, encode with CLIP, average-pool."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    embeddings = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(img_input)
        embeddings.append(emb.cpu())

    cap.release()

    if not embeddings:
        return None

    # Average pool across frames → single 512-d vector
    stacked = torch.cat(embeddings, dim=0)
    return stacked.mean(dim=0)


def extract_audio_features(video_path, whisper_model, clip_model, device):
    """Extract audio → transcribe with Whisper → encode transcript with CLIP text encoder."""
    # Extract audio to a temp wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", tmp_path],
            capture_output=True, timeout=30
        )
        if result.returncode != 0:
            return None

        # Transcribe
        transcription = whisper_model.transcribe(tmp_path, fp16=(device == "cuda"))
        text = transcription["text"].strip()

        if not text:
            return None

        # Encode transcript with CLIP text encoder
        tokens = clip.tokenize([text[:77]], truncate=True).to(device)  # CLIP max 77 tokens
        with torch.no_grad():
            text_emb = clip_model.encode_text(tokens)

        return text_emb.cpu().squeeze(0), text

    except Exception as e:
        print(f"  Audio extraction failed: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def discover_dataset(data_dir):
    """
    Returns:
        labeled: list of (video_path, folder_name) for videos in subfolders
        unlabeled: list of video_path for videos in root (unsorted)
        label_names: sorted list of folder names
    """
    labeled = []
    unlabeled = []
    folders = set()

    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            folder_name = item.name
            for vid in sorted(item.glob("*.mp4")):
                labeled.append((vid, folder_name))
                folders.add(folder_name)
        elif item.suffix == ".mp4":
            unlabeled.append(item)

    label_names = sorted(folders)
    return labeled, unlabeled, label_names


def main():
    device = get_device()
    print(f"Using device: {device}")

    clip_model, preprocess, whisper_model = load_models(device)
    labeled, unlabeled, label_names = discover_dataset(DATA_DIR)

    print(f"\nDataset summary:")
    print(f"  Labeled videos: {len(labeled)}")
    print(f"  Unlabeled videos: {len(unlabeled)}")
    print(f"  Categories: {label_names}")

    label_to_idx = {name: i for i, name in enumerate(label_names)}

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Extract features for LABELED videos ---
    print(f"\n{'='*60}")
    print("Extracting features for labeled videos...")
    print(f"{'='*60}")

    features_list = []
    labels_list = []
    paths_list = []
    transcripts = {}

    for video_path, folder_name in tqdm(labeled, desc="Labeled"):
        # Visual
        vis_emb = extract_visual_features(video_path, clip_model, preprocess, device)
        if vis_emb is None:
            print(f"  Skipping {video_path.name} (no visual features)")
            continue

        # Audio
        audio_result = extract_audio_features(video_path, whisper_model, clip_model, device)
        if audio_result is not None:
            audio_emb, transcript = audio_result
            transcripts[str(video_path)] = transcript
        else:
            # Zero vector as fallback for missing audio
            audio_emb = torch.zeros(vis_emb.shape[0])

        # Normalize each modality before concatenation
        vis_emb = vis_emb / vis_emb.norm()
        audio_emb = audio_emb / (audio_emb.norm() + 1e-8)

        # Concatenate: [visual_512 | audio_512] = 1024-d
        combined = torch.cat([vis_emb, audio_emb], dim=0)
        features_list.append(combined)
        labels_list.append(label_to_idx[folder_name])
        paths_list.append(str(video_path))

    labeled_data = {
        "features": torch.stack(features_list),
        "labels": torch.tensor(labels_list),
        "label_names": label_names,
        "video_paths": paths_list,
    }
    torch.save(labeled_data, OUTPUT_DIR / "labeled_embeddings.pt")
    print(f"\nSaved labeled embeddings: {labeled_data['features'].shape}")

    # --- Extract features for UNLABELED videos ---
    print(f"\n{'='*60}")
    print("Extracting features for unlabeled videos...")
    print(f"{'='*60}")

    unlabeled_features = []
    unlabeled_paths = []

    for video_path in tqdm(unlabeled, desc="Unlabeled"):
        vis_emb = extract_visual_features(video_path, clip_model, preprocess, device)
        if vis_emb is None:
            print(f"  Skipping {video_path.name}")
            continue

        audio_result = extract_audio_features(video_path, whisper_model, clip_model, device)
        if audio_result is not None:
            audio_emb, transcript = audio_result
            transcripts[str(video_path)] = transcript
        else:
            audio_emb = torch.zeros(vis_emb.shape[0])

        vis_emb = vis_emb / vis_emb.norm()
        audio_emb = audio_emb / (audio_emb.norm() + 1e-8)

        combined = torch.cat([vis_emb, audio_emb], dim=0)
        unlabeled_features.append(combined)
        unlabeled_paths.append(str(video_path))

    if unlabeled_features:
        unlabeled_data = {
            "features": torch.stack(unlabeled_features),
            "video_paths": unlabeled_paths,
        }
        torch.save(unlabeled_data, OUTPUT_DIR / "unlabeled_embeddings.pt")
        print(f"\nSaved unlabeled embeddings: {unlabeled_data['features'].shape}")

    # Save transcripts for inspection
    with open(OUTPUT_DIR / "transcripts.json", "w") as f:
        json.dump(transcripts, f, indent=2)
    print(f"Saved {len(transcripts)} transcripts")

    print("\nDone! Artifacts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
