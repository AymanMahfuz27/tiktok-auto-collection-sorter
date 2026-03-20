# TikTok Video Auto-Sorter

A multimodal machine learning system that automatically categorizes TikTok videos into user-defined folders by analyzing visual and audio content. Achieves **~90% accuracy** on personal video collections using transfer learning from foundation models.
In depth read: https://aymanmahfuz27-tiktok-auto-collection-sorter.mintlify.app/concepts/architecture
## Problem & Solution

TikTok's folder organization requires three taps (save → view folders → select folder), creating enough friction that most users abandon the feature. This project solves that by predicting the correct folder at save time, reducing the flow to a single confirmation tap.

**Key Challenge**: Can we predict a user's personal organizational taxonomy from video content alone? Short-form videos contain multiple modalities (visuals, audio, speech, text overlays), but each folder category has consistent enough signals across these modalities that a classifier can learn them.

## Architecture

### Multimodal Feature Extraction
- **Visual Features**: Sample 5 frames uniformly from each video → encode with CLIP (ViT-B/32) → average-pool to 512-d vector
- **Audio Features**: Extract audio track → transcribe with Whisper → encode transcript with CLIP's text encoder → 512-d vector
- **Combined Representation**: Concatenate both modalities → 1024-d vector, L2-normalized per modality to prevent dominance

### Classification Pipeline
- Two-layer MLP (256 → 128 → N classes)
- Class-weighted cross-entropy loss to handle imbalanced data
- Trains in seconds; feature extraction (~10 min for 600 videos) is the bottleneck

### Why This Approach?
Instead of training a video model from scratch, this project leverages pretrained foundation models (CLIP, Whisper) for feature extraction and trains a lightweight classifier on top. This provides:
- **Efficiency**: Fast training and inference
- **Effectiveness**: Strong performance with minimal labeled data
- **Transferability**: Foundation models capture rich semantic representations

## Results

- **Accuracy**: ~90% on 213 labeled videos across 8 categories
- **Initial Performance**: 93.8% cross-validation accuracy on 128 videos (6 categories)
- **Per-Category Performance**: Categories with strong audiovisual signatures (e.g., Quran recitation with distinct visual framing and Arabic speech) achieve near-perfect recall
- **Data Efficiency**: Strong performance with relatively small labeled datasets

## Tech Stack

- **ML/AI**: PyTorch, CLIP (OpenAI), Whisper (OpenAI)
- **Backend**: FastAPI
- **Frontend**: Vanilla HTML/CSS/JavaScript (single file, no build step)
- **Language**: Python end-to-end
- **Codebase**: ~900 lines across 4 core files

## Features

### Interactive Labeling Interface
- Full-screen video player modeled after TikTok's UI
- Real-time model predictions with top-3 confidence scores
- Keyboard shortcuts (1-8) for rapid labeling
- Visual highlighting of predicted folder
- Auto-advance to next video after labeling

### Active Learning Workflow
- Retrain button triggers full pipeline refresh (feature extraction → training → prediction regeneration)
- UI updates automatically when retraining completes
- Enables iterative improvement through rapid labeling loops

## Project Structure

```
tiktoks/
├── extract_features.py    # Multimodal feature extraction (CLIP + Whisper)
├── train.py               # Model training (k-NN, Logistic Regression, MLP)
├── predict.py             # Inference pipeline
├── server.py              # FastAPI backend
├── index.html             # Frontend UI
├── artifacts/             # Model checkpoints, embeddings, predictions
└── data/                  # Video dataset
```

## Usage

1. **Extract Features**: Run `python extract_features.py` to process videos through CLIP and Whisper
2. **Train Model**: Run `python train.py` to train the classifier
3. **Generate Predictions**: Run `python predict.py` to create predictions for unlabeled videos
4. **Launch UI**: Run `python server.py` and open the web interface for interactive labeling

## Future Directions

- **Active Learning**: Prioritize videos with lowest model confidence for labeling
- **Zero-Shot Bootstrapping**: Use CLIP's text-image alignment to bootstrap new folders from folder names alone
- **Browser Extension**: Integrate with TikTok's save action to suggest folders at save time

## Key Technical Highlights

- **Multimodal Learning**: Combines visual and audio/text features for robust classification
- **Transfer Learning**: Leverages pretrained foundation models for efficient feature extraction
- **Class Imbalance Handling**: Implements class-weighted loss to learn minority categories effectively
- **Full-Stack ML System**: End-to-end pipeline from feature extraction to interactive UI
- **Production-Ready Architecture**: Clean separation of concerns, efficient inference, scalable design

