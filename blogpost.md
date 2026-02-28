# I trained a neural net to sort my TikTok saves

On TikTok, you can save videos to folders. But the UX for it is three separate taps: save, view folders, pick a folder. That friction is enough that most people just don't bother. Everything gets dumped into one unsorted pile, and the folder system goes unused.

I wanted to fix that. What if, when you press save, the app already knew which folder the video should go in? You'd get a one-tap confirmation instead of a three-tap flow.

The question was whether you could actually predict someone's personal folder taxonomy from the video content alone. These are short-form videos with a lot going on: visuals, audio, speech, music, text overlays. But the hypothesis was that each folder has a consistent enough signal across those modalities that a classifier could learn it.

So I built one.

## The pipeline

I didn't train a video model from scratch. Instead I used pretrained foundation models for feature extraction and trained a lightweight classifier on top.

For visual features, I sample 5 frames uniformly from each video and encode them with CLIP (ViT-B/32), which produces a 512-d embedding per frame. I average-pool these into a single visual vector per video. For audio features, I extract the audio track, transcribe it with Whisper, and encode the transcript with CLIP's text encoder, producing another 512-d vector. This captures the spoken content, which turns out to be a strong discriminative signal between categories.

I concatenate both modality vectors into a 1024-d representation, L2-normalize each modality independently so neither dominates, and feed it into a two-layer MLP (256 → 128 → N classes) trained with class-weighted cross-entropy. Class weighting was necessary because my labeled data was heavily skewed toward certain folders. Without it, the model converged to predicting the majority class and ignoring everything else. With it, it learned the minority categories properly.

The whole classifier trains in seconds. The expensive part is the one-time feature extraction pass through CLIP and Whisper, which takes about 10 minutes for 600 videos on a consumer GPU.

## The data

When TikTok was about to get banned, I saved all my bookmarked videos to a USB drive. Around 600 of them. Some I had already sorted into folders, some I hadn't. The sorted ones became my training set. The unsorted ones became the inference set.

I started with 128 labeled videos across 6 categories and got 93.8% cross-validation accuracy. After labeling more data through the tool I built (described below), I expanded to 213 videos across 8 categories and the model held at ~90% accuracy. Per-category recall varies: folders with strong audiovisual signatures (like Quran recitation, which has distinct visual framing and Arabic speech) reach near-perfect recall. More subjective categories are harder to learn, especially with fewer examples, which is expected.

## The sorting interface

To make the labeling loop fast, I built a local web UI modeled after TikTok's own player. Full-screen video in the center, folder buttons on the left, and the model's top-3 predictions with confidence scores on the right. The predicted folder button is visually highlighted so you can see the suggestion at a glance.

You watch a video, press the correct folder button (or a keyboard shortcut, 1-8), and it moves the file and auto-advances to the next video. There's a retrain button that re-runs the full pipeline in the background: feature extraction, training, and prediction regeneration. The UI refreshes with updated predictions when it finishes.

It works well. You settle into a flow where each video takes a couple of seconds, and the model's predictions are right often enough that you're mostly just confirming.

## What's next

A few directions I'm thinking about. Active learning, where the UI prioritizes showing videos the model is least confident about, since those are the labels that would improve the model the most. Zero-shot bootstrapping for new folders using CLIP's text-image alignment, so you can create a "cooking" folder and get rough predictions from just the folder name before you have any labeled examples. And eventually a browser extension that hooks into TikTok's save action and suggests a folder at save time.

## Stack

Python end to end. CLIP and Whisper for feature extraction, PyTorch for the classifier, FastAPI for serving, and a single HTML file with inline CSS/JS for the frontend. About 900 lines across 4 files, no build step.
