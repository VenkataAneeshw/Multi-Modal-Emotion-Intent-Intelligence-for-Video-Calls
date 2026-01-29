# EMOTIA Product Requirements Document

## 1. Product Overview

### Problem
Video calls remove many human signals. Recruiters, educators, sales teams, and therapists lack objective insights into:
- Emotional state
- Engagement
- Confidence
- Intent (confusion, agreement, hesitation)

Manual observation is subjective, inconsistent, and non-scalable.

### Solution
A real-time multi-modal AI system that analyzes:
- Facial expressions (video)
- Vocal tone (audio)
- Spoken language (text)
- Temporal behavior (over time)

…and produces interpretable, ethical, probabilistic insights.

### Target Users
- Recruiters & hiring platforms
- EdTech platforms
- Sales & customer success teams
- Remote therapy & coaching platforms
- Product teams analyzing user calls

## 2. Core Features

### 2.1 Live Video Call Analysis
- Real-time emotion detection
- Engagement tracking
- Confidence & stress indicators
- Timeline-based emotion shifts

### 2.2 Post-Call Analytics Dashboard
- Emotion timeline
- Intent heatmap
- Modality influence breakdown
- Key moments (confusion spikes, stress peaks)

### 2.3 Multi-Modal Explainability
Why a prediction was made:
- Face vs voice vs text contribution
- Visual overlays (heatmaps)
- Confidence intervals (not hard labels)

### 2.4 Ethics & Bias Controls
- Bias evaluation toggle
- Per-modality opt-out
- Clear disclaimers (non-diagnostic, assistive AI)

## 3. UI / UX Vision

### 3.1 UI Style
- Dark mode only
- Glassmorphism cards
- Neon accent colors (cyan / violet / lime)
- Smooth micro-animations
- Real-time waveform + emotion graphs

### 3.2 Main Dashboard

#### Left Panel
- Live video feed
- Face bounding box
- Micro-expression indicators

#### Center
- Emotion timeline (animated)
- Engagement meter (0–100)
- Confidence score

#### Right Panel
- Intent probabilities
- Stress indicators
- Modality contribution bars

### 3.3 Post-Call Report UI
- Scrollable emotion timeline
- Clickable "critical moments"
- Modality dominance chart
- Exportable report (PDF)

### 3.4 UI Components (Must-Have)
- Animated confidence rings
- Temporal scrubber
- Heatmap overlays
- Tooltips explaining AI decisions

## 4. Technical Architecture

### 4.1 Input Pipeline
- Webcam video (25–30 FPS)
- Microphone audio
- Real-time ASR
- Sliding temporal windows (5–10 sec)

### 4.2 Model Architecture (Production-Grade)

#### 🔹 Visual Branch
- Vision Transformer (ViT) fine-tuned for facial expressions
- Face detection + alignment
- Temporal pooling

#### 🔹 Audio Branch
- Audio → Mel-spectrogram
- CNN + Transformer
- Prosody, pitch, rhythm modeling

#### 🔹 Text Branch
- Transformer-based language model
- Fine-tuned for intent & sentiment
- Confidence / hesitation phrase detection

#### 🔹 Fusion Network (KEY DIFFERENTIATOR)
- Cross-modal attention
- Dynamic modality weighting
- Temporal transformer for sequence learning

#### 🔹 Output Heads
- Emotion classification
- Intent classification
- Engagement regression
- Confidence regression

## 5. Models to Use (Strong + Realistic)

### Visual
- ViT-Base / EfficientNet
- Pretrained on face emotion datasets

### Audio
- Wav2Vec-style embeddings
- CNN-Transformer hybrid

### Text
- Transformer encoder (fine-tuned)
- Focus on conversational intent

### Fusion
- Custom attention-based multi-head network
- (this is your original contribution)

## 6. Datasets (CV-Worthy)

### Facial Emotion
- FER-2013
- AffectNet
- RAF-DB

### Audio Emotion
- RAVDESS
- CREMA-D

### Speech + Intent
- IEMOCAP
- MELD (multi-party dialogue)

### Strategy
- Pretrain each modality separately
- Fine-tune jointly
- Align timestamps across modalities

## 7. Training & Evaluation

### Training
- Multi-task learning
- Weighted losses per output
- Curriculum learning (single → multi-modal)

### Metrics
- F1-score per emotion
- Concordance correlation (regression)
- Confusion matrices
- Per-modality ablation

## 8. Deployment

### Backend
- FastAPI
- GPU inference support
- Streaming inference pipeline

### Frontend
- Next.js / React
- WebRTC video
- Web Audio API
- WebGL visualizations

### Infrastructure
- Dockerized services
- Modular microservices
- Model versioning

## 9. Non-Functional Requirements
- Real-time latency < 200ms
- Modular model replacement
- Privacy-first design
- No biometric storage by default