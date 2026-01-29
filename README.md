# EMOTIA - Multi-Modal Emotion & Intent Intelligence
# Production-Ready AI System for Video Call Analysis

## Overview
EMOTIA is a real-time multi-modal AI platform that analyzes video calls to infer emotional state, conversational intent, engagement, and confidence. It fuses facial expressions (vision), vocal tone (audio), spoken language (text), and temporal context using advanced neural architectures.

## Architecture
- **Vision Branch**: Vision Transformer (ViT) for facial emotion detection
- **Audio Branch**: CNN + Transformer for prosody and vocal analysis
- **Text Branch**: Transformer encoder for intent and sentiment analysis
- **Fusion Network**: Cross-modal attention with temporal modeling
- **Outputs**: Emotion classification, intent classification, engagement regression, confidence estimation

## Features
- Real-time inference (<200ms latency)
- Ethical bias controls and explainability
- Modular, Docker-ready deployment
- Dark-mode futuristic UI with glassmorphism
- Live video analysis with emotion timelines

## Quick Start
1. Clone the repository
2. Run `docker-compose up --build`
3. Access frontend at http://localhost:3000
4. Backend API at http://localhost:8000

## Project Structure
```
├── models/          # ML model implementations
├── backend/         # FastAPI inference service
├── frontend/        # Next.js UI application
├── data/           # Datasets and preprocessing
├── scripts/        # Training and evaluation scripts
├── tests/          # Unit and integration tests
├── docs/           # Documentation and diagrams
└── docker-compose.yml
```

## Requirements
- Docker and Docker Compose
- GPU recommended for training/inference
- Python 3.11+, Node.js 18+

## Training
See `scripts/train.py` for multi-modal training pipeline.

## Evaluation
Run `scripts/evaluate.py` for metrics and ablation studies.

## Ethics & Limitations
- Non-diagnostic AI tool
- Bias evaluation available
- No biometric data storage by default
- See `docs/ethics.md` for details

## License
MIT License