# EMOTIA Architecture

## System Overview

EMOTIA is a multi-modal AI system that analyzes video calls to infer emotional state, conversational intent, engagement, and confidence using facial expressions, vocal tone, spoken language, and temporal context.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │    │  Audio Input    │    │   Text Input    │
│   (25-30 FPS)   │    │  (16kHz WAV)    │    │  (ASR Trans.)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Vision Branch   │    │ Audio Branch    │    │ Text Branch     │
│ • ViT-Base      │    │ • CNN + Trans.  │    │ • BERT Encoder  │
│ • Face Detect   │    │ • Wav2Vec2      │    │ • Intent Detect │
│ • Emotion Class │    │ • Prosody       │    │ • Sentiment     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │   Cross-Modal Fusion        │
                   │ • Attention Mechanism       │
                   │ • Dynamic Weighting         │
                   │ • Temporal Transformer      │
                   │ • Modality Contributions    │
                   └─────────────────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │   Multi-Task Outputs        │
                   │ • Emotion Classification    │
                   │ • Intent Classification     │
                   │ • Engagement Regression     │
                   │ • Confidence Estimation     │
                   └─────────────────────────────┘
```

## Component Details

### Vision Branch
- **Input**: RGB video frames (224x224)
- **Face Detection**: OpenCV Haar cascades
- **Feature Extraction**: Vision Transformer (ViT-Base)
- **Fine-tuning**: FER-2013, AffectNet, RAF-DB datasets
- **Output**: Emotion logits (7 classes), confidence score

### Audio Branch
- **Input**: Audio waveforms (16kHz, 3-second windows)
- **Preprocessing**: Mel-spectrogram extraction
- **Feature Extraction**: Wav2Vec2 + CNN layers
- **Prosody Analysis**: Pitch, rhythm, energy features
- **Output**: Emotion logits, stress/confidence score

### Text Branch
- **Input**: Transcribed speech text
- **Preprocessing**: Tokenization, cleaning
- **Feature Extraction**: BERT-base for intent/sentiment
- **Intent Detection**: Hesitation phrases, confidence markers
- **Output**: Intent logits (5 classes), sentiment logits

### Fusion Network
- **Modality Projection**: Linear layers to common embedding space (256D)
- **Cross-Attention**: Multi-head attention between modalities
- **Temporal Modeling**: Transformer encoder for sequence processing
- **Dynamic Weighting**: Learned modality importance scores
- **Outputs**: Fused predictions with contribution weights

## Data Flow

1. **Input Processing**: Video frames, audio chunks, ASR text
2. **Sliding Windows**: 5-10 second temporal windows
3. **Feature Extraction**: Parallel processing per modality
4. **Fusion**: Cross-modal attention and temporal aggregation
5. **Prediction**: Multi-task classification/regression
6. **Explainability**: Modality contribution scores

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Application                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                WebRTC Video Stream                 │    │
│  │  • Camera Access                                  │    │
│  │  • Audio Capture                                  │    │
│  │  • Real-time Streaming                             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Inference Pipeline                    │    │
│  │  • Model Loading                                  │    │
│  │  • Preprocessing                                  │    │
│  │  • GPU Inference                                  │    │
│  │  • Post-processing                                │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Real-time Processing                  │    │
│  │  • Sliding Window Buffering                       │    │
│  │  • Asynchronous Processing                        │    │
│  │  • Streaming Responses                            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Response Formatting                      │
│  • JSON API Responses                                     │
│  • Real-time WebSocket Updates                            │
│  • Batch Processing for Post-call Analysis                │
└─────────────────────────────────────────────────────────────┘
```

## Performance Requirements

- **Latency**: <200ms end-to-end
- **Throughput**: 25-30 FPS video processing
- **Accuracy**: F1 > 0.80 for emotion classification
- **Scalability**: Horizontal scaling with load balancer
- **Reliability**: 99.9% uptime, graceful degradation

## Security Considerations

- **Data Privacy**: No biometric storage by default
- **Encryption**: TLS 1.3 for all communications
- **Access Control**: API key authentication
- **Audit Logging**: All inference requests logged
- **Compliance**: GDPR, CCPA compliance features