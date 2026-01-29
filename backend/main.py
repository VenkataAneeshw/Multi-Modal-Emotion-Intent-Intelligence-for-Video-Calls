from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import io
from PIL import Image
import librosa
import asyncio
from typing import List, Dict, Optional
import time
import logging
import sys
import os

# Add parent directory to path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.vision import VisionEmotionModel
from models.audio import AudioEmotionModel
from models.text import TextIntentModel
from models.fusion import MultiModalFusion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EMOTIA API", description="Multi-Modal Emotion & Intent Intelligence API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize models (load from checkpoints in production)
vision_model = VisionEmotionModel().to(device)
audio_model = AudioEmotionModel().to(device)
text_model = TextIntentModel().to(device)
fusion_model = MultiModalFusion().to(device)

# Load trained weights (placeholder)
# vision_model.load_state_dict(torch.load('models/checkpoints/vision.pth'))
# audio_model.load_state_dict(torch.load('models/checkpoints/audio.pth'))
# text_model.load_state_dict(torch.load('models/checkpoints/text.pth'))
# fusion_model.load_state_dict(torch.load('models/checkpoints/fusion.pth'))

vision_model.eval()
audio_model.eval()
text_model.eval()
fusion_model.eval()

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
intent_labels = ['agreement', 'confusion', 'hesitation', 'confidence', 'neutral']

@app.get("/")
async def root():
    return {"message": "EMOTIA Multi-Modal Emotion & Intent Intelligence API"}

@app.post("/analyze/frame")
async def analyze_frame(
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = None
):
    """
    Analyze a single frame with optional audio and text.
    Returns emotion, intent, engagement, confidence, and modality contributions.
    """
    start_time = time.time()

    try:
        # Process image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))
        image_np = np.array(image_pil)

        # Detect faces and extract features
        faces = vision_model.detect_faces(image_np)
        if not faces:
            raise HTTPException(status_code=400, detail="No faces detected in image")

        vision_logits, vision_conf = vision_model.extract_features(faces)
        vision_features = vision_model.vit(pixel_values=torch.stack([
            vision_model.transform(face) for face in faces
        ]).to(device)).last_hidden_state[:, 0, :].mean(dim=0)  # Average across faces

        # Process audio if provided
        audio_features = None
        if audio:
            audio_data = await audio.read()
            audio_np, _ = librosa.load(io.BytesIO(audio_data), sr=16000, duration=3.0)
            audio_tensor = torch.tensor(audio_np, dtype=torch.float32).to(device)
            audio_logits, audio_stress = audio_model(audio_tensor.unsqueeze(0))
            audio_features = audio_model.wav2vec(audio_tensor.unsqueeze(0)).last_hidden_state.mean(dim=1)

        # Process text if provided
        text_features = None
        if text:
            input_ids, attention_mask = text_model.preprocess_text(text)
            input_ids = input_ids.to(device).unsqueeze(0)
            attention_mask = attention_mask.to(device).unsqueeze(0)
            intent_logits, sentiment_logits, text_conf = text_model(input_ids, attention_mask)
            text_features = text_model.bert(input_ids, attention_mask).pooler_output

        # Default features if modality missing
        if audio_features is None:
            audio_features = torch.zeros(1, 128).to(device)
        if text_features is None:
            text_features = torch.zeros(1, 768).to(device)

        # Fuse modalities
        with torch.no_grad():
            results = fusion_model(
                vision_features.unsqueeze(0),
                audio_features,
                text_features
            )

        # Convert to readable format
        emotion_probs = torch.softmax(results['emotion'], dim=1)[0].cpu().numpy()
        intent_probs = torch.softmax(results['intent'], dim=1)[0].cpu().numpy()

        response = {
            "emotion": {
                "predictions": {emotion_labels[i]: float(prob) for i, prob in enumerate(emotion_probs)},
                "dominant": emotion_labels[np.argmax(emotion_probs)]
            },
            "intent": {
                "predictions": {intent_labels[i]: float(prob) for i, prob in enumerate(intent_probs)},
                "dominant": intent_labels[np.argmax(intent_probs)]
            },
            "engagement": float(results['engagement'].cpu().numpy()),
            "confidence": float(results['confidence'].cpu().numpy()),
            "modality_contributions": {
                "vision": float(results['contributions'][0].cpu().numpy()),
                "audio": float(results['contributions'][1].cpu().numpy()),
                "text": float(results['contributions'][2].cpu().numpy())
            },
            "processing_time": time.time() - start_time
        }

        return response

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/analyze/stream")
async def analyze_stream(data: Dict):
    """
    Analyze streaming video/audio/text data.
    Expects base64 encoded frames and audio chunks.
    """
    # Placeholder for streaming analysis
    # In production, this would handle WebRTC streams
    return {"message": "Streaming analysis not yet implemented"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)