from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional
import torch
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import redis
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge
import uuid
import sys
import os

# Add parent directory to path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.advanced.advanced_fusion import AdvancedMultiModalFusion
from models.advanced.data_augmentation import AdvancedPreprocessingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('emotia_requests_total', 'Total requests', ['endpoint', 'status'])
INFERENCE_TIME = Histogram('emotia_inference_duration_seconds', 'Inference time', ['model'])
ACTIVE_CONNECTIONS = Gauge('emotia_active_websocket_connections', 'Active WebSocket connections')
MODEL_VERSIONS = Gauge('emotia_model_versions', 'Model version info', ['version', 'accuracy'])

app = FastAPI(title="EMOTIA Advanced API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Model registry for versioning
model_registry = {}
current_model_version = "v2.0.0"

# Redis for caching and session management
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            'start_time': time.time(),
            'frames_processed': 0,
            'last_activity': time.time()
        }
        ACTIVE_CONNECTIONS.inc()
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            del self.session_data[session_id]
            ACTIVE_CONNECTIONS.dec()
            logger.info(f"WebSocket disconnected: {session_id}")

    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# Load models
def load_models():
    """Load and version models"""
    global model_registry

    # Load advanced fusion model
    advanced_model = AdvancedMultiModalFusion().to(device)
    # In production, load from checkpoint
    # advanced_model.load_state_dict(torch.load('models/checkpoints/advanced_fusion.pth'))
    advanced_model.eval()

    model_registry[current_model_version] = {
        'model': advanced_model,
        'accuracy': 0.85,  # Placeholder
        'created_at': time.time(),
        'preprocessing': AdvancedPreprocessingPipeline()
    }

    MODEL_VERSIONS.labels(version=current_model_version, accuracy=0.85).set(1)
    logger.info(f"Loaded model version: {current_model_version}")

load_models()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    load_models()

@app.get("/")
async def root():
    return {
        "message": "EMOTIA Advanced Multi-Modal Emotion & Intent Intelligence API v2.0",
        "version": current_model_version,
        "endpoints": [
            "/analyze/frame",
            "/analyze/stream",
            "/ws/analyze/{session_id}",
            "/models/versions",
            "/health",
            "/metrics"
        ]
    }

@app.get("/models/versions")
async def get_model_versions():
    """Get available model versions"""
    versions = {}
    for version, info in model_registry.items():
        versions[version] = {
            'accuracy': info['accuracy'],
            'created_at': info['created_at']
        }
    return versions

@app.post("/analyze/frame")
async def analyze_frame(
    image_data: bytes = None,
    audio_data: bytes = None,
    text: str = None,
    model_version: str = current_model_version
):
    """Advanced frame analysis with caching and metrics"""
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint='/analyze/frame', status='started').inc()

    try:
        # Get model
        if model_version not in model_registry:
            raise HTTPException(status_code=400, detail=f"Model version {model_version} not found")

        model_info = model_registry[model_version]
        model = model_info['model']
        preprocessor = model_info['preprocessing']

        # Create cache key
        cache_key = f"{hash(image_data or '')}:{hash(audio_data or '')}:{hash(text or '')}:{model_version}"
        cached_result = redis_client.get(cache_key)

        if cached_result:
            REQUEST_COUNT.labels(endpoint='/analyze/frame', status='cached').inc()
            return json.loads(cached_result)

        # Process inputs
        vision_input = None
        audio_input = None
        text_input = None

        if image_data:
            # Decode and preprocess image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            vision_input = preprocessor.preprocess_face(image)
            if vision_input is not None:
                vision_input = vision_input.unsqueeze(0).to(device)

        if audio_data:
            # Decode and preprocess audio
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            audio_input = preprocessor.preprocess_audio(audio_np)
            if audio_input is not None:
                audio_input = audio_input.unsqueeze(0).to(device)

        if text:
            # Preprocess text
            text_input = preprocessor.preprocess_text(text, model.clip_tokenizer if hasattr(model, 'clip_tokenizer') else None)
            text_input = {k: v.to(device) for k, v in text_input.items()}

        # Run inference
        with torch.no_grad():
            with INFERENCE_TIME.labels(model=model_version).time():
                outputs = model(
                    vision_input=vision_input,
                    audio_input=audio_input,
                    text_input=text_input
                )

        # Process outputs
        result = {
            'emotion': {
                'probabilities': torch.softmax(outputs['emotion_logits'], dim=1)[0].cpu().numpy().tolist(),
                'dominant': torch.argmax(outputs['emotion_logits'], dim=1)[0].item()
            },
            'intent': {
                'probabilities': torch.softmax(outputs['intent_logits'], dim=1)[0].cpu().numpy().tolist(),
                'dominant': torch.argmax(outputs['intent_logits'], dim=1)[0].item()
            },
            'engagement': {
                'mean': outputs['engagement_mean'][0].item(),
                'uncertainty': outputs['engagement_var'][0].item()
            },
            'confidence': {
                'mean': outputs['confidence_mean'][0].item(),
                'uncertainty': outputs['confidence_var'][0].item()
            },
            'modality_importance': outputs['modality_importance'][0].cpu().numpy().tolist(),
            'processing_time': time.time() - start_time,
            'model_version': model_version
        }

        # Cache result
        redis_client.setex(cache_key, 3600, json.dumps(result))  # Cache for 1 hour

        REQUEST_COUNT.labels(endpoint='/analyze/frame', status='success').inc()
        return result

    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/analyze/frame', status='error').inc()
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.websocket("/ws/analyze/{session_id}")
async def websocket_analyze(websocket: WebSocket, session_id: str):
    """Real-time streaming analysis via WebSocket"""
    await manager.connect(websocket, session_id)

    try:
        while True:
            # Receive data
            data = await websocket.receive_json()

            # Process in background
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                process_streaming_data,
                data,
                session_id
            )

            # Send result
            await manager.send_personal_message(json.dumps(result), session_id)

            # Update session stats
            manager.session_data[session_id]['frames_processed'] += 1
            manager.session_data[session_id]['last_activity'] = time.time()

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {str(e)}")
        await manager.send_personal_message(json.dumps({"error": str(e)}), session_id)
        manager.disconnect(session_id)

def process_streaming_data(data, session_id):
    """Process streaming data in background thread"""
    # Similar to analyze_frame but optimized for streaming
    model_info = model_registry[current_model_version]
    model = model_info['model']

    # Process data (simplified for demo)
    result = {
        'session_id': session_id,
        'timestamp': time.time(),
        'emotion': {'dominant': 0},  # Placeholder
        'engagement': 0.5
    }

    return result

@app.get("/health")
async def health_check():
    """Advanced health check with system metrics"""
    return {
        "status": "healthy",
        "version": current_model_version,
        "device": str(device),
        "active_connections": len(manager.active_connections),
        "model_versions": list(model_registry.keys()),
        "redis_connected": redis_client.ping() if redis_client else False,
        "timestamp": time.time()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        prom.generate_latest(),
        media_type="text/plain"
    )

@app.post("/models/deploy/{version}")
async def deploy_model(version: str, background_tasks: BackgroundTasks):
    """Deploy a new model version (admin endpoint)"""
    if version not in model_registry:
        raise HTTPException(status_code=404, detail=f"Model version {version} not found")

    global current_model_version
    current_model_version = version

    # Background task to update metrics
    background_tasks.add_task(update_model_metrics, version)

    return {"message": f"Deployed model version {version}"}

def update_model_metrics(version):
    """Update Prometheus metrics for new model version"""
    info = model_registry[version]
    MODEL_VERSIONS.labels(version=version, accuracy=info['accuracy']).set(1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple workers for better performance
        loop="uvloop"  # Faster event loop
    )