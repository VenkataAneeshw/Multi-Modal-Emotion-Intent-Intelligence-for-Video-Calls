import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision import transforms
import cv2
import numpy as np

class VisionEmotionModel(nn.Module):
    """
    Vision Transformer for facial emotion recognition.
    Fine-tuned on FER-2013/AffectNet datasets.
    """
    def __init__(self, num_emotions=7, pretrained=True):
        super().__init__()
        self.num_emotions = num_emotions

        # Load pre-trained ViT
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            config = ViTConfig()
            self.vit = ViTModel(config)

        # Freeze base layers if fine-tuning
        for param in self.vit.parameters():
            param.requires_grad = False

        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_emotions)
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 0-1 confidence
        )

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        """
        x: batch of images (B, C, H, W) or list of face crops
        Returns: emotion_logits, confidence
        """
        if isinstance(x, list):
            # Handle list of face images
            batch = torch.stack([self.transform(img) for img in x])
        else:
            batch = x

        outputs = self.vit(pixel_values=batch)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        emotion_logits = self.emotion_classifier(cls_token)
        confidence = self.confidence_head(cls_token)

        return emotion_logits, confidence.squeeze()

    def detect_faces(self, frame):
        """
        Detect faces in a video frame using OpenCV.
        Returns list of face crops.
        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        face_crops = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                face_crops.append(face)

        return face_crops

    def extract_features(self, faces):
        """
        Extract emotion features from detected faces.
        """
        if not faces:
            return None, None

        with torch.no_grad():
            emotion_logits, confidence = self.forward(faces)

        return emotion_logits, confidence