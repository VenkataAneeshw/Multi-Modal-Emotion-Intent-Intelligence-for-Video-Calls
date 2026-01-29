import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
import librosa
import numpy as np

class AudioEmotionModel(nn.Module):
    """
    CNN + Transformer for audio emotion recognition.
    Uses Wav2Vec2 backbone for feature extraction.
    """
    def __init__(self, num_emotions=7, pretrained=True):
        super().__init__()
        self.num_emotions = num_emotions

        # Load pre-trained Wav2Vec2
        if pretrained:
            self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        else:
            config = Wav2Vec2Config()
            self.wav2vec = Wav2Vec2Model(config)

        # Freeze base layers
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        hidden_size = self.wav2vec.config.hidden_size

        # CNN for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Transformer for sequence modeling
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512),
            num_layers=4
        )

        # Emotion classification
        self.emotion_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions)
        )

        # Stress/confidence estimation
        self.stress_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input_values):
        """
        input_values: batch of audio waveforms (B, T)
        Returns: emotion_logits, stress_score
        """
        # Extract features with Wav2Vec2
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state  # (B, T, hidden_size)

        # Transpose for CNN (B, hidden_size, T)
        hidden_states = hidden_states.transpose(1, 2)

        # CNN feature extraction
        cnn_features = self.cnn(hidden_states).squeeze(-1)  # (B, 128)

        # Add sequence dimension for transformer
        cnn_features = cnn_features.unsqueeze(1)  # (B, 1, 128)

        # Transformer
        transformer_out = self.transformer(cnn_features)  # (B, 1, 128)
        pooled_features = transformer_out.mean(dim=1)  # (B, 128)

        emotion_logits = self.emotion_classifier(pooled_features)
        stress_score = self.stress_head(pooled_features)

        return emotion_logits, stress_score.squeeze()

    def preprocess_audio(self, audio_path, sample_rate=16000, duration=3.0):
        """
        Load and preprocess audio file.
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)

        # Pad/truncate to fixed length
        target_length = int(sample_rate * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        return torch.tensor(audio, dtype=torch.float32)

    def extract_prosody_features(self, audio):
        """
        Extract additional prosody features (pitch, rhythm, etc.)
        """
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=audio.numpy(), sr=16000)
        pitch = np.mean(pitches[pitches > 0])

        # RMS energy
        rms = librosa.feature.rms(y=audio.numpy())[0].mean()

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio.numpy())[0].mean()

        return torch.tensor([pitch, rms, zcr], dtype=torch.float32)