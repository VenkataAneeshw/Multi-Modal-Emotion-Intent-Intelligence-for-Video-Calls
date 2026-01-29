import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import torchaudio
import torchaudio.transforms as AT
import numpy as np
import random
from PIL import Image
import librosa

class AdvancedDataAugmentation:
    """
    Advanced data augmentation pipeline for multi-modal training
    """

    def __init__(self):
        # Vision augmentations
        self.vision_transforms = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1),
            T.RandomApply([T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.2),
            T.RandomHorizontalFlip(p=0.1),
            T.RandomApply([T.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))], p=0.1),
        ])

        # Audio augmentations
        self.audio_sample_rate = 16000

    def augment_vision(self, image):
        """
        Apply advanced vision augmentations
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Apply standard augmentations
        augmented = self.vision_transforms(image)

        # Additional advanced augmentations
        if random.random() < 0.1:
            # Simulate different lighting conditions
            augmented = TF.adjust_gamma(augmented, random.uniform(0.8, 1.2))

        if random.random() < 0.1:
            # Add noise
            img_array = np.array(augmented)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            augmented = Image.fromarray(img_array)

        return augmented

    def augment_audio(self, audio, sample_rate):
        """
        Apply advanced audio augmentations
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        augmented_audios = [audio]

        # Time stretching
        if random.random() < 0.3:
            rate = random.uniform(0.8, 1.2)
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            augmented_audios.append(stretched)

        # Pitch shifting
        if random.random() < 0.3:
            steps = random.randint(-2, 2)
            pitched = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=steps)
            augmented_audios.append(pitched)

        # Add background noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.01, len(audio))
            noisy = audio + noise
            augmented_audios.append(noisy)

        # Volume perturbation
        if random.random() < 0.3:
            volume_factor = random.uniform(0.7, 1.3)
            volume_aug = audio * volume_factor
            augmented_audios.append(volume_aug)

        # Random cropping/padding
        if random.random() < 0.2:
            target_length = int(sample_rate * random.uniform(2.5, 4.0))
            if len(audio) > target_length:
                start = random.randint(0, len(audio) - target_length)
                cropped = audio[start:start + target_length]
            else:
                padding = target_length - len(audio)
                cropped = np.pad(audio, (0, padding), 'constant')
            augmented_audios.append(cropped)

        # Select one augmentation or original
        selected = random.choice(augmented_audios)

        # Ensure consistent length (3 seconds)
        target_length = sample_rate * 3
        if len(selected) > target_length:
            selected = selected[:target_length]
        elif len(selected) < target_length:
            selected = np.pad(selected, (0, target_length - len(selected)), 'constant')

        return torch.tensor(selected, dtype=torch.float32)

    def augment_text(self, text, tokenizer):
        """
        Apply text augmentations
        """
        augmented_texts = [text]

        # Synonym replacement (simplified)
        if random.random() < 0.2:
            words = text.split()
            if len(words) > 3:
                # Simple synonym replacement (would need a proper synonym dictionary)
                idx = random.randint(0, len(words) - 1)
                # For demo, just shuffle some words
                if random.random() < 0.5:
                    random.shuffle(words)
                    synonym_aug = ' '.join(words)
                    augmented_texts.append(synonym_aug)

        # Backtranslation augmentation would go here (requires translation models)

        # Random deletion
        if random.random() < 0.1:
            words = text.split()
            if len(words) > 3:
                keep_prob = 0.9
                kept_words = [w for w in words if random.random() < keep_prob]
                if kept_words:
                    deletion_aug = ' '.join(kept_words)
                    augmented_texts.append(deletion_aug)

        selected_text = random.choice(augmented_texts)
        return selected_text

class AdvancedPreprocessingPipeline:
    """
    Advanced preprocessing pipeline with quality checks and normalization
    """

    def __init__(self, target_face_size=(224, 224), target_audio_length=3.0):
        self.target_face_size = target_face_size
        self.target_audio_length = target_audio_length
        self.sample_rate = 16000

        # Quality thresholds
        self.min_face_confidence = 0.7
        self.min_audio_snr = 10.0  # dB

    def preprocess_face(self, face_image, bbox=None, landmarks=None):
        """
        Advanced face preprocessing with alignment and quality checks
        """
        # Quality check
        if not self._check_face_quality(face_image):
            return None

        # Convert to PIL if needed
        if isinstance(face_image, np.ndarray):
            face_image = Image.fromarray(face_image)

        # Face alignment using landmarks if available
        if landmarks is not None:
            face_image = self._align_face(face_image, landmarks)

        # Resize and normalize
        face_image = face_image.resize(self.target_face_size, Image.BILINEAR)

        # Convert to tensor
        face_tensor = TF.to_tensor(face_image)

        # Normalize (ImageNet stats for CLIP compatibility)
        normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                              std=[0.26862954, 0.26130258, 0.27577711])
        face_tensor = normalize(face_tensor)

        return face_tensor

    def preprocess_audio(self, audio_path_or_array, sample_rate=None):
        """
        Advanced audio preprocessing with quality checks
        """
        # Load audio
        if isinstance(audio_path_or_array, str):
            audio, sr = librosa.load(audio_path_or_array, sr=self.sample_rate)
        else:
            audio = audio_path_or_array
            sr = sample_rate or self.sample_rate

        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Quality check
        if not self._check_audio_quality(audio):
            return None

        # Voice activity detection (simple energy-based)
        audio = self._voice_activity_detection(audio)

        # Normalize audio
        audio = self._normalize_audio(audio)

        # Ensure consistent length
        target_samples = int(self.sample_rate * self.target_audio_length)
        if len(audio) > target_samples:
            # Random crop
            start = random.randint(0, len(audio) - target_samples)
            audio = audio[start:start + target_samples]
        elif len(audio) < target_samples:
            # Pad with zeros
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        return torch.tensor(audio, dtype=torch.float32)

    def preprocess_text(self, text, tokenizer, max_length=128):
        """
        Advanced text preprocessing
        """
        # Clean text
        text = self._clean_text(text)

        # Tokenize
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoding

    def _check_face_quality(self, face_image):
        """
        Check face image quality
        """
        if isinstance(face_image, np.ndarray):
            # Check resolution
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                return False

            # Check brightness
            brightness = np.mean(face_image)
            if brightness < 30 or brightness > 225:
                return False

            # Check contrast
            contrast = np.std(face_image)
            if contrast < 10:
                return False

        return True

    def _check_audio_quality(self, audio):
        """
        Check audio quality using SNR
        """
        # Simple SNR calculation
        signal_power = np.mean(audio ** 2)
        noise_power = np.var(audio - np.convolve(audio, np.ones(100)/100, mode='same'))
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

        return snr >= self.min_audio_snr

    def _align_face(self, face_image, landmarks):
        """
        Align face using facial landmarks
        """
        # Simplified alignment - in practice would use proper face alignment
        # For now, just return the image
        return face_image

    def _voice_activity_detection(self, audio, threshold=0.01):
        """
        Simple voice activity detection
        """
        # Calculate energy
        energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]

        # Find segments above threshold
        active_segments = energy > threshold

        if np.any(active_segments):
            # Keep only active segments
            active_indices = np.where(active_segments)[0]
            start_idx = active_indices[0] * 512
            end_idx = (active_indices[-1] + 1) * 512
            return audio[start_idx:end_idx]

        return audio

    def _normalize_audio(self, audio):
        """
        Normalize audio amplitude
        """
        # Peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio

    def _clean_text(self, text):
        """
        Clean and normalize text
        """
        import re

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.lower()