import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import argparse
import os
from tqdm import tqdm

from models.vision import VisionEmotionModel
from models.audio import AudioEmotionModel
from models.text import TextIntentModel
from models.fusion import MultiModalFusion

class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal training with aligned vision, audio, text data.
    """
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        # Load preprocessed data
        # This would load aligned samples from FER-2013, RAVDESS, IEMOCAP, etc.
        self.samples = self.load_samples()

    def load_samples(self):
        # Placeholder for loading aligned multi-modal data
        # In practice, this would load from processed HDF5 or pickle files
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'vision': sample['vision'],  # face image or features
            'audio': sample['audio'],    # audio waveform or features
            'text': sample['text'],      # tokenized text
            'emotion': sample['emotion'],  # emotion label
            'intent': sample['intent'],    # intent label
            'engagement': sample['engagement'],  # engagement score
            'confidence': sample['confidence']   # confidence score
        }

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    emotion_preds, emotion_labels = [], []
    intent_preds, intent_labels = [], []

    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        vision = batch['vision'].to(device)
        audio = batch['audio'].to(device)
        text_input_ids = batch['text']['input_ids'].to(device)
        text_attention_mask = batch['text']['attention_mask'].to(device)

        emotion_labels_batch = batch['emotion'].to(device)
        intent_labels_batch = batch['intent'].to(device)
        engagement_labels = batch['engagement'].to(device)
        confidence_labels = batch['confidence'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(vision, audio, text_input_ids, text_attention_mask)

        # Compute losses
        emotion_loss = criterion['emotion'](outputs['emotion'], emotion_labels_batch)
        intent_loss = criterion['intent'](outputs['intent'], intent_labels_batch)
        engagement_loss = criterion['engagement'](outputs['engagement'], engagement_labels)
        confidence_loss = criterion['confidence'](outputs['confidence'], confidence_labels)

        # Weighted multi-task loss
        loss = (emotion_loss + intent_loss + engagement_loss + confidence_loss) / 4

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Collect predictions for metrics
        emotion_preds.extend(outputs['emotion'].argmax(dim=1).cpu().numpy())
        emotion_labels.extend(emotion_labels_batch.cpu().numpy())
        intent_preds.extend(outputs['intent'].argmax(dim=1).cpu().numpy())
        intent_labels.extend(intent_labels_batch.cpu().numpy())

    # Compute metrics
    emotion_acc = accuracy_score(emotion_labels, emotion_preds)
    emotion_f1 = f1_score(emotion_labels, emotion_preds, average='weighted')
    intent_acc = accuracy_score(intent_labels, intent_preds)
    intent_f1 = f1_score(intent_labels, intent_preds, average='weighted')

    return total_loss / len(dataloader), emotion_acc, emotion_f1, intent_acc, intent_f1

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    emotion_preds, emotion_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            vision = batch['vision'].to(device)
            audio = batch['audio'].to(device)
            text_input_ids = batch['text']['input_ids'].to(device)
            text_attention_mask = batch['text']['attention_mask'].to(device)

            emotion_labels_batch = batch['emotion'].to(device)
            intent_labels_batch = batch['intent'].to(device)
            engagement_labels = batch['engagement'].to(device)
            confidence_labels = batch['confidence'].to(device)

            outputs = model(vision, audio, text_input_ids, text_attention_mask)

            emotion_loss = criterion['emotion'](outputs['emotion'], emotion_labels_batch)
            intent_loss = criterion['intent'](outputs['intent'], intent_labels_batch)
            engagement_loss = criterion['engagement'](outputs['engagement'], engagement_labels)
            confidence_loss = criterion['confidence'](outputs['confidence'], confidence_labels)

            loss = (emotion_loss + intent_loss + engagement_loss + confidence_loss) / 4
            total_loss += loss.item()

            emotion_preds.extend(outputs['emotion'].argmax(dim=1).cpu().numpy())
            emotion_labels.extend(emotion_labels_batch.cpu().numpy())

    emotion_acc = accuracy_score(emotion_labels, emotion_preds)
    emotion_f1 = f1_score(emotion_labels, emotion_preds, average='weighted')

    return total_loss / len(dataloader), emotion_acc, emotion_f1

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize models
    vision_model = VisionEmotionModel(num_emotions=args.num_emotions)
    audio_model = AudioEmotionModel(num_emotions=args.num_emotions)
    text_model = TextIntentModel(num_intents=args.num_intents)

    # For simplicity, train fusion model with pre-extracted features
    # In practice, you'd train end-to-end
    fusion_model = MultiModalFusion(
        vision_dim=768,  # ViT hidden size
        audio_dim=128,   # Audio feature dim
        text_dim=768,    # BERT hidden size
        num_emotions=args.num_emotions,
        num_intents=args.num_intents
    ).to(device)

    # Loss functions
    criterion = {
        'emotion': nn.CrossEntropyLoss(),
        'intent': nn.CrossEntropyLoss(),
        'engagement': nn.MSELoss(),
        'confidence': nn.MSELoss()
    }

    optimizer = optim.Adam(fusion_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Datasets
    train_dataset = MultiModalDataset(args.data_dir, 'train')
    val_dataset = MultiModalDataset(args.data_dir, 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, train_f1, intent_acc, intent_f1 = train_epoch(
            fusion_model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, val_f1 = validate_epoch(fusion_model, val_loader, criterion, device)

        print(".4f")
        print(".4f")

        scheduler.step()

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(fusion_model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))

    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EMOTIA Multi-Modal Model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--output_dir', type=str, default='./models/checkpoints', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_emotions', type=int, default=7, help='Number of emotion classes')
    parser.add_argument('--num_intents', type=int, default=5, help='Number of intent classes')

    args = parser.parse_args()
    main(args)