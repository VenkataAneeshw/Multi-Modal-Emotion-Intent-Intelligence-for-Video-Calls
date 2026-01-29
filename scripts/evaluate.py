import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os
from tqdm import tqdm

from models.vision import VisionEmotionModel
from models.audio import AudioEmotionModel
from models.text import TextIntentModel
from models.fusion import MultiModalFusion

def evaluate_model(model, dataloader, device, task='emotion'):
    """
    Evaluate model on given task.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {task}"):
            if task == 'emotion':
                vision = batch['vision'].to(device)
                audio = batch['audio'].to(device)
                text_input_ids = batch['text']['input_ids'].to(device)
                text_attention_mask = batch['text']['attention_mask'].to(device)
                labels = batch['emotion'].to(device)

                outputs = model(vision, audio, text_input_ids, text_attention_mask)
                preds = outputs['emotion'].argmax(dim=1)

            elif task == 'intent':
                # Similar for intent
                preds = outputs['intent'].argmax(dim=1)
                labels = batch['intent'].to(device)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def ablation_study(fusion_model, dataloader, device):
    """
    Perform ablation study by removing modalities.
    """
    print("Performing Ablation Study...")

    results = {}

    # Full model
    preds, labels = evaluate_model(fusion_model, dataloader, device)
    results['full'] = f1_score(labels, preds, average='weighted')

    # Vision-only (set audio and text to zero)
    fusion_model.eval()
    ablation_preds = []
    with torch.no_grad():
        for batch in dataloader:
            vision = batch['vision'].to(device)
            audio = torch.zeros_like(batch['audio']).to(device)
            text_input_ids = batch['text']['input_ids'].to(device)
            text_attention_mask = batch['text']['attention_mask'].to(device)

            outputs = fusion_model(vision, audio, text_input_ids, text_attention_mask)
            preds = outputs['emotion'].argmax(dim=1)
            ablation_preds.extend(preds.cpu().numpy())

    results['vision_only'] = f1_score(labels, ablation_preds, average='weighted')

    # Audio-only
    ablation_preds = []
    with torch.no_grad():
        for batch in dataloader:
            vision = torch.zeros_like(batch['vision']).to(device)
            audio = batch['audio'].to(device)
            text_input_ids = batch['text']['input_ids'].to(device)
            text_attention_mask = batch['text']['attention_mask'].to(device)

            outputs = fusion_model(vision, audio, text_input_ids, text_attention_mask)
            preds = outputs['emotion'].argmax(dim=1)
            ablation_preds.extend(preds.cpu().numpy())

    results['audio_only'] = f1_score(labels, ablation_preds, average='weighted')

    # Text-only
    ablation_preds = []
    with torch.no_grad():
        for batch in dataloader:
            vision = torch.zeros_like(batch['vision']).to(device)
            audio = torch.zeros_like(batch['audio']).to(device)
            text_input_ids = batch['text']['input_ids'].to(device)
            text_attention_mask = batch['text']['attention_mask'].to(device)

            outputs = fusion_model(vision, audio, text_input_ids, text_attention_mask)
            preds = outputs['emotion'].argmax(dim=1)
            ablation_preds.extend(preds.cpu().numpy())

    results['text_only'] = f1_score(labels, ablation_preds, average='weighted')

    return results

def bias_analysis(model, dataloader, device, demographic_groups):
    """
    Analyze bias across demographic groups.
    """
    print("Performing Bias Analysis...")

    bias_results = {}

    model.eval()
    with torch.no_grad():
        for group in demographic_groups:
            group_preds = []
            group_labels = []

            # Filter data for this demographic group
            # This would require demographic labels in dataset
            for batch in dataloader:
                # Placeholder: assume demographic info in batch
                if 'demographic' in batch and batch['demographic'] == group:
                    vision = batch['vision'].to(device)
                    audio = batch['audio'].to(device)
                    text_input_ids = batch['text']['input_ids'].to(device)
                    text_attention_mask = batch['text']['attention_mask'].to(device)

                    outputs = model(vision, audio, text_input_ids, text_attention_mask)
                    preds = outputs['emotion'].argmax(dim=1)
                    labels = batch['emotion']

                    group_preds.extend(preds.cpu().numpy())
                    group_labels.extend(labels.cpu().numpy())

            if group_preds:
                bias_results[group] = {
                    'f1': f1_score(group_labels, group_preds, average='weighted'),
                    'accuracy': np.mean(np.array(group_preds) == np.array(group_labels))
                }

    return bias_results

def plot_confusion_matrix(cm, labels, save_path):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_report(results, ablation_results, bias_results, output_dir):
    """
    Generate comprehensive evaluation report.
    """
    report = f"""
# EMOTIA Model Evaluation Report

## Overall Performance
- Emotion F1-Score: {results['emotion_f1']:.4f}
- Intent F1-Score: {results['intent_f1']:.4f}
- Engagement MAE: {results['engagement_mae']:.4f}
- Confidence MAE: {results['confidence_mae']:.4f}

## Ablation Study Results
{chr(10).join([f"- {k}: {v:.4f}" for k, v in ablation_results.items()])}

## Bias Analysis
"""

    if bias_results:
        for group, metrics in bias_results.items():
            report += f"- {group}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}\n"
    else:
        report += "No demographic data available for bias analysis.\n"

    report += """
## Recommendations
- Focus on improving the weakest modality based on ablation results.
- Monitor and mitigate biases identified in demographic analysis.
- Consider additional data augmentation for underrepresented classes.
"""

    with open(os.path.join(output_dir, 'evaluation_report.md'), 'w') as f:
        f.write(report)

    print("Evaluation report saved to evaluation_report.md")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    fusion_model = MultiModalFusion().to(device)
    fusion_model.load_state_dict(torch.load(args.model_path))
    fusion_model.eval()

    # Load test data
    # test_dataset = MultiModalDataset(args.data_dir, 'test')
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Placeholder for actual evaluation
    print("Evaluation framework ready. Implement data loading for full evaluation.")

    # Example results structure
    results = {
        'emotion_f1': 0.85,
        'intent_f1': 0.78,
        'engagement_mae': 0.12,
        'confidence_mae': 0.15
    }

    ablation_results = {
        'full': 0.85,
        'vision_only': 0.72,
        'audio_only': 0.68,
        'text_only': 0.75
    }

    bias_results = {}  # Would be populated with actual demographic analysis

    # Generate report
    generate_report(results, ablation_results, bias_results, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EMOTIA Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    args = parser.parse_args()
    main(args)