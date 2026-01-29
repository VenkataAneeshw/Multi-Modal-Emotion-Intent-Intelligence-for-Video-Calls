import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import re

class TextIntentModel(nn.Module):
    """
    Transformer-based model for text intent and sentiment analysis.
    Fine-tuned BERT for conversational intent detection.
    """
    def __init__(self, num_intents=5, pretrained=True):
        super().__init__()
        self.num_intents = num_intents

        # Load pre-trained BERT
        if pretrained:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            from transformers import BertConfig
            config = BertConfig()
            self.bert = BertModel(config)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Freeze base layers
        for param in self.bert.parameters():
            param.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        # Intent classification head
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_intents)
        )

        # Sentiment/emotion head
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7 emotions
        )

        # Confidence/hesitation detection
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        """
        input_ids: tokenized text (B, seq_len)
        attention_mask: attention mask (B, seq_len)
        Returns: intent_logits, sentiment_logits, confidence
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token

        intent_logits = self.intent_classifier(pooled_output)
        sentiment_logits = self.sentiment_head(pooled_output)
        confidence = self.confidence_head(pooled_output)

        return intent_logits, sentiment_logits, confidence.squeeze()

    def preprocess_text(self, text):
        """
        Preprocess and tokenize text input.
        """
        # Clean text
        text = self.clean_text(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

    def clean_text(self, text):
        """
        Clean and normalize text.
        """
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.lower()

    def detect_hesitation_phrases(self, text):
        """
        Detect phrases indicating hesitation or confusion.
        """
        hesitation_keywords = [
            'um', 'uh', 'like', 'you know', 'sort of', 'kind of',
            'i think', 'maybe', 'perhaps', 'i\'m not sure'
        ]

        text_lower = text.lower()
        hesitation_score = sum(1 for keyword in hesitation_keywords if keyword in text_lower)

        return min(hesitation_score / 5.0, 1.0)  # Normalize to 0-1

    def extract_intent_features(self, text):
        """
        Extract intent-related features from text.
        """
        with torch.no_grad():
            input_ids, attention_mask = self.preprocess_text(text)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            intent_logits, sentiment_logits, confidence = self.forward(input_ids, attention_mask)

        return {
            'intent_logits': intent_logits,
            'sentiment_logits': sentiment_logits,
            'confidence': confidence,
            'hesitation_score': self.detect_hesitation_phrases(text)
        }