import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import math

class AdvancedMultiModalFusion(nn.Module):
    """
    Advanced multi-modal fusion using CLIP-inspired architecture
    with contrastive learning and improved attention mechanisms.
    """
    def __init__(self, embed_dim=512, num_emotions=7, num_intents=5, use_clip=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_clip = use_clip

        if use_clip:
            # Use CLIP for multi-modal understanding
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            # Freeze CLIP backbone
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Advanced modality projectors with layer normalization
        self.vision_projector = nn.Sequential(
            nn.Linear(768, embed_dim),  # CLIP vision dim
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.audio_projector = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.text_projector = nn.Sequential(
            nn.Linear(768, embed_dim),  # CLIP text dim
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Multi-head cross-attention with different attention patterns
        self.vision_to_audio_attn = nn.MultiheadAttention(embed_dim, 8, dropout=0.1, batch_first=True)
        self.audio_to_text_attn = nn.MultiheadAttention(embed_dim, 8, dropout=0.1, batch_first=True)
        self.text_to_vision_attn = nn.MultiheadAttention(embed_dim, 8, dropout=0.1, batch_first=True)

        # Self-attention for each modality
        self.vision_self_attn = nn.MultiheadAttention(embed_dim, 8, dropout=0.1, batch_first=True)
        self.audio_self_attn = nn.MultiheadAttention(embed_dim, 8, dropout=0.1, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(embed_dim, 8, dropout=0.1, batch_first=True)

        # Temporal modeling with position encoding
        self.max_seq_len = 50
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, embed_dim))
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=6
        )

        # Contrastive learning temperature
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # Advanced output heads with uncertainty estimation
        self.emotion_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_emotions)
        )

        self.intent_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_intents)
        )

        self.engagement_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2)  # Mean and variance for uncertainty
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2)  # Mean and variance for uncertainty
        )

        # Modality importance scoring
        self.modality_scorer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
            nn.Softmax(dim=-1)
        )

    def encode_modalities(self, vision_input=None, audio_input=None, text_input=None):
        """Encode each modality to common embedding space"""
        embeddings = {}

        if vision_input is not None:
            if self.use_clip:
                # Use CLIP vision encoder
                vision_outputs = self.clip_model.vision_model(vision_input)
                vision_emb = vision_outputs.pooler_output
            else:
                vision_emb = vision_input
            embeddings['vision'] = self.vision_projector(vision_emb)

        if audio_input is not None:
            embeddings['audio'] = self.audio_projector(audio_input)

        if text_input is not None:
            if self.use_clip:
                # Use CLIP text encoder
                text_outputs = self.clip_model.text_model(**text_input)
                text_emb = text_outputs.pooler_output
            else:
                text_emb = text_input
            embeddings['text'] = self.text_projector(text_emb)

        return embeddings

    def cross_modal_attention(self, embeddings):
        """Perform cross-modal attention between available modalities"""
        modalities = list(embeddings.keys())
        attended_features = {}

        # Self-attention for each modality first
        for mod in modalities:
            feat = embeddings[mod].unsqueeze(1)  # Add sequence dim
            attended, _ = getattr(self, f"{mod}_self_attn")(feat, feat, feat)
            attended_features[mod] = attended.squeeze(1)

        # Cross-modal attention
        if 'vision' in modalities and 'audio' in modalities:
            v2a, _ = self.vision_to_audio_attn(
                attended_features['vision'].unsqueeze(1),
                attended_features['audio'].unsqueeze(1),
                attended_features['audio'].unsqueeze(1)
            )
            attended_features['vision'] = attended_features['vision'] + v2a.squeeze(1)

        if 'audio' in modalities and 'text' in modalities:
            a2t, _ = self.audio_to_text_attn(
                attended_features['audio'].unsqueeze(1),
                attended_features['text'].unsqueeze(1),
                attended_features['text'].unsqueeze(1)
            )
            attended_features['audio'] = attended_features['audio'] + a2t.squeeze(1)

        if 'text' in modalities and 'vision' in modalities:
            t2v, _ = self.text_to_vision_attn(
                attended_features['text'].unsqueeze(1),
                attended_features['vision'].unsqueeze(1),
                attended_features['vision'].unsqueeze(1)
            )
            attended_features['text'] = attended_features['text'] + t2v.squeeze(1)

        return attended_features

    def temporal_modeling(self, attended_features, seq_len=None):
        """Apply temporal transformer if sequence data is available"""
        if seq_len is None or seq_len == 1:
            # Single timestep - just average
            combined = torch.stack(list(attended_features.values())).mean(dim=0)
            return combined.unsqueeze(0)

        # Multi-timestep temporal modeling
        # Concatenate modalities across time
        temporal_seq = []
        for t in range(seq_len):
            timestep_features = []
            for mod_features in attended_features.values():
                if mod_features.dim() > 2:  # Has time dimension
                    timestep_features.append(mod_features[:, t])
                else:
                    timestep_features.append(mod_features)
            temporal_seq.append(torch.stack(timestep_features).mean(dim=0))

        temporal_input = torch.stack(temporal_seq, dim=1)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        seq_len_actual = min(temporal_input.size(1), self.max_seq_len)
        temporal_input = temporal_input + self.temporal_pos_embed[:, :seq_len_actual]

        # Apply temporal transformer
        temporal_output = self.temporal_transformer(temporal_input)

        return temporal_output

    def compute_modality_importance(self, embeddings):
        """Compute importance scores for each modality"""
        modality_features = []
        for mod in ['vision', 'audio', 'text']:
            if mod in embeddings:
                modality_features.append(embeddings[mod])
            else:
                modality_features.append(torch.zeros_like(list(embeddings.values())[0]))

        combined = torch.cat(modality_features, dim=-1)
        importance_scores = self.modality_scorer(combined)
        return importance_scores

    def forward(self, vision_input=None, audio_input=None, text_input=None, seq_len=None):
        """
        Forward pass with advanced fusion
        """
        # Encode modalities
        embeddings = self.encode_modalities(vision_input, audio_input, text_input)

        if not embeddings:
            raise ValueError("At least one modality must be provided")

        # Cross-modal attention
        attended_features = self.cross_modal_attention(embeddings)

        # Temporal modeling
        temporal_output = self.temporal_modeling(attended_features, seq_len)

        # Global representation (use last timestep or average)
        if seq_len and seq_len > 1:
            global_repr = temporal_output[:, -1]  # Last timestep
        else:
            global_repr = temporal_output.squeeze(0)

        # Compute modality importance
        importance_scores = self.compute_modality_importance(embeddings)

        # Generate predictions with uncertainty
        emotion_logits = self.emotion_head(global_repr)
        intent_logits = self.intent_head(global_repr)

        engagement_params = self.engagement_head(global_repr)
        engagement_mean = torch.sigmoid(engagement_params[:, 0])
        engagement_var = F.softplus(engagement_params[:, 1])

        confidence_params = self.confidence_head(global_repr)
        confidence_mean = torch.sigmoid(confidence_params[:, 0])
        confidence_var = F.softplus(confidence_params[:, 1])

        return {
            'emotion_logits': emotion_logits,
            'intent_logits': intent_logits,
            'engagement_mean': engagement_mean,
            'engagement_var': engagement_var,
            'confidence_mean': confidence_mean,
            'confidence_var': confidence_var,
            'modality_importance': importance_scores,
            'embeddings': embeddings,
            'temporal_features': temporal_output
        }

    def contrastive_loss(self, embeddings, temperature=0.07):
        """Compute contrastive loss for multi-modal alignment"""
        if len(embeddings) < 2:
            return torch.tensor(0.0)

        # Normalize embeddings
        normalized_embs = {k: F.normalize(v, dim=-1) for k, v in embeddings.items()}

        total_loss = 0
        count = 0

        modalities = list(normalized_embs.keys())
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j:
                    # Contrastive loss between mod1 and mod2
                    logits = torch.matmul(normalized_embs[mod1], normalized_embs[mod2].T) / temperature
                    labels = torch.arange(logits.size(0)).to(logits.device)
                    loss = F.cross_entropy(logits, labels)
                    total_loss += loss
                    count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0)