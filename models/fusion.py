import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing vision, audio, and text features.
    """
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key_value):
        """
        query: (B, seq_len_q, embed_dim)
        key_value: (B, seq_len_kv, embed_dim)
        """
        # Project to attention space
        q = self.query_proj(query)
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)

        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(q, k, v)

        # Residual connection and normalization
        output = self.norm(query + self.dropout(attn_output))

        return output, attn_weights

class TemporalTransformer(nn.Module):
    """
    Temporal transformer for modeling sequences across time windows.
    """
    def __init__(self, embed_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, seq_len, embed_dim) - sequence of fused features over time
        """
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)

class MultiModalFusion(nn.Module):
    """
    Complete fusion network combining vision, audio, text with temporal modeling.
    """
    def __init__(self, vision_dim=768, audio_dim=128, text_dim=768, embed_dim=256,
                 num_emotions=7, num_intents=5):
        super().__init__()
        self.embed_dim = embed_dim

        # Modality projectors
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # Cross-modal attention layers
        self.vision_to_audio_attn = CrossModalAttention(embed_dim)
        self.audio_to_text_attn = CrossModalAttention(embed_dim)
        self.text_to_vision_attn = CrossModalAttention(embed_dim)

        # Temporal modeling
        self.temporal_transformer = TemporalTransformer(embed_dim)

        # Dynamic modality weighting
        self.modality_weights = nn.Parameter(torch.ones(3))  # vision, audio, text

        # Output heads
        self.emotion_classifier = nn.Linear(embed_dim, num_emotions)
        self.intent_classifier = nn.Linear(embed_dim, num_intents)
        self.engagement_regressor = nn.Linear(embed_dim, 1)
        self.confidence_regressor = nn.Linear(embed_dim, 1)

        # Modality contribution estimator
        self.contribution_estimator = nn.Linear(embed_dim * 3, 3)  # weights for each modality

    def forward(self, vision_features, audio_features, text_features, temporal_seq=False):
        """
        vision_features: (B, vision_dim) or (B, T, vision_dim)
        audio_features: (B, audio_dim) or (B, T, audio_dim)
        text_features: (B, text_dim) or (B, T, text_dim)
        temporal_seq: whether inputs are temporal sequences
        """
        # Project to common embedding space
        v_proj = self.vision_proj(vision_features)  # (B, embed_dim) or (B, T, embed_dim)
        a_proj = self.audio_proj(audio_features)
        t_proj = self.text_proj(text_features)

        if temporal_seq:
            # Handle temporal sequences
            B, T, _ = v_proj.shape

            # Reshape for attention: (B*T, 1, embed_dim)
            v_flat = v_proj.view(B*T, 1, -1)
            a_flat = a_proj.view(B*T, 1, -1)
            t_flat = t_proj.view(B*T, 1, -1)

            # Cross-modal attention
            v_attn, _ = self.vision_to_audio_attn(v_flat, a_flat)
            a_attn, _ = self.audio_to_text_attn(a_flat, t_flat)
            t_attn, _ = self.text_to_vision_attn(t_flat, v_flat)

            # Combine attended features
            fused = (v_attn + a_attn + t_attn) / 3  # (B*T, 1, embed_dim)

            # Reshape back to temporal: (B, T, embed_dim)
            fused = fused.view(B, T, -1)

            # Temporal transformer
            temporal_out = self.temporal_transformer(fused)  # (B, T, embed_dim)

            # Pool temporal dimension (take last timestep or mean)
            pooled = temporal_out[:, -1, :]  # (B, embed_dim)

        else:
            # Single timestep fusion
            # Cross-modal attention
            v_attn, _ = self.vision_to_audio_attn(v_proj.unsqueeze(1), a_proj.unsqueeze(1))
            a_attn, _ = self.audio_to_text_attn(a_proj.unsqueeze(1), t_proj.unsqueeze(1))
            t_attn, _ = self.text_to_vision_attn(t_proj.unsqueeze(1), v_proj.unsqueeze(1))

            # Weighted fusion
            weights = F.softmax(self.modality_weights, dim=0)
            fused = weights[0] * v_attn.squeeze(1) + \
                   weights[1] * a_attn.squeeze(1) + \
                   weights[2] * t_attn.squeeze(1)

            pooled = fused

        # Output predictions
        emotion_logits = self.emotion_classifier(pooled)
        intent_logits = self.intent_classifier(pooled)
        engagement = torch.sigmoid(self.engagement_regressor(pooled))
        confidence = torch.sigmoid(self.confidence_regressor(pooled))

        # Modality contributions
        contributions = torch.softmax(self.contribution_estimator(
            torch.cat([v_proj.mean(dim=-1 if temporal_seq else 0, keepdim=True),
                      a_proj.mean(dim=-1 if temporal_seq else 0, keepdim=True),
                      t_proj.mean(dim=-1 if temporal_seq else 0, keepdim=True)], dim=-1)
        ), dim=-1)

        return {
            'emotion': emotion_logits,
            'intent': intent_logits,
            'engagement': engagement.squeeze(),
            'confidence': confidence.squeeze(),
            'contributions': contributions.squeeze()
        }

    def get_modality_weights(self):
        """
        Return normalized modality weights for explainability.
        """
        return F.softmax(self.modality_weights, dim=0)