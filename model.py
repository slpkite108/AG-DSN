import math
from typing import NamedTuple

import torch
import torch.nn as nn
import timm


class VideoBackbone(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()

        model_name = 'vit_small_patch16_dinov3_qkvb.lvd1689m'
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0
        )
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        vit_feature_dim = 384

        self.projection = nn.Linear(vit_feature_dim, feature_dim)

    def forward(self, x):

        b, t, c, h, w = x.shape

        x_flat = x.reshape(b * t, c, h, w)

        with torch.no_grad():
            features_flat = self.feature_extractor(x_flat)

        projected_features = self.projection(features_flat)

        return projected_features.view(b, t, -1)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class TemporalState(NamedTuple):
    frame_memory: torch.Tensor
    clip_memory: torch.Tensor


class MultiStreamTALModel(nn.Module):
    def __init__(self, num_classes=4, d_model=384, nhead=4, num_encoder_layers=4,
                 num_decoder_layers=4, use_backbone=False,
                 classifier_dropout=0.0, frame_memory_len: int = 32,
                 clip_memory_len: int = 64, prediction_len: int = 32):
        super().__init__()
        if prediction_len < 1:
            raise ValueError("prediction_len must include at least the current frame.")
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_backbone = use_backbone
        self.frame_memory_len = max(frame_memory_len, 0)
        self.clip_memory_len = max(clip_memory_len, 0)
        self.prediction_len = prediction_len
        self.backbone = VideoBackbone(d_model) if use_backbone else None

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('sinusoidal_div_term', div_term, persistent=False)
        self.prediction_queries = nn.Parameter(torch.empty(1, prediction_len, d_model))
        nn.init.normal_(self.prediction_queries, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True,
        )
        self.observation_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.present_injection = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.context_encoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.anticipation_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.observation_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.frame_classifier = MLP(d_model, d_model, num_classes, classifier_dropout)
        self.clip_classifier = MLP(d_model, d_model, num_classes, classifier_dropout)

    def initial_state(self, batch_size, device=None, dtype=None):
        device = self.prediction_queries.device if device is None else device
        dtype = self.prediction_queries.dtype if dtype is None else dtype
        return TemporalState(
            torch.zeros(batch_size, self.frame_memory_len, self.d_model, device=device, dtype=dtype),
            torch.zeros(batch_size, self.clip_memory_len, self.d_model, device=device, dtype=dtype),
        )

    def _normalize_history(self, history, capacity, current):
        if capacity == 0:
            return current.new_zeros(current.size(0), 0, self.d_model)
        if history is None:
            return current.new_zeros(current.size(0), capacity, self.d_model)
        history = history.to(device=current.device, dtype=current.dtype)
        if history.size(1) >= capacity:
            return history[:, -capacity:]
        pad = current.new_zeros(current.size(0), capacity - history.size(1), self.d_model)
        return torch.cat([pad, history], dim=1)

    def _position(self, sequence):
        positions = torch.arange(sequence.size(1), device=sequence.device, dtype=sequence.dtype)
        angles = positions[:, None] * self.sinusoidal_div_term.to(sequence)
        encoding = torch.zeros_like(sequence[:1])
        encoding[0, :, 0::2] = angles.sin()
        encoding[0, :, 1::2] = angles.cos()[:, :self.d_model // 2]
        return sequence + encoding

    def _features(self, x, use_backbone):
        use_backbone = self.use_backbone if use_backbone is None else use_backbone
        if use_backbone:
            if self.backbone is None:
                raise RuntimeError("Backbone is disabled for this model instance.")
            if x.dim() != 5:
                raise ValueError("Expected video frames of shape (B, T, C, H, W).")
            return self.backbone(x)
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"Expected pooled features of shape (B, T, {self.d_model}).")
        return x

    def _encode_observations(self, features):
        batch_size, time_len, _ = features.shape
        encoded = self.observation_encoder(features.reshape(batch_size * time_len, 1, self.d_model))
        return encoded.reshape(batch_size, time_len, self.d_model)

    def _present(self, frame_memory, z):
        queries = self._position(self.prediction_queries).expand(z.size(0), -1, -1)
        return self.present_injection(tgt=queries, memory=torch.cat([self._position(frame_memory), z], dim=1))

    def _context(self, clip_memory, y):
        queries = clip_memory
        if queries.size(1) == 0:
            queries = y.new_zeros(y.size(0), 1, self.d_model)
        return self.context_encoder(tgt=queries, memory=y)

    def _anticipate(self, y, context):
        return self.anticipation_decoder(tgt=y, memory=context)

    def _next_state(self, state, z):
        recent = torch.cat([state.frame_memory, z], dim=1)
        evicted = recent[:, :1]
        long_history = torch.cat([state.clip_memory, evicted], dim=1)
        return TemporalState(
            self._normalize_history(recent, self.frame_memory_len, z).detach(),
            self._normalize_history(long_history, self.clip_memory_len, z).detach(),
        )

    def _forward_current(self, z, state):
        y = self._present(state.frame_memory, z)
        context = self._context(state.clip_memory, y)
        anticipated = self._anticipate(y, context)
        injected = self.observation_decoder(tgt=z, memory=anticipated)
        return {
            'current_logits': self.frame_classifier(injected),
            'future_logits': self.clip_classifier(anticipated),
            'state': self._next_state(state, z),
        }

    def forward(self, x, history_state=None, use_backbone=None):
        """Advance one current frame. Start each video with history_state=None."""
        features = self._features(x, use_backbone)
        if features.size(1) != 1:
            raise ValueError("forward expects one current frame. Use forward_window for observed history plus current.")
        z = self._encode_observations(features)
        state = TemporalState(
            self._normalize_history(None if history_state is None else history_state.frame_memory,
                                    self.frame_memory_len, z),
            self._normalize_history(None if history_state is None else history_state.clip_memory,
                                    self.clip_memory_len, z),
        )
        return self._forward_current(z, state)

    def forward_window(self, observed, long_history=None, use_backbone=None):
        """Use H history frames plus current, and optional older pre-injection Z."""
        features = self._features(observed, use_backbone)
        if features.size(1) != self.frame_memory_len + 1:
            raise ValueError("forward_window expects frame_memory_len history frames plus one current frame.")
        encoded = self._encode_observations(features)
        z = encoded[:, -1:]
        state = TemporalState(
            encoded[:, :-1],
            self._normalize_history(long_history, self.clip_memory_len, z),
        )
        return self._forward_current(z, state)


AGDSN = MultiStreamTALModel
