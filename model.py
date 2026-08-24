import math
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

        x_flat = x.view(b * t, c, h, w)

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

class MultiStreamTALModel(nn.Module):

    def __init__(self, num_classes=4, d_model=384, nhead=4, num_encoder_layers=4,
                 num_decoder_layers=4, use_backbone=False,
                 classifier_dropout=0.0, clip_memory_len: int = 64,
                 clip_context_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_backbone = use_backbone

        self.backbone = VideoBackbone(feature_dim=d_model) if self.use_backbone else None

        self.max_seq_len = 1024
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('sinusoidal_div_term', div_term, persistent=False)
        pos_table = self._build_sinusoidal_table(
            self.max_seq_len,
            self.sinusoidal_div_term,
            device=self.sinusoidal_div_term.device,
            dtype=self.sinusoidal_div_term.dtype,
        )
        self.register_buffer('positional_encoding', pos_table, persistent=False)

        self.echo_compressor = nn.Linear(2*d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )

        self.frame_dropout = nn.Dropout(p=0.2)
        self.frame_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.frame_transformer_1 = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.classifier_dropout = classifier_dropout
        self.frame_classifier = MLP(d_model, d_model, num_classes, dropout=self.classifier_dropout)

        self.clip_dropout = nn.Dropout(p=0.2)
        self.clip_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.clip_transformer_1 = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.clip_classifier = MLP(d_model, d_model, num_classes, dropout=self.classifier_dropout)
        self.frame_memory_len = max(clip_memory_len, 0)
        self.clip_context_len = max(clip_context_len, 0)

    def _normalize_history_tensor(self, tensor, max_len, batch_size, device, dtype):
        if max_len <= 0:
            return None
        if tensor is None:
            return torch.zeros(batch_size, max_len, self.d_model, device=device, dtype=dtype)
        tensor = tensor.to(device=device, dtype=dtype)
        cur_len = tensor.size(1)
        if cur_len > max_len:
            return tensor[:, -max_len:, :]
        if cur_len < max_len:
            pad = torch.zeros(batch_size, max_len - cur_len, self.d_model, device=device, dtype=dtype)
            return torch.cat([pad, tensor], dim=1)
        return tensor

    def _normalize_echo_memory(self, tensor, batch_size, time_len, device, dtype):
        if time_len <= 0:
            return torch.zeros(batch_size, 0, self.d_model, device=device, dtype=dtype)
        if tensor is None:
            return torch.zeros(batch_size, time_len, self.d_model, device=device, dtype=dtype)
        tensor = tensor.to(device=device, dtype=dtype)
        cur_len = tensor.size(1)
        if cur_len > time_len:
            return tensor[:, -time_len:, :]
        if cur_len < time_len:
            pad = torch.zeros(batch_size, time_len - cur_len, self.d_model, device=device, dtype=dtype)
            return torch.cat([pad, tensor], dim=1)
        return tensor

    def _build_sinusoidal_table(self, length, div_term, device=None, dtype=torch.float32):
        device = device or div_term.device
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        sine = torch.sin(position * div_term)
        cosine = torch.cos(position * div_term)
        table = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        table[:, 0::2] = sine
        table[:, 1::2] = cosine
        return table

    def _ensure_positional_capacity(self, seq_len):
        if seq_len <= self.max_seq_len:
            return
        new_len = max(seq_len, self.max_seq_len * 2)
        new_table = self._build_sinusoidal_table(
            new_len,
            self.sinusoidal_div_term,
            device=self.sinusoidal_div_term.device,
            dtype=self.sinusoidal_div_term.dtype,
        )
        self.positional_encoding = new_table
        self.max_seq_len = new_len

    def _sinusoidal_from_positions(self, positions):

        div_term = self.sinusoidal_div_term.to(device=positions.device, dtype=positions.dtype)
        angles = positions.unsqueeze(-1) * div_term
        sin_term = torch.sin(angles)
        cos_term = torch.cos(angles)
        embed = torch.zeros(*positions.shape, self.d_model, device=positions.device, dtype=positions.dtype)
        embed[..., 0::2] = sin_term
        embed[..., 1::2] = cos_term
        return embed

    def _resolve_time_positions(self, extra_kwargs, batch_size, time_len, device):
        base_dtype = self.sinusoidal_div_term.dtype
        time_positions = extra_kwargs.get('frame_timestamps')
        if time_positions is None:
            time_positions = extra_kwargs.get('frame_indices')
        if time_positions is None:
            return torch.arange(time_len, device=device, dtype=base_dtype).unsqueeze(0).expand(batch_size, -1)

        time_positions = time_positions.to(device=device, dtype=base_dtype)
        if time_positions.dim() == 1:
            if time_positions.size(0) != time_len:
                raise ValueError("frame_timestamps length must match sequence length.")
            return time_positions.unsqueeze(0).expand(batch_size, -1)
        if (
            time_positions.dim() == 2
            and time_positions.size(1) == time_len
            and time_positions.size(0) == 1
            and batch_size > 1
        ):
            return time_positions.expand(batch_size, -1)
        if time_positions.shape != (batch_size, time_len):
            raise ValueError("frame_timestamps/frame_indices must have shape (T,) or (B, T) matching the input batch.")
        return time_positions

    def _apply_spatiotemporal_encoding(self, features, batch_size, time_len, device, dtype, extra_kwargs):
        pos_encoding = self.positional_encoding[:time_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        pos_encoding = pos_encoding.to(device=device, dtype=dtype)
        time_positions = self._resolve_time_positions(extra_kwargs, batch_size, time_len, device)
        time_encoding = self._sinusoidal_from_positions(time_positions).to(dtype=dtype)
        return features + pos_encoding

    def _prepare_memory_buffers(self, history_state, batch_size, time_len, device, dtype):
        history_state = history_state or {}
        frame_memory = None
        clip_to_frame_memory = None
        echo_memory = None
        if isinstance(history_state, dict):
            frame_memory = history_state.get('frame_memory')
            clip_to_frame_memory = history_state.get('frame_context')
            if clip_to_frame_memory is None:
                clip_to_frame_memory = history_state.get('clip_to_frame_memory')
            echo_memory = history_state.get('echo_memory')
        frame_memory = self._normalize_history_tensor(frame_memory, self.frame_memory_len, batch_size, device, dtype)
        clip_to_frame_memory = self._normalize_history_tensor(clip_to_frame_memory, self.clip_context_len, batch_size, device, dtype)
        echo_memory = self._normalize_echo_memory(echo_memory, batch_size, time_len, device, dtype)
        return frame_memory, clip_to_frame_memory, echo_memory

    def _ensure_branch_memory(self, memory_tensor, fallback_len, batch_size, device, dtype):
        if memory_tensor is not None and memory_tensor.size(1) > 0:
            return memory_tensor
        effective_len = fallback_len if fallback_len > 0 else 1
        return torch.zeros(batch_size, effective_len, self.d_model, device=device, dtype=dtype)

    def forward(self, x, history_state=None, use_backbone=None, **kwargs):
        use_backbone = self.use_backbone if use_backbone is None else use_backbone

        if use_backbone and self.backbone is None:
            raise RuntimeError("Backbone is disabled for this model instance.")
        if use_backbone:
            if x.dim() != 5:
                raise ValueError("Expected input shape (B, T, C, H, W) when use_backbone is True.")
            b = x.shape[0]
            features = self.backbone(x)
        else:
            if x.dim() != 3 or x.size(-1) != self.d_model:
                raise ValueError(f"Expected precomputed features of shape (B, T, {self.d_model}) when use_backbone is False.")
            b = x.shape[0]
            features = x

        t = features.shape[1]
        self._ensure_positional_capacity(t)
        frame_memory, clip_to_frame_memory, echo_memory = self._prepare_memory_buffers(
            history_state,
            batch_size=b,
            time_len=t,
            device=features.device,
            dtype=features.dtype
        )
        features_with_pos = self._apply_spatiotemporal_encoding(
            features,
            batch_size=b,
            time_len=t,
            device=features.device,
            dtype=features.dtype,
            extra_kwargs=kwargs,
        )

        combined_frame_input = torch.cat([features_with_pos, echo_memory], dim=-1)
        frame_input = self.echo_compressor(combined_frame_input)
        memory_for_frame = self._ensure_branch_memory(
            clip_to_frame_memory,
            self.clip_context_len,
            batch_size=b,
            device=features.device,
            dtype=features.dtype,
        )
        memory_for_frame = self.frame_dropout(memory_for_frame)
        frame_features = self.frame_transformer(tgt=frame_input, memory=memory_for_frame)
        frame_features = self.frame_transformer_1(frame_features)
        current_logits = self.frame_classifier(frame_features)

        memory_for_clip = self._ensure_branch_memory(
            frame_memory,
            self.frame_memory_len,
            batch_size=b,
            device=features.device,
            dtype=features.dtype,
        )
        clip_memory_len = memory_for_clip.size(1)
        if clip_memory_len > 0:
            if echo_memory.size(1) >= clip_memory_len:
                echo_slice = echo_memory[:, -clip_memory_len:, :]
            else:
                pad_len = clip_memory_len - echo_memory.size(1)
                pad = torch.zeros(b, pad_len, self.d_model, device=features.device, dtype=features.dtype)
                echo_slice = torch.cat([pad, echo_memory], dim=1)
            combined_clip_memory = torch.cat([memory_for_clip, echo_slice], dim=-1)
            memory_for_clip = self.echo_compressor(combined_clip_memory)
            memory_for_clip = self.clip_dropout(memory_for_clip)
        clip_features = self.clip_transformer(tgt=features_with_pos, memory=memory_for_clip)
        clip_features = self.clip_transformer_1(clip_features)
        future_logits = self.clip_classifier(clip_features)

        new_state = {}
        if self.frame_memory_len > 0:
            new_state['frame_memory'] = frame_features[:, -self.frame_memory_len:, :].detach()
        if self.clip_context_len > 0:
            new_state['frame_context'] = clip_features[:, -self.clip_context_len:, :].detach()
        new_state['echo_memory'] = features_with_pos.detach()
        if not new_state:
            new_state = None

        return {
            'current_logits': current_logits,
            'future_logits': future_logits,
            'state': new_state
        }

AGDSN = MultiStreamTALModel
