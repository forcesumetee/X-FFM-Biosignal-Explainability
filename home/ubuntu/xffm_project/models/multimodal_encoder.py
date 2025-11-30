"""
Multimodal Encoder for X-FFM
Author: Sumetee Jirapattarasakul

This module implements a multimodal encoder for biosignals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SignalEncoder(nn.Module):
    """Encoder for a single biosignal modality"""
    
    def __init__(
        self,
        signal_length: int,
        in_channels: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.signal_length = signal_length
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # 1D Convolutional layers for feature extraction
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = hidden_dim * (2 ** i)
            layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            ])
            current_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output length after convolutions
        self.output_length = signal_length // (2 ** num_layers)
        self.output_channels = hidden_dim * (2 ** (num_layers - 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input signal [batch_size, in_channels, signal_length]
        
        Returns:
            Encoded features [batch_size, output_channels, output_length]
        """
        return self.conv_layers(x)


class MultimodalEncoder(nn.Module):
    """
    Multimodal encoder that processes multiple biosignal modalities
    and fuses them using cross-modal attention
    """
    
    def __init__(
        self,
        modality_configs: Dict[str, Dict],
        fusion_dim: int = 256,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        self.modality_names = list(modality_configs.keys())
        self.fusion_dim = fusion_dim
        
        # Create encoder for each modality
        self.encoders = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        
        for modality, config in modality_configs.items():
            encoder = SignalEncoder(
                signal_length=config['signal_length'],
                in_channels=config.get('in_channels', 1),
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('num_layers', 3)
            )
            self.encoders[modality] = encoder
            
            # Project to fusion dimension
            proj_input_dim = encoder.output_channels * encoder.output_length
            self.projections[modality] = nn.Linear(proj_input_dim, fusion_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(
        self,
        signals: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            signals: Dictionary of input signals
                     {modality: [batch_size, in_channels, signal_length]}
            return_attention: Whether to return attention weights
        
        Returns:
            fused_features: [batch_size, fusion_dim]
            attention_weights: [batch_size, num_modalities, num_modalities] (optional)
        """
        batch_size = next(iter(signals.values())).size(0)
        
        # Encode each modality
        encoded_features = []
        for modality in self.modality_names:
            if modality in signals:
                # Encode
                features = self.encoders[modality](signals[modality])
                # Flatten
                features = features.view(batch_size, -1)
                # Project
                features = self.projections[modality](features)
                encoded_features.append(features)
        
        # Stack features: [batch_size, num_modalities, fusion_dim]
        encoded_features = torch.stack(encoded_features, dim=1)
        
        # Apply cross-modal attention
        attended_features, attention_weights = self.cross_attention(
            query=encoded_features,
            key=encoded_features,
            value=encoded_features,
            need_weights=return_attention
        )
        
        # Apply layer normalization and residual connection
        attended_features = self.layer_norm(attended_features + encoded_features)
        
        # Global average pooling across modalities
        fused_features = attended_features.mean(dim=1)  # [batch_size, fusion_dim]
        
        if return_attention:
            return fused_features, attention_weights
        else:
            return fused_features, None


def create_multimodal_encoder(
    modality_configs: Dict[str, Dict],
    fusion_dim: int = 256
) -> MultimodalEncoder:
    """Factory function to create multimodal encoder"""
    return MultimodalEncoder(
        modality_configs=modality_configs,
        fusion_dim=fusion_dim
    )
