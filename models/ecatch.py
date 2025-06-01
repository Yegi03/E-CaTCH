import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional, Tuple

from .encoders.text_encoder import TextEncoder
from .encoders.image_encoder import ImageEncoder
from .attention.intra_modal import IntraModalAttention
from .attention.cross_modal import CrossModalAttention
from .fusion.soft_gating import SoftGatingFusion
from .temporal.trend_lstm import TrendLSTM

class ECatchModel(nn.Module):
    """E-CaTCH: Event-Centric Temporal Cross-modal Hybrid model."""
    
    def __init__(self,
                 text_encoder: TextEncoder,
                 image_encoder: ImageEncoder,
                 embed_dim: int = 768,
                 num_heads: int = 8,
                 lstm_hidden_dim: int = 512,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.1,
                 alpha: float = 0.1):  # Decay coefficient for temporal weighting
        """
        Initialize E-CaTCH model.
        
        Args:
            text_encoder: Text encoder module
            image_encoder: Image encoder module
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            lstm_hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            alpha: Decay coefficient for temporal weighting
        """
        super().__init__()
        
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.embed_dim = embed_dim
        
        # Intra-modal attention
        self.text_attention = IntraModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.image_attention = IntraModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Soft gating fusion
        self.fusion = SoftGatingFusion(
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Temporal modeling
        self.trend_lstm = TrendLSTM(
            input_dim=embed_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
            alpha=alpha
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2)  # Binary classification
        )
        
    def forward(self,
                text_inputs: Dict[str, torch.Tensor],
                image_inputs: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text_inputs: Dictionary containing text encoder inputs
            image_inputs: Image inputs [batch_size, channels, height, width]
            timestamps: Optional timestamps for temporal weighting [batch_size, seq_len]
            mask: Optional mask for padding [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - logits: Classification logits [batch_size, 2]
                - text_attention: Optional text attention weights
                - image_attention: Optional image attention weights
                - cross_attention: Optional cross-modal attention weights
                - gates: Optional fusion gating values
                - momentum: Current momentum signal
                - temporal_weights: Optional temporal weights
        """
        # Encode text and images
        text_features = self.text_encoder(**text_inputs)
        image_features = self.image_encoder(image_inputs)
        
        # Apply intra-modal attention
        text_refined = self.text_attention(
            text_features,
            return_attention=return_attention
        )
        
        image_refined = self.image_attention(
            image_features,
            return_attention=return_attention
        )
        
        # Apply cross-modal attention
        cross_output = self.cross_attention(
            text_refined['output'] if return_attention else text_refined,
            image_refined['output'] if return_attention else image_refined,
            return_attention=return_attention
        )
        
        # Fuse cross-modal features
        fused = self.fusion(
            cross_output['text_to_image'],
            cross_output['image_to_text'],
            return_gates=return_attention
        )
        
        # Apply temporal modeling with temporal weighting
        temporal = self.trend_lstm(
            fused['output'],
            timestamps=timestamps,
            mask=mask,
            return_hidden=return_attention
        )
        
        # Classification
        logits = self.classifier(temporal['output'].mean(dim=1))
        
        result = {
            'logits': logits,
            'momentum': temporal['momentum']
        }
        
        # Add attention weights if requested
        if return_attention:
            result.update({
                'text_attention': text_refined['attention'],
                'image_attention': image_refined['attention'],
                'cross_attention': cross_output['attention'],
                'gates': fused['gates'],
                'hidden': temporal['hidden']
            })
            
        # Add temporal weights if they were computed
        if timestamps is not None:
            result['temporal_weights'] = temporal['temporal_weights']
            
        return result
    
    def get_attention_weights(self,
                            text_inputs: Dict[str, torch.Tensor],
                            image_inputs: torch.Tensor,
                            timestamps: Optional[torch.Tensor] = None,
                            mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Get attention weights without computing output.
        
        Args:
            text_inputs: Dictionary containing text encoder inputs
            image_inputs: Image inputs [batch_size, channels, height, width]
            timestamps: Optional timestamps for temporal weighting [batch_size, seq_len]
            mask: Optional mask for padding [batch_size, seq_len]
            
        Returns:
            Dictionary containing attention weights
        """
        # Encode text and images
        text_features = self.text_encoder(**text_inputs)
        image_features = self.image_encoder(image_inputs)
        
        # Get intra-modal attention weights
        text_attention = self.text_attention.get_attention_weights(text_features)
        image_attention = self.image_attention.get_attention_weights(image_features)
        
        # Get cross-modal attention weights
        cross_attention = self.cross_attention.get_attention_weights(
            text_features,
            image_features
        )
        
        # Get fusion gates
        gates = self.fusion.get_gates(
            cross_attention['text_to_image'],
            cross_attention['image_to_text']
        )
        
        result = {
            'text_attention': text_attention,
            'image_attention': image_attention,
            'cross_attention': cross_attention,
            'gates': gates
        }
        
        # Add temporal weights if timestamps are provided
        if timestamps is not None:
            result['temporal_weights'] = self.trend_lstm.compute_temporal_weights(timestamps)
            
        return result 