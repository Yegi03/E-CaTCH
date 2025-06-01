import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class SoftGatingFusion(nn.Module):
    """Soft-gated fusion mechanism for combining cross-modal features."""
    
    def __init__(self,
                 embed_dim: int,
                 gate_dim: int = 256,
                 dropout: float = 0.1):
        """
        Initialize soft gating fusion.
        
        Args:
            embed_dim: Input embedding dimension
            gate_dim: Dimension of gating vector
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.gate_dim = gate_dim
        
        # Concatenation projection
        self.concat_proj = nn.Linear(2 * embed_dim, gate_dim)
        
        # Gating projection
        self.gate_proj = nn.Linear(gate_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                text_to_image: torch.Tensor,
                image_to_text: torch.Tensor,
                return_gates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text_to_image: Text->image attention output [batch_size, seq_len, embed_dim]
            image_to_text: Image->text attention output [batch_size, seq_len, embed_dim]
            return_gates: Whether to return gating values
            
        Returns:
            Dictionary containing:
                - output: Fused features [batch_size, seq_len, embed_dim]
                - gates: Optional gating values
        """
        # Concatenate features
        concat_features = torch.cat([text_to_image, image_to_text], dim=-1)
        
        # Project concatenated features
        concat_proj = self.concat_proj(concat_features)
        
        # Compute gating values
        gates = torch.sigmoid(self.gate_proj(concat_proj))
        
        # Apply gating
        gated_features = gates * text_to_image + (1 - gates) * image_to_text
        
        # Project output
        output = self.out_proj(gated_features)
        output = self.dropout(output)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        result = {'output': output}
        
        # Add gating values if requested
        if return_gates:
            result['gates'] = gates
            
        return result
    
    def get_gates(self,
                 text_to_image: torch.Tensor,
                 image_to_text: torch.Tensor) -> torch.Tensor:
        """
        Get gating values without computing output.
        
        Args:
            text_to_image: Text->image attention output [batch_size, seq_len, embed_dim]
            image_to_text: Image->text attention output [batch_size, seq_len, embed_dim]
            
        Returns:
            Gating values [batch_size, seq_len, embed_dim]
        """
        # Concatenate features
        concat_features = torch.cat([text_to_image, image_to_text], dim=-1)
        
        # Project concatenated features
        concat_proj = self.concat_proj(concat_features)
        
        # Compute gating values
        gates = torch.sigmoid(self.gate_proj(concat_proj))
        
        return gates 