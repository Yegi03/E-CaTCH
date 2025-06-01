import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class IntraModalAttention(nn.Module):
    """Multi-head self-attention for intra-modal refinement."""
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize intra-modal attention.
        
        Args:
            embed_dim: Input embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - output: Refined features [batch_size, seq_len, embed_dim]
                - attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        # Layer normalization
        x_norm = self.layer_norm(x)
        
        # Linear projections
        q = self.q_proj(x_norm)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x_norm)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x_norm)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum
        context = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        context = context.view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        # Residual connection
        output = output + x
        
        result = {'output': output}
        
        # Add attention weights if requested
        if return_attention:
            result['attention_weights'] = attention_weights
            
        return result
    
    def get_attention_weights(self,
                            x: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights without computing output.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # Layer normalization
        x_norm = self.layer_norm(x)
        
        # Linear projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights 