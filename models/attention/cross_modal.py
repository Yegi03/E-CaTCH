import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention between text and image features."""
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize cross-modal attention.
        
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
        
        # Query projections
        self.q_text_proj = nn.Linear(embed_dim, embed_dim)
        self.q_image_proj = nn.Linear(embed_dim, embed_dim)
        
        # Key-value projections
        self.kv_text_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.kv_image_proj = nn.Linear(embed_dim, 2 * embed_dim)
        
        # Output projections
        self.text_out_proj = nn.Linear(embed_dim, embed_dim)
        self.image_out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization
        self.text_norm = nn.LayerNorm(embed_dim)
        self.image_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                text_features: torch.Tensor,
                image_features: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None,
                image_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text_features: Text features [batch_size, text_len, embed_dim]
            image_features: Image features [batch_size, image_len, embed_dim]
            text_mask: Optional text attention mask [batch_size, text_len]
            image_mask: Optional image attention mask [batch_size, image_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - text_to_image: Text->image attention output
                - image_to_text: Image->text attention output
                - attention_weights: Optional attention weights
        """
        batch_size = text_features.size(0)
        
        # Layer normalization
        text_norm = self.text_norm(text_features)
        image_norm = self.image_norm(image_features)
        
        # Text->Image attention
        text_to_image = self._cross_attention(
            query=self.q_text_proj(text_norm),
            key_value=self.kv_image_proj(image_norm),
            mask=image_mask,
            return_attention=return_attention
        )
        
        # Image->Text attention
        image_to_text = self._cross_attention(
            query=self.q_image_proj(image_norm),
            key_value=self.kv_text_proj(text_norm),
            mask=text_mask,
            return_attention=return_attention
        )
        
        # Project outputs
        text_to_image = self.text_out_proj(text_to_image['output'])
        image_to_text = self.image_out_proj(image_to_text['output'])
        
        # Apply dropout
        text_to_image = self.dropout(text_to_image)
        image_to_text = self.dropout(image_to_text)
        
        # Add residual connections
        text_to_image = text_to_image + text_features
        image_to_text = image_to_text + image_features
        
        result = {
            'text_to_image': text_to_image,
            'image_to_text': image_to_text
        }
        
        # Add attention weights if requested
        if return_attention:
            result['attention_weights'] = {
                'text_to_image': text_to_image['attention_weights'],
                'image_to_text': image_to_text['attention_weights']
            }
            
        return result
    
    def _cross_attention(self,
                        query: torch.Tensor,
                        key_value: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute cross-attention between query and key-value pairs.
        
        Args:
            query: Query tensor [batch_size, query_len, embed_dim]
            key_value: Key-value tensor [batch_size, kv_len, 2*embed_dim]
            mask: Optional attention mask [batch_size, kv_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - output: Attention output
                - attention_weights: Optional attention weights
        """
        batch_size, query_len, _ = query.size()
        kv_len = key_value.size(1)
        
        # Split key and value
        key, value = key_value.chunk(2, dim=-1)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, query_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, kv_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, kv_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        query = query.transpose(1, 2)  # [batch_size, num_heads, query_len, head_dim]
        key = key.transpose(1, 2)  # [batch_size, num_heads, kv_len, head_dim]
        value = value.transpose(1, 2)  # [batch_size, num_heads, kv_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, kv_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum
        context = torch.matmul(attention_weights, value)  # [batch_size, num_heads, query_len, head_dim]
        
        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous()  # [batch_size, query_len, num_heads, head_dim]
        context = context.view(batch_size, query_len, self.embed_dim)
        
        result = {'output': context}
        
        # Add attention weights if requested
        if return_attention:
            result['attention_weights'] = attention_weights
            
        return result 