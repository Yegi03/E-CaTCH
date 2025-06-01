import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, Optional, Tuple

class TextEncoder(nn.Module):
    """BERT-based text encoder that projects to a fixed dimension."""
    
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 output_dim: int = 512,
                 freeze_bert: bool = False):
        """
        Initialize text encoder.
        
        Args:
            model_name: Name of pretrained BERT model
            output_dim: Output dimension for projection
            freeze_bert: Whether to freeze BERT parameters
        """
        super().__init__()
        
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Projection layer
        self.projection = nn.Linear(
            self.bert.config.hidden_size,  # 768 for bert-base
            output_dim
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - features: Projected features [batch_size, output_dim]
                - attention_weights: Optional attention weights
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        # Get [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Project to output dimension
        projected = self.projection(cls_embedding)  # [batch_size, output_dim]
        
        # Apply layer normalization
        features = self.layer_norm(projected)
        
        result = {'features': features}
        
        # Add attention weights if requested
        if return_attention:
            result['attention_weights'] = outputs.attentions
            
        return result
    
    def encode_text(self,
                   text: str,
                   max_length: int = 128,
                   device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convenience method to encode a single text string.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            device: Device to place tensors on
            
        Returns:
            Encoded features [1, output_dim]
        """
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device if specified
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get features
        with torch.no_grad():
            outputs = self.forward(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask']
            )
            
        return outputs['features']
    
    def encode_batch(self,
                    texts: list,
                    max_length: int = 128,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convenience method to encode a batch of text strings.
        
        Args:
            texts: List of input text strings
            max_length: Maximum sequence length
            device: Device to place tensors on
            
        Returns:
            Encoded features [batch_size, output_dim]
        """
        # Tokenize
        tokens = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device if specified
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get features
        with torch.no_grad():
            outputs = self.forward(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask']
            )
            
        return outputs['features'] 