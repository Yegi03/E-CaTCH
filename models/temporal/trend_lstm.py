import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class TrendLSTM(nn.Module):
    """LSTM-based temporal modeling with momentum tracking."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 alpha: float = 0.1):  # Decay coefficient for temporal weighting
        """
        Initialize trend LSTM.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            alpha: Decay coefficient for temporal weighting
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.alpha = alpha
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        self.out_proj = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim,
            input_dim
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize momentum buffer
        self.register_buffer('momentum', torch.zeros(input_dim))
        
    def compute_temporal_weights(self,
                               timestamps: torch.Tensor,
                               current_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute temporal weights λᵢ using exponential decay.
        
        Args:
            timestamps: Tensor of timestamps [batch_size, seq_len]
            current_time: Optional reference time for decay calculation
            
        Returns:
            Temporal weights [batch_size, seq_len]
        """
        if current_time is None:
            current_time = timestamps.max(dim=1, keepdim=True)[0]
            
        # Compute time differences in hours
        time_diff = (current_time - timestamps) / 3600  # Convert to hours
        
        # Apply exponential decay: λᵢ = exp(-α * Δt)
        weights = torch.exp(-self.alpha * time_diff)
        
        # Normalize weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        return weights
        
    def forward(self,
                x: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            timestamps: Optional timestamps for temporal weighting [batch_size, seq_len]
            mask: Optional mask for padding [batch_size, seq_len]
            return_hidden: Whether to return hidden states
            
        Returns:
            Dictionary containing:
                - output: Processed features [batch_size, seq_len, input_dim]
                - hidden: Optional hidden states
                - momentum: Updated momentum signal
                - temporal_weights: Optional temporal weights
        """
        # Apply temporal weighting if timestamps are provided
        if timestamps is not None:
            temporal_weights = self.compute_temporal_weights(timestamps)
            x = x * temporal_weights.unsqueeze(-1)
        
        # Pack sequence if mask is provided
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_x)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            output, (hidden, cell) = self.lstm(x)
        
        # Project output
        output = self.out_proj(output)
        output = self.dropout(output)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Update momentum using exponential moving average
        with torch.no_grad():
            self.momentum = 0.9 * self.momentum + 0.1 * output.mean(dim=(0, 1))
        
        result = {
            'output': output,
            'momentum': self.momentum
        }
        
        # Add hidden states if requested
        if return_hidden:
            result['hidden'] = hidden
            
        # Add temporal weights if they were computed
        if timestamps is not None:
            result['temporal_weights'] = temporal_weights
            
        return result
    
    def get_hidden_states(self,
                         x: torch.Tensor,
                         timestamps: Optional[torch.Tensor] = None,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get hidden states without computing output.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            timestamps: Optional timestamps for temporal weighting [batch_size, seq_len]
            mask: Optional mask for padding [batch_size, seq_len]
            
        Returns:
            Hidden states [num_layers * num_directions, batch_size, hidden_dim]
        """
        # Apply temporal weighting if timestamps are provided
        if timestamps is not None:
            temporal_weights = self.compute_temporal_weights(timestamps)
            x = x * temporal_weights.unsqueeze(-1)
            
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed_x)
        else:
            _, (hidden, _) = self.lstm(x)
            
        return hidden 