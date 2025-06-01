import json
import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime
from pathlib import Path

def load_jsonl(path: str) -> List[Dict]:
    """Load a .json or .jsonl file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def normalize_timestamp(ts: str) -> float:
    """Normalize a timestamp to float (for decay weighting)."""
    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()

def compute_temporal_weights(timestamps: List[str], alpha: float = 0.1) -> np.ndarray:
    """
    Compute temporal decay weights for posts in a window.
    
    Args:
        timestamps: List of timestamps
        alpha: Decay coefficient controlling weight distribution
    
    Returns:
        Array of weights normalized to sum to 1
    """
    # Convert timestamps to float
    float_ts = np.array([normalize_timestamp(ts) for ts in timestamps])
    
    # Compute weights using exponential decay
    t_max = float_ts.max()
    weights = np.exp(-alpha * (t_max - float_ts))
    
    # Normalize weights to sum to 1
    return weights / weights.sum()

def compute_semantic_shift(current_window: torch.Tensor, 
                         previous_window: torch.Tensor) -> torch.Tensor:
    """
    Compute semantic shift between consecutive windows.
    
    Args:
        current_window: Current window representation
        previous_window: Previous window representation
    
    Returns:
        Semantic shift vector
    """
    return current_window - previous_window

def compute_temporal_momentum(semantic_shift: torch.Tensor,
                            previous_momentum: torch.Tensor,
                            beta: float = 0.9) -> torch.Tensor:
    """
    Compute temporal momentum signal.
    
    Args:
        semantic_shift: Current semantic shift
        previous_momentum: Previous momentum value
        beta: Smoothing coefficient
    
    Returns:
        Updated momentum signal
    """
    current_magnitude = torch.norm(semantic_shift, p=2)
    return beta * previous_momentum + (1 - beta) * current_magnitude

def cross_modal_attention(text_features: torch.Tensor,
                         image_features: torch.Tensor,
                         num_heads: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform bidirectional cross-modal attention.
    
    Args:
        text_features: Text modality features
        image_features: Image modality features
        num_heads: Number of attention heads
    
    Returns:
        Tuple of (text->image attention, image->text attention)
    """
    # Compute attention scores
    text_to_image = torch.matmul(text_features, image_features.transpose(-2, -1))
    image_to_text = torch.matmul(image_features, text_features.transpose(-2, -1))
    
    # Scale attention scores
    d_k = text_features.size(-1)
    text_to_image = text_to_image / np.sqrt(d_k)
    image_to_text = image_to_text / np.sqrt(d_k)
    
    # Apply softmax
    text_to_image_weights = torch.softmax(text_to_image, dim=-1)
    image_to_text_weights = torch.softmax(image_to_text, dim=-1)
    
    # Compute attended representations
    text_to_image_attended = torch.matmul(text_to_image_weights, image_features)
    image_to_text_attended = torch.matmul(image_to_text_weights, text_features)
    
    return text_to_image_attended, image_to_text_attended

def soft_gate_fusion(text_to_image: torch.Tensor,
                    image_to_text: torch.Tensor,
                    gate_dim: int = 256) -> torch.Tensor:
    """
    Fuse cross-modal representations using soft gating.
    
    Args:
        text_to_image: Text->image attention output
        image_to_text: Image->text attention output
        gate_dim: Dimension of gating vector
    
    Returns:
        Fused representation
    """
    # Concatenate representations
    concat_features = torch.cat([text_to_image, image_to_text], dim=-1)
    
    # Compute gating weights
    gate = torch.sigmoid(torch.nn.Linear(concat_features.size(-1), gate_dim)(concat_features))
    
    # Apply gating
    fused = gate * text_to_image + (1 - gate) * image_to_text
    
    return fused

def aggregate_window(posts: List[Dict], alpha: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Aggregate a batch of posts into temporal window with decay weighting.
    
    Args:
        posts: List of post dictionaries containing features and timestamps
        alpha: Decay coefficient for temporal weighting
    
    Returns:
        Dictionary containing aggregated features and metadata
    """
    # Extract features and timestamps
    text_features = torch.stack([p['text_feat'] for p in posts])
    image_features = torch.stack([p['image_feat'] for p in posts])
    timestamps = [p['timestamp'] for p in posts]
    labels = torch.tensor([p['label'] for p in posts])
    
    # Compute temporal weights
    weights = compute_temporal_weights(timestamps, alpha)
    weights = torch.from_numpy(weights).float()
    
    # Apply cross-modal attention
    text_to_image, image_to_text = cross_modal_attention(text_features, image_features)
    
    # Fuse with soft gating
    fused_features = soft_gate_fusion(text_to_image, image_to_text)
    
    # Weighted aggregation
    weighted_features = (fused_features * weights.unsqueeze(-1)).sum(dim=0)
    
    return {
        'features': weighted_features,
        'labels': labels,
        'timestamps': torch.tensor([normalize_timestamp(ts) for ts in timestamps]),
        'weights': weights
    }

def get_window_weights(timestamps: torch.Tensor, 
                      decay_factor: float = 0.1) -> torch.Tensor:
    """Calculate temporal weights for posts in a window."""
    # Normalize timestamps to [0, 1]
    normalized_ts = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    
    # Calculate exponential decay weights
    weights = torch.exp(-decay_factor * normalized_ts)
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    return weights

def load_event_data(event_path: str) -> Dict:
    """Load event data from JSON file."""
    with open(event_path, 'r') as f:
        return json.load(f)

def load_window_data(window_path: str) -> Dict:
    """Load window data from JSON file."""
    with open(window_path, 'r') as f:
        return json.load(f)

def get_feature_paths(data_dir: str, 
                     split: str, 
                     modality: str,
                     dataset: str = 'fakeddit') -> Dict[str, str]:
    """Get paths to feature files."""
    base_path = Path(data_dir) / 'processed' / dataset / 'features'
    return {
        'bert': str(base_path / f'{split}_bert.npy'),
        'resnet': str(base_path / f'{split}_resnet.npy')
    }

def validate_data(data: Dict) -> bool:
    """Validate data dictionary has required keys and correct types."""
    required_keys = ['features', 'labels', 'timestamps']
    
    # Check all required keys exist
    if not all(key in data for key in required_keys):
        return False
    
    # Check types
    if not isinstance(data['features'], torch.Tensor):
        return False
    if not isinstance(data['labels'], torch.Tensor):
        return False
    if not isinstance(data['timestamps'], torch.Tensor):
        return False
    
    # Check shapes
    batch_size = data['features'].size(0)
    if not all(tensor.size(0) == batch_size for tensor in data.values()):
        return False
    
    return True

def load_fact_check_data(data_dir: str, split: str) -> Dict[str, str]:
    """Load fact-check URLs for COVID-19 dataset."""
    fact_check_path = Path(data_dir) / 'processed' / 'covid19' / f'{split}_fact_checks.json'
    if fact_check_path.exists():
        with open(fact_check_path, 'r') as f:
            return json.load(f)
    return {}

def validate_fact_check_url(url: str) -> bool:
    """Validate if a fact-check URL is accessible."""
    import requests
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except:
        return False

def get_dataset_config(dataset: str) -> Dict:
    """Get dataset-specific configuration."""
    configs = {
        'fakeddit': {
            'num_classes': 2,
            'feature_dims': {
                'bert': 768,
                'resnet': 2048
            }
        },
        'ind': {
            'num_classes': 2,
            'feature_dims': {
                'bert': 768,
                'resnet': 2048
            }
        },
        'covid19': {
            'num_classes': 2,
            'feature_dims': {
                'bert': 768,
                'resnet': 2048
            },
            'has_fact_checks': True
        }
    }
    return configs.get(dataset, {}) 