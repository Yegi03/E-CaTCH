import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

def compute_class_weights(labels: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: Ground truth labels [batch_size]
        epsilon: Small constant for numerical stability
        
    Returns:
        Class weights [num_classes]
    """
    # Count samples per class
    n_samples = labels.size(0)
    n_classes = labels.max().item() + 1
    n_per_class = torch.bincount(labels, minlength=n_classes)
    
    # Compute mean samples per class
    n_mean = n_samples / n_classes
    
    # Compute weights: w_c = n_mean / (n_c + epsilon)
    weights = n_mean / (n_per_class.float() + epsilon)
    
    return weights

def cross_entropy_with_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute weighted binary cross-entropy loss.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        weights: Optional class weights [num_classes]
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple of:
            - Weighted loss per sample [batch_size]
            - Dictionary containing:
                - total_loss: Mean loss across batch
                - per_class_loss: Loss per class
    """
    # Compute weights if not provided
    if weights is None:
        weights = compute_class_weights(labels, epsilon)
    
    # Move weights to same device as logits
    weights = weights.to(logits.device)
    
    # Compute per-sample loss
    per_sample_loss = F.cross_entropy(
        logits,
        labels,
        reduction='none',
        weight=weights
    )
    
    # Compute per-class loss
    per_class_loss = {}
    for c in range(weights.size(0)):
        mask = (labels == c)
        if mask.any():
            per_class_loss[f'class_{c}_loss'] = per_sample_loss[mask].mean()
    
    # Compute total loss
    total_loss = per_sample_loss.mean()
    
    return per_sample_loss, {
        'total_loss': total_loss,
        'per_class_loss': per_class_loss
    }

def temporal_consistency_loss(
    trend_embeddings: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute temporal consistency loss between consecutive embeddings.
    
    Args:
        trend_embeddings: LSTM outputs [batch_size, seq_len, embed_dim]
        mask: Optional padding mask [batch_size, seq_len]
        
    Returns:
        Tuple of:
            - Per-timestep loss [batch_size, seq_len-1]
            - Dictionary containing:
                - total_loss: Mean loss across batch and time
                - per_timestep_loss: Loss per timestep
    """
    # Get consecutive embeddings
    t_curr = trend_embeddings[:, 1:, :]  # [batch_size, seq_len-1, embed_dim]
    t_prev = trend_embeddings[:, :-1, :]  # [batch_size, seq_len-1, embed_dim]
    
    # Compute L2 distance
    l2_dist = torch.norm(t_curr - t_prev, dim=-1)  # [batch_size, seq_len-1]
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(t_curr, t_prev, dim=-1)  # [batch_size, seq_len-1]
    
    # Compute per-timestep loss
    per_timestep_loss = l2_dist * cos_sim  # [batch_size, seq_len-1]
    
    # Apply mask if provided
    if mask is not None:
        # Create mask for consecutive pairs
        pair_mask = mask[:, 1:] & mask[:, :-1]  # [batch_size, seq_len-1]
        per_timestep_loss = per_timestep_loss * pair_mask.float()
    
    # Compute total loss
    total_loss = per_timestep_loss.mean()
    
    return per_timestep_loss, {
        'total_loss': total_loss,
        'per_timestep_loss': per_timestep_loss.mean(dim=0)  # [seq_len-1]
    }

def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    trend_embeddings: torch.Tensor,
    model: Optional[nn.Module] = None,
    mask: Optional[torch.Tensor] = None,
    lambda_tc: float = 0.1,
    lambda_reg: float = 0.01,
    epsilon: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    Compute total loss combining CE, temporal consistency, and L2 regularization.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        trend_embeddings: LSTM outputs [batch_size, seq_len, embed_dim]
        model: Optional model for L2 regularization
        mask: Optional padding mask [batch_size, seq_len]
        lambda_tc: Weight for temporal consistency loss
        lambda_reg: Weight for L2 regularization
        epsilon: Small constant for numerical stability
        
    Returns:
        Dictionary containing:
            - total_loss: Combined loss
            - ce_loss: Cross-entropy loss
            - tc_loss: Temporal consistency loss
            - reg_loss: L2 regularization loss
            - per_class_loss: Loss per class
            - per_timestep_loss: Loss per timestep
    """
    # Compute cross-entropy loss
    _, ce_dict = cross_entropy_with_weights(logits, labels, epsilon=epsilon)
    
    # Compute temporal consistency loss
    _, tc_dict = temporal_consistency_loss(trend_embeddings, mask)
    
    # Initialize result dictionary
    result = {
        'ce_loss': ce_dict['total_loss'],
        'tc_loss': tc_dict['total_loss'],
        'per_class_loss': ce_dict['per_class_loss'],
        'per_timestep_loss': tc_dict['per_timestep_loss']
    }
    
    # Add L2 regularization if model is provided
    if model is not None and lambda_reg > 0:
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss += torch.norm(param, p=2)
        reg_loss = lambda_reg * reg_loss
        result['reg_loss'] = reg_loss
    else:
        reg_loss = torch.tensor(0.0, device=logits.device)
        result['reg_loss'] = reg_loss
    
    # Compute total loss
    result['total_loss'] = (
        ce_dict['total_loss'] +
        lambda_tc * tc_dict['total_loss'] +
        reg_loss
    )
    
    return result

def get_hard_examples(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    weights: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k highest-loss examples for hard example mining.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        k: Number of hard examples to select
        weights: Optional class weights [num_classes]
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple of:
            - Selected logits [k, num_classes]
            - Selected labels [k]
    """
    # Compute per-sample loss
    per_sample_loss, _ = cross_entropy_with_weights(
        logits, labels, weights, epsilon
    )
    
    # Get indices of top-k highest loss samples
    _, indices = torch.topk(per_sample_loss, k=min(k, per_sample_loss.size(0)))
    
    # Select corresponding logits and labels
    selected_logits = logits[indices]
    selected_labels = labels[indices]
    
    return selected_logits, selected_labels 