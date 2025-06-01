import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR
)
from typing import Dict, Optional, Union, List

def get_optimizer(
    model: nn.Module,
    config: Dict
) -> Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model: Model to optimize
        config: Dictionary containing:
            - lr: Learning rate
            - weight_decay: L2 regularization
            - betas: Adam betas
            - eps: Adam epsilon
            
    Returns:
        Configured optimizer
    """
    # Get optimizer parameters
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    betas = config.get('betas', (0.9, 0.999))
    eps = config.get('eps', 1e-8)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )
    
    return optimizer

def get_scheduler(
    optimizer: Optimizer,
    config: Dict,
    num_training_steps: Optional[int] = None
) -> Union[CosineAnnealingLR, ReduceLROnPlateau, StepLR]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Dictionary containing:
            - scheduler_type: 'cosine', 'plateau', or 'step'
            - warmup_steps: Number of warmup steps
            - min_lr: Minimum learning rate
            - patience: Patience for plateau scheduler
            - factor: Factor for step scheduler
            - step_size: Step size for step scheduler
        num_training_steps: Total number of training steps
        
    Returns:
        Configured scheduler
    """
    scheduler_type = config.get('scheduler_type', 'cosine')
    
    if scheduler_type == 'cosine':
        # Cosine annealing with warmup
        warmup_steps = config.get('warmup_steps', 0)
        min_lr = config.get('min_lr', 0)
        
        if num_training_steps is None:
            raise ValueError("num_training_steps must be provided for cosine scheduler")
            
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=min_lr
        )
        
    elif scheduler_type == 'plateau':
        # Reduce on plateau
        patience = config.get('patience', 10)
        factor = config.get('factor', 0.1)
        min_lr = config.get('min_lr', 0)
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
        
    elif scheduler_type == 'step':
        # Step decay
        step_size = config.get('step_size', 30)
        factor = config.get('factor', 0.1)
        
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=factor
        )
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    return scheduler

def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0
) -> float:
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: Model whose gradients to clip
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    # Compute total gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    return total_norm 