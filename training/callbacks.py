import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any
from pathlib import Path

class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 monitor: str = 'val_loss'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics
            monitor: Metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
        
    def on_epoch_end(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Whether to stop training
        """
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return False
            
        if self.mode == 'min':
            is_better = current_value < self.best_value - self.min_delta
        else:
            is_better = current_value > self.best_value + self.min_delta
            
        if is_better:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            
        self.should_stop = self.counter >= self.patience
        return self.should_stop

class ModelCheckpoint:
    """Callback to save model checkpoints."""
    
    def __init__(self,
                 save_dir: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_last: bool = True):
        """
        Initialize model checkpoint.
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' for loss, 'max' for metrics
            save_best_only: Whether to save only best model
            save_last: Whether to save last model
        """
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_epoch_end(self,
                    epoch: int,
                    model: torch.nn.Module,
                    metrics: Dict[str, float]) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            model: Model to save
            metrics: Dictionary of metrics
        """
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return
            
        # Check if current model is best
        is_best = False
        if self.mode == 'min':
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value
            
        if is_best:
            self.best_value = current_value
            
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }
        
        # Save best model
        if is_best and self.save_best_only:
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
            
        # Save last model
        if self.save_last:
            torch.save(checkpoint, self.save_dir / 'last_model.pt')

class LearningRateMonitor:
    """Callback to monitor learning rate."""
    
    def __init__(self, log_dir: str):
        """
        Initialize learning rate monitor.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lr_history = []
        
    def on_epoch_end(self,
                    epoch: int,
                    optimizer: torch.optim.Optimizer) -> None:
        """
        Log learning rate.
        
        Args:
            epoch: Current epoch
            optimizer: Optimizer to monitor
        """
        # Get current learning rate
        lr = optimizer.param_groups[0]['lr']
        self.lr_history.append(lr)
        
        # Save to file
        with open(self.log_dir / 'lr_history.json', 'w') as f:
            json.dump(self.lr_history, f)

class MetricsLogger:
    """Callback to log training metrics."""
    
    def __init__(self,
                 log_dir: str,
                 metrics: List[str]):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
            metrics: List of metrics to log
        """
        self.log_dir = Path(log_dir)
        self.metrics = metrics
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics history
        self.history = {metric: [] for metric in metrics}
        
    def on_epoch_end(self,
                    epoch: int,
                    metrics: Dict[str, float]) -> None:
        """
        Log metrics.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        # Update history
        for metric in self.metrics:
            value = metrics.get(metric)
            if value is not None:
                self.history[metric].append(value)
                
        # Save to CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_dir / 'metrics.csv', index=False)
        
    def get_best_epoch(self, metric: str, mode: str = 'min') -> int:
        """
        Get epoch with best metric value.
        
        Args:
            metric: Metric to check
            mode: 'min' for loss, 'max' for metrics
            
        Returns:
            Best epoch
        """
        if metric not in self.history:
            raise ValueError(f"Metric {metric} not found in history")
            
        values = self.history[metric]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
            
        return best_idx 