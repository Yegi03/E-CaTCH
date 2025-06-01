import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from models.ecatch import ECaTCH
from .loss import (
    cross_entropy_with_weights,
    temporal_consistency_loss,
    compute_total_loss,
    get_hard_examples
)
from .optimizer import get_optimizer, get_scheduler, clip_gradients
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    MetricsLogger
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class Trainer:
    """Main trainer class for E-CaTCH model."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model: E-CaTCH model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(
            model=self.model,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            scheduler_type=config['scheduler_type'],
            num_epochs=config['num_epochs'],
            num_steps=len(train_loader),
            warmup_steps=config.get('warmup_steps', 0)
        )
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Setup logging
        self._setup_logging()
        
    def _setup_callbacks(self) -> List[Any]:
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping
        if self.config.get('early_stopping'):
            callbacks.append(EarlyStopping(
                patience=self.config['early_stopping_patience'],
                min_delta=self.config.get('early_stopping_min_delta', 0.0),
                mode='min',
                monitor='val_loss'
            ))
            
        # Model checkpoint
        if self.config.get('save_checkpoints'):
            callbacks.append(ModelCheckpoint(
                save_dir=self.config['checkpoint_dir'],
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_last=True
            ))
            
        # Learning rate monitor
        callbacks.append(LearningRateMonitor(
            log_dir=self.config['log_dir']
        ))
        
        # Metrics logger
        callbacks.append(MetricsLogger(
            log_dir=self.config['log_dir'],
            metrics=['train_loss', 'val_loss', 'train_acc', 'val_acc']
        ))
        
        return callbacks
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['log_dir'] / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move batch to device
            text = batch['text'].to(self.device)
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            timestamps = batch.get('timestamps')
            if timestamps is not None:
                timestamps = timestamps.to(self.device)
                
            # Forward pass
            outputs = self.model(
                text=text,
                images=images,
                timestamps=timestamps,
                return_attention_weights=True
            )
            
            # Compute losses
            ce_loss = cross_entropy_with_weights(
                outputs['logits'],
                labels,
                self.train_loader.dataset.class_weights.to(self.device)
            )
            
            temp_loss = temporal_consistency_loss(
                outputs['lstm_output'],
                timestamps
            ) if timestamps is not None else 0.0
            
            loss_dict = compute_total_loss(
                ce_loss=ce_loss,
                temp_loss=temp_loss,
                model=self.model,
                l2_lambda=self.config['l2_lambda']
            )
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip'):
                clip_gradients(
                    self.model,
                    self.config['grad_clip_norm']
                )
                
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            preds = (outputs['logits'] > 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100 * correct / total
            })
            
        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': 100 * correct / total
        }
        
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                text = batch['text'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                timestamps = batch.get('timestamps')
                if timestamps is not None:
                    timestamps = timestamps.to(self.device)
                    
                # Forward pass
                outputs = self.model(
                    text=text,
                    images=images,
                    timestamps=timestamps
                )
                
                # Compute loss
                ce_loss = cross_entropy_with_weights(
                    outputs['logits'],
                    labels,
                    self.val_loader.dataset.class_weights.to(self.device)
                )
                
                temp_loss = temporal_consistency_loss(
                    outputs['lstm_output'],
                    timestamps
                ) if timestamps is not None else 0.0
                
                loss_dict = compute_total_loss(
                    ce_loss=ce_loss,
                    temp_loss=temp_loss,
                    model=self.model,
                    l2_lambda=self.config['l2_lambda']
                )
                loss = loss_dict['total_loss']
                
                # Update metrics
                total_loss += loss_dict['total_loss'].item()
                preds = (outputs['logits'] > 0).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': 100 * correct / total
        }
        
    def train(self):
        """Main training loop."""
        logging.info("Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            logging.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            logging.info(
                f"Train Loss: {metrics['train_loss']:.4f}, "
                f"Train Acc: {metrics['train_acc']:.2f}%, "
                f"Val Loss: {metrics['val_loss']:.4f}, "
                f"Val Acc: {metrics['val_acc']:.2f}%"
            )
            
            # Call callbacks
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    if callback.on_epoch_end(metrics):
                        logging.info("Early stopping triggered")
                        return
                elif isinstance(callback, ModelCheckpoint):
                    callback.on_epoch_end(epoch, self.model, metrics)
                elif isinstance(callback, LearningRateMonitor):
                    callback.on_epoch_end(epoch, self.optimizer)
                elif isinstance(callback, MetricsLogger):
                    callback.on_epoch_end(epoch, metrics)
                    
        logging.info("Training completed!")

if __name__ == '__main__':
    train('config/default.yaml') 