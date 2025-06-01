import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from .utils import (
    load_event_data,
    load_window_data,
    load_post_embedding,
    get_feature_paths,
    normalize_timestamp,
    compute_temporal_weights,
    aggregate_window,
    compute_semantic_shift,
    compute_temporal_momentum,
    cross_modal_attention,
    soft_gate_fusion,
    load_fact_check_data,
    validate_fact_check_url
)

class COVID19Dataset(Dataset):
    """Dataset class for loading individual posts from COVID-19 dataset."""
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 event_id: Optional[str] = None,
                 window_id: Optional[str] = None,
                 use_cache: bool = True,
                 transform = None,
                 handle_imbalance: bool = True,
                 include_fact_checks: bool = True,
                 alpha: float = 0.1,
                 beta: float = 0.9):
        """
        Initialize COVID-19 dataset.
        
        Args:
            data_dir: Path to dataset directory
            split: Data split (train/dev/test)
            event_id: Optional event ID to filter by
            window_id: Optional window ID to filter by
            use_cache: Whether to use cached features
            transform: Optional transform to apply to image features
            handle_imbalance: Whether to handle class imbalance
            include_fact_checks: Whether to include fact-check URLs
            alpha: Decay coefficient for temporal weighting
            beta: Smoothing coefficient for temporal momentum
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.event_id = event_id
        self.window_id = window_id
        self.use_cache = use_cache
        self.transform = transform
        self.handle_imbalance = handle_imbalance
        self.include_fact_checks = include_fact_checks
        self.alpha = alpha
        self.beta = beta
        
        # Load events and windows
        self.events = self._load_events()
        self.windows = self._load_windows()
        
        # Load fact-check URLs if needed
        if include_fact_checks:
            self.fact_checks = load_fact_check_data(data_dir, split)
        else:
            self.fact_checks = {}
        
        # Create samples
        self.samples = self._create_samples()
        
        # Handle class imbalance if needed
        if handle_imbalance and split == 'train':
            self.samples = self._balance_samples()
            
        # Initialize temporal momentum
        self.previous_momentum = torch.zeros(1)
        
    def _load_events(self) -> Dict:
        """Load event data."""
        events_path = self.data_dir / 'processed' / 'covid19' / 'events'
        if self.event_id:
            return load_event_data(str(events_path / f'{self.event_id}.json'))
        return {p.stem: load_event_data(str(p)) for p in events_path.glob('*.json')}
    
    def _load_windows(self) -> Dict:
        """Load window data."""
        windows_path = self.data_dir / 'processed' / 'covid19' / 'windows'
        if self.window_id:
            return load_window_data(str(windows_path / f'{self.window_id}.json'))
        return {p.stem: load_window_data(str(p)) for p in windows_path.glob('*.json')}
    
    def _create_samples(self) -> List[Dict]:
        """Create list of samples from events and windows."""
        samples = []
        feature_paths = get_feature_paths(self.data_dir, self.split, 'covid19')
        
        for event_id, event in self.events.items():
            for window_id, window in self.windows.items():
                if window['event_id'] == event_id:
                    for post in window['posts']:
                        # Load features
                        text_feat = load_post_embedding(post['id'], 'bert', feature_paths['bert'])
                        image_feat = load_post_embedding(post['id'], 'resnet', feature_paths['resnet'])
                        
                        # Convert to tensors
                        text_feat = torch.from_numpy(text_feat).float()
                        image_feat = torch.from_numpy(image_feat).float()
                        
                        # Apply cross-modal attention
                        text_to_image, image_to_text = cross_modal_attention(
                            text_feat.unsqueeze(0), 
                            image_feat.unsqueeze(0)
                        )
                        
                        # Fuse with soft gating
                        fused_feat = soft_gate_fusion(
                            text_to_image.squeeze(0),
                            image_to_text.squeeze(0)
                        )
                        
                        sample = {
                            'id': post['id'],
                            'features': fused_feat,
                            'label': post['label'],
                            'timestamp': post['timestamp'],
                            'event_id': event_id,
                            'window_id': window_id
                        }
                        
                        # Add fact-check URL if available
                        if self.include_fact_checks and post['id'] in self.fact_checks:
                            sample['fact_check_url'] = self.fact_checks[post['id']]
                        
                        samples.append(sample)
        
        return samples
    
    def _balance_samples(self) -> List[Dict]:
        """Balance samples by oversampling minority class."""
        labels = np.array([s['label'] for s in self.samples])
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if len(unique_labels) != 2:
            return self.samples
        
        # Find minority class
        minority_label = unique_labels[np.argmin(counts)]
        minority_samples = [s for s in self.samples if s['label'] == minority_label]
        majority_samples = [s for s in self.samples if s['label'] != minority_label]
        
        # Oversample minority class
        n_samples = len(majority_samples)
        minority_oversampled = np.random.choice(
            minority_samples,
            size=n_samples,
            replace=True
        ).tolist()
        
        return majority_samples + minority_oversampled
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Apply transform if specified
        if self.transform is not None:
            sample['features'] = self.transform(sample['features'])
        
        return sample

class COVID19WindowDataset(Dataset):
    """Dataset class for loading entire windows from COVID-19 dataset."""
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 event_id: Optional[str] = None,
                 window_id: Optional[str] = None,
                 use_cache: bool = True,
                 transform = None,
                 handle_imbalance: bool = True,
                 include_fact_checks: bool = True,
                 alpha: float = 0.1,
                 beta: float = 0.9):
        """
        Initialize COVID-19 window dataset.
        
        Args:
            data_dir: Path to dataset directory
            split: Data split (train/dev/test)
            event_id: Optional event ID to filter by
            window_id: Optional window ID to filter by
            use_cache: Whether to use cached features
            transform: Optional transform to apply to image features
            handle_imbalance: Whether to handle class imbalance
            include_fact_checks: Whether to include fact-check URLs
            alpha: Decay coefficient for temporal weighting
            beta: Smoothing coefficient for temporal momentum
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.event_id = event_id
        self.window_id = window_id
        self.use_cache = use_cache
        self.transform = transform
        self.handle_imbalance = handle_imbalance
        self.include_fact_checks = include_fact_checks
        self.alpha = alpha
        self.beta = beta
        
        # Load events and windows
        self.events = self._load_events()
        self.windows = self._load_windows()
        
        # Load fact-check URLs if needed
        if include_fact_checks:
            self.fact_checks = load_fact_check_data(data_dir, split)
        else:
            self.fact_checks = {}
        
        # Create window samples
        self.window_samples = self._create_window_samples()
        
        # Handle class imbalance if needed
        if handle_imbalance and split == 'train':
            self.window_samples = self._balance_samples()
            
        # Initialize temporal momentum
        self.previous_momentum = torch.zeros(1)
        
    def _load_events(self) -> Dict:
        """Load event data."""
        events_path = self.data_dir / 'processed' / 'covid19' / 'events'
        if self.event_id:
            return load_event_data(str(events_path / f'{self.event_id}.json'))
        return {p.stem: load_event_data(str(p)) for p in events_path.glob('*.json')}
    
    def _load_windows(self) -> Dict:
        """Load window data."""
        windows_path = self.data_dir / 'processed' / 'covid19' / 'windows'
        if self.window_id:
            return load_window_data(str(windows_path / f'{self.window_id}.json'))
        return {p.stem: load_window_data(str(p)) for p in windows_path.glob('*.json')}
    
    def _create_window_samples(self) -> List[Dict]:
        """Create list of window samples."""
        window_samples = []
        feature_paths = get_feature_paths(self.data_dir, self.split, 'covid19')
        
        for event_id, event in self.events.items():
            for window_id, window in self.windows.items():
                if window['event_id'] == event_id:
                    # Load features for all posts in window
                    posts = []
                    for post in window['posts']:
                        text_feat = load_post_embedding(post['id'], 'bert', feature_paths['bert'])
                        image_feat = load_post_embedding(post['id'], 'resnet', feature_paths['resnet'])
                        
                        post_data = {
                            'id': post['id'],
                            'text_feat': torch.from_numpy(text_feat).float(),
                            'image_feat': torch.from_numpy(image_feat).float(),
                            'label': post['label'],
                            'timestamp': post['timestamp']
                        }
                        
                        # Add fact-check URL if available
                        if self.include_fact_checks and post['id'] in self.fact_checks:
                            post_data['fact_check_url'] = self.fact_checks[post['id']]
                        
                        posts.append(post_data)
                    
                    # Aggregate window with temporal weighting
                    window_data = aggregate_window(posts, self.alpha)
                    
                    # Compute semantic shift if not first window
                    if len(window_samples) > 0:
                        semantic_shift = compute_semantic_shift(
                            window_data['features'],
                            window_samples[-1]['features']
                        )
                        # Update temporal momentum
                        self.previous_momentum = compute_temporal_momentum(
                            semantic_shift,
                            self.previous_momentum,
                            self.beta
                        )
                        window_data['momentum'] = self.previous_momentum
                    else:
                        window_data['momentum'] = torch.zeros(1)
                    
                    window_samples.append({
                        'event_id': event_id,
                        'window_id': window_id,
                        **window_data
                    })
        
        return window_samples
    
    def _balance_samples(self) -> List[Dict]:
        """Balance samples by oversampling minority class."""
        labels = np.array([s['labels'].mean() for s in self.window_samples])
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if len(unique_labels) != 2:
            return self.window_samples
        
        # Find minority class
        minority_label = unique_labels[np.argmin(counts)]
        minority_samples = [s for s in self.window_samples if s['labels'].mean() == minority_label]
        majority_samples = [s for s in self.window_samples if s['labels'].mean() != minority_label]
        
        # Oversample minority class
        n_samples = len(majority_samples)
        minority_oversampled = np.random.choice(
            minority_samples,
            size=n_samples,
            replace=True
        ).tolist()
        
        return majority_samples + minority_oversampled
    
    def __len__(self) -> int:
        return len(self.window_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.window_samples[idx]
        
        # Apply transform if specified
        if self.transform is not None:
            sample['features'] = self.transform(sample['features'])
        
        return sample 