import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict, Tuple
import json
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm

def normalize_timestamps(timestamps: List[str]) -> np.ndarray:
    """Normalize timestamps to [0,1] range."""
    # Convert string timestamps to datetime objects
    dt_timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
    # Convert to float timestamps
    float_timestamps = np.array([dt.timestamp() for dt in dt_timestamps])
    # Normalize to [0,1]
    return (float_timestamps - float_timestamps.min()) / (float_timestamps.max() - float_timestamps.min())

def cluster_pseudo_events(
    embeddings: np.ndarray,
    timestamps: List[str],
    post_ids: List[str],
    num_clusters: int = 500,
    alpha: float = 0.1
) -> Dict[int, Dict]:
    """
    Clusters posts into pseudo-events using BERT embeddings and temporal proximity.
    
    Args:
        embeddings: np.ndarray of shape (N, D) containing BERT embeddings
        timestamps: List of timestamps for each post
        post_ids: List of post IDs
        num_clusters: Number of pseudo-events to create
        alpha: Weight for temporal component in clustering
    
    Returns:
        Dictionary mapping event IDs to event data
    """
    # Normalize timestamps
    ts_norm = normalize_timestamps(timestamps).reshape(-1, 1)
    
    # Combine embeddings with temporal features
    # Scale temporal component by alpha to control its influence
    features = np.hstack([embeddings, alpha * ts_norm])
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        affinity='cosine',
        linkage='average'
    )
    cluster_assignments = clustering.fit_predict(features)
    
    # Group posts by cluster
    events = {}
    for i, cluster_id in enumerate(cluster_assignments):
        if cluster_id not in events:
            events[cluster_id] = {
                'post_ids': [],
                'timestamps': [],
                'embeddings': []
            }
        
        events[cluster_id]['post_ids'].append(post_ids[i])
        events[cluster_id]['timestamps'].append(timestamps[i])
        events[cluster_id]['embeddings'].append(embeddings[i])
    
    return events

def save_events(events: Dict[int, Dict], output_dir: str):
    """Save events to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for event_id, event_data in events.items():
        # Convert embeddings to list for JSON serialization
        event_data['embeddings'] = [emb.tolist() for emb in event_data['embeddings']]
        
        # Save event data
        with open(output_path / f'event_{event_id}.json', 'w') as f:
            json.dump(event_data, f)

def create_temporal_windows(
    events: Dict[int, Dict],
    window_size: int = 5,
    overlap: float = 0.5
) -> Dict[int, Dict]:
    """
    Create overlapping temporal windows for each event.
    
    Args:
        events: Dictionary of events
        window_size: Number of posts per window
        overlap: Fraction of overlap between windows
    
    Returns:
        Dictionary mapping event IDs to window data
    """
    windows = {}
    
    for event_id, event_data in events.items():
        # Sort posts by timestamp
        sorted_indices = np.argsort(event_data['timestamps'])
        post_ids = np.array(event_data['post_ids'])[sorted_indices]
        timestamps = np.array(event_data['timestamps'])[sorted_indices]
        embeddings = np.array(event_data['embeddings'])[sorted_indices]
        
        # Calculate step size based on overlap
        step = int(window_size * (1 - overlap))
        
        # Create windows
        event_windows = []
        for i in range(0, len(post_ids) - window_size + 1, step):
            window = {
                'post_ids': post_ids[i:i + window_size].tolist(),
                'timestamps': timestamps[i:i + window_size].tolist(),
                'embeddings': embeddings[i:i + window_size].tolist()
            }
            event_windows.append(window)
        
        windows[event_id] = event_windows
    
    return windows

def save_windows(windows: Dict[int, List[Dict]], output_dir: str):
    """Save windows to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for event_id, event_windows in windows.items():
        for window_idx, window_data in enumerate(event_windows):
            # Save window data
            with open(output_path / f'event_{event_id}_window_{window_idx}.json', 'w') as f:
                json.dump(window_data, f) 