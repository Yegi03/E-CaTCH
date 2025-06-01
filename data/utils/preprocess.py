import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import requests
from pathlib import Path
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
import re

class DataPreprocessor:
    def __init__(self, data_dir: str, cache_dir: str = None):
        self.data_dir = data_dir
        self.cache_dir = cache_dir or os.path.join(data_dir, 'external', 'image_cache')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_model.eval()
        
        # Create necessary directories
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize raw text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text.lower()
    
    def download_image(self, url: str, output_path: str) -> bool:
        """Download image from URL and save to output path."""
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return False
    
    def generate_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings for a list of texts."""
        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts, desc="Generating BERT embeddings"):
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding)
        return np.vstack(embeddings)
    
    def generate_resnet_embeddings(self, img_paths: List[str]) -> np.ndarray:
        """Generate ResNet embeddings for a list of images."""
        embeddings = []
        transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
        
        with torch.no_grad():
            for img_path in tqdm(img_paths, desc="Generating ResNet embeddings"):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img).unsqueeze(0)
                    features = self.resnet_model(img)
                    embedding = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
                    embeddings.append(embedding.numpy())
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    # Use zero vector for failed images
                    embeddings.append(np.zeros((1, 2048)))
        return np.vstack(embeddings)
    
    def cluster_pseudo_events(self, 
                            bert_embeds: np.ndarray, 
                            timestamps: List[str],
                            n_clusters: int = 100) -> Dict[int, List[int]]:
        """Cluster posts into pseudo-events using BERT embeddings and timestamps."""
        # Convert timestamps to float values
        time_values = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() 
                      for ts in timestamps]
        time_values = np.array(time_values).reshape(-1, 1)
        
        # Combine embeddings and timestamps
        combined_features = np.hstack([bert_embeds, time_values])
        
        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(combined_features)
        
        # Group post indices by cluster
        events = {}
        for idx, label in enumerate(cluster_labels):
            if label not in events:
                events[label] = []
            events[label].append(idx)
        
        return events
    
    def segment_overlapping_windows(self, 
                                  posts: List[Dict],
                                  window_size: int = 5,
                                  stride: int = 2) -> List[List[Dict]]:
        """Split each event into overlapping windows."""
        windows = []
        for i in range(0, len(posts) - window_size + 1, stride):
            window = posts[i:i + window_size]
            windows.append(window)
        return windows
    
    def preprocess_fakeddit(self):
        """Preprocess the Fakeddit dataset."""
        raw_dir = os.path.join(self.data_dir, 'raw', 'fakeddit')
        processed_dir = os.path.join(self.data_dir, 'processed', 'fakeddit')
        
        # Create processed directories
        for subdir in ['events', 'windows', 'features']:
            os.makedirs(os.path.join(processed_dir, subdir), exist_ok=True)
        
        # Process each split
        for split in ['train', 'dev', 'test']:
            print(f"Processing {split} split...")
            
            # Load data
            df = pd.read_csv(os.path.join(raw_dir, f'{split}.csv'))
            
            # Clean text
            df['text'] = df['text'].apply(self.clean_text)
            
            # Download images
            for idx, row in tqdm(df.iterrows(), desc=f"Downloading {split} images"):
                img_id = row['image_id']
                img_path = os.path.join(processed_dir, 'images', img_id)
                
                if not os.path.exists(img_path):
                    self.download_image(row['image_url'], img_path)
            
            # Generate embeddings
            bert_embeds = self.generate_bert_embeddings(df['text'].tolist())
            img_paths = [os.path.join(processed_dir, 'images', img_id) 
                        for img_id in df['image_id']]
            resnet_embeds = self.generate_resnet_embeddings(img_paths)
            
            # Save embeddings
            np.save(os.path.join(processed_dir, 'features', f'{split}_bert.npy'), bert_embeds)
            np.save(os.path.join(processed_dir, 'features', f'{split}_resnet.npy'), resnet_embeds)
            
            # Cluster into events
            events = self.cluster_pseudo_events(
                bert_embeds, 
                df['timestamp'].tolist()
            )
            
            # Save events
            for event_id, post_indices in events.items():
                event_data = {
                    'post_ids': df.iloc[post_indices]['id'].tolist(),
                    'timestamps': df.iloc[post_indices]['timestamp'].tolist(),
                    'labels': df.iloc[post_indices]['label'].tolist()
                }
                
                with open(os.path.join(processed_dir, 'events', f'event_{event_id}.json'), 'w') as f:
                    json.dump(event_data, f)
                
                # Create windows
                posts = [{
                    'id': df.iloc[idx]['id'],
                    'text': df.iloc[idx]['text'],
                    'image_id': df.iloc[idx]['image_id'],
                    'label': df.iloc[idx]['label'],
                    'timestamp': df.iloc[idx]['timestamp']
                } for idx in post_indices]
                
                windows = self.segment_overlapping_windows(posts)
                
                for window_idx, window in enumerate(windows):
                    window_data = {
                        'post_ids': [p['id'] for p in window],
                        'timestamps': [p['timestamp'] for p in window],
                        'labels': [p['label'] for p in window]
                    }
                    
                    with open(os.path.join(processed_dir, 'windows', 
                                         f'event_{event_id}_window_{window_idx}.json'), 'w') as f:
                        json.dump(window_data, f)
            
            print(f"Processed {split} split: {len(df)} samples")
    
    def preprocess_ind(self):
        """Preprocess the India Elections dataset."""
        # Similar structure to Fakeddit but with dataset-specific adjustments
        pass
    
    def preprocess_covid19(self):
        """Preprocess the COVID-19 dataset."""
        # Similar structure to Fakeddit but with dataset-specific adjustments
        pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                      choices=['fakeddit', 'ind', 'covid19'],
                      help='Dataset to preprocess')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Root directory for data')
    parser.add_argument('--cache_dir', type=str, default=None,
                      help='Directory for caching downloaded images')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.data_dir, args.cache_dir)
    
    if args.dataset == 'fakeddit':
        preprocessor.preprocess_fakeddit()
    elif args.dataset == 'ind':
        preprocessor.preprocess_ind()
    elif args.dataset == 'covid19':
        preprocessor.preprocess_covid19()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

if __name__ == '__main__':
    main() 