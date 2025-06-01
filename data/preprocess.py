import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel

class PseudoEventConstructor:
    """Constructs pseudo-events from social media posts using BERT embeddings and agglomerative clustering."""
    
    def __init__(
        self,
        bert_model: str = 'bert-base-uncased',
        distance_threshold: float = 0.5,
        min_cluster_size: int = 5
    ):
        """
        Initialize the pseudo-event constructor.
        
        Args:
            bert_model: Name of the BERT model to use
            distance_threshold: Maximum distance for agglomerative clustering
            min_cluster_size: Minimum number of posts in a cluster
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertModel.from_pretrained(bert_model).to(self.device)
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        
    def get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract BERT [CLS] embeddings from texts.
        
        Args:
            texts: List of text posts
            
        Returns:
            numpy array of BERT embeddings
        """
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and get BERT embeddings
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding='max_length'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Get [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
        
        return np.array(embeddings)
    
    def cluster_posts(
        self,
        embeddings: np.ndarray,
        timestamps: List[datetime]
    ) -> np.ndarray:
        """
        Cluster posts using agglomerative clustering.
        
        Args:
            embeddings: BERT embeddings of posts
            timestamps: List of post timestamps
            
        Returns:
            numpy array of cluster labels
        """
        # Initialize clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,  # Let the distance threshold determine clusters
            distance_threshold=self.distance_threshold,
            linkage='average'
        )
        
        # Get cluster labels
        labels = clustering.fit_predict(embeddings)
        
        # Filter small clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        small_clusters = unique_labels[counts < self.min_cluster_size]
        
        # Assign small clusters to -1 (noise)
        for label in small_clusters:
            labels[labels == label] = -1
            
        return labels
    
    def construct_events(
        self,
        posts: List[Dict],
        output_path: str
    ) -> None:
        """
        Construct pseudo-events from posts and save to file.
        
        Args:
            posts: List of post dictionaries with 'text', 'timestamp', etc.
            output_path: Path to save the processed data
        """
        # Extract texts and timestamps
        texts = [post['text'] for post in posts]
        timestamps = [datetime.fromisoformat(post['timestamp']) for post in posts]
        
        # Get BERT embeddings
        embeddings = self.get_bert_embeddings(texts)
        
        # Cluster posts
        event_labels = self.cluster_posts(embeddings, timestamps)
        
        # Add event IDs to posts
        processed_posts = []
        for post, event_id in zip(posts, event_labels):
            processed_post = post.copy()
            processed_post['event_id'] = int(event_id)
            processed_posts.append(processed_post)
        
        # Save processed data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(processed_posts, f, indent=2)
            
        print(f"Processed {len(posts)} posts into {len(np.unique(event_labels))} events")
        print(f"Saved to {output_path}")

def main():
    """Example usage of the pseudo-event constructor."""
    # Example posts
    posts = [
        {
            'post_id': '1',
            'text': 'Breaking: Major event in city center',
            'image_path': 'images/1.jpg',
            'timestamp': '2024-03-20T10:00:00'
        },
        # Add more posts...
    ]
    
    # Initialize constructor
    constructor = PseudoEventConstructor(
        bert_model='bert-base-uncased',
        distance_threshold=0.5,
        min_cluster_size=5
    )
    
    # Construct events
    constructor.construct_events(
        posts=posts,
        output_path='data/processed/events.json'
    )

if __name__ == '__main__':
    main() 