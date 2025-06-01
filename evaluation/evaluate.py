import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any
from .metrics import compute_metrics

def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on test set and compute metrics.
    
    Args:
        model: E-CaTCH model
        dataloader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing the five evaluation metrics
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            text = batch['text'].to(device)
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            timestamps = batch.get('timestamps')
            if timestamps is not None:
                timestamps = timestamps.to(device)
                
            # Forward pass
            outputs = model(
                text=text,
                images=images,
                timestamps=timestamps
            )
            
            # Get predictions and probabilities
            logits = outputs['logits']
            probs = torch.sigmoid(logits)  # Binary classification
            preds = (probs > 0.5).float()
            
            # Store results
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            
    # Concatenate all batches
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    
    # Compute metrics
    return compute_metrics(y_true, y_pred, y_prob)

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics, save_path=None):
    plt.figure(figsize=(10, 6))
    metrics_values = list(metrics.values())
    metrics_names = list(metrics.keys())
    
    plt.bar(metrics_names, metrics_values)
    plt.title('Evaluation Metrics')
    plt.ylim(0, 1)
    
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # Example usage
    from models.ecatch import ECaTCH
    from datasets.fakeddit import FakedditDataset
    from torch.utils.data import DataLoader
    
    # Load model and data
    model = ECaTCH(config)
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    model = model.to(device)
    
    test_dataset = FakedditDataset(data_dir='data/processed/fakeddit', split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    
    # Plot results
    plot_confusion_matrix(test_loader.dataset.labels, test_loader.dataset.preds, 'evaluation/confusion_matrix.png')
    plot_metrics(metrics, 'evaluation/metrics.png')
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 