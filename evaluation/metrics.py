import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from typing import Dict, Union, Tuple

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute the five standard evaluation metrics as specified in the paper:
    Accuracy, Precision, Recall, F1-score, and AUC-ROC.
    
    Args:
        y_true: Ground truth labels [n_samples]
        y_pred: Predicted labels [n_samples]
        y_prob: Predicted probabilities for class 1 [n_samples]
        
    Returns:
        Dictionary containing the five metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob)
    } 