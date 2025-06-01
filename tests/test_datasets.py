import torch
import pytest
from datasets.fakeddit import FakedditDataset
from datasets.ind import INDDataset
from datasets.covid import COVIDDataset

def test_fakeddit_dataset():
    """Test Fakeddit dataset loading and processing."""
    dataset = FakedditDataset(data_dir='data/processed/fakeddit', split='train')
    
    # Test sample
    sample = dataset[0]
    assert isinstance(sample['text'], torch.Tensor)
    assert isinstance(sample['images'], torch.Tensor)
    assert isinstance(sample['labels'], torch.Tensor)
    assert isinstance(sample['timestamps'], torch.Tensor)
    
    # Test shapes
    assert sample['text'].shape[0] == 512  # BERT max length
    assert sample['images'].shape[0] == 3  # RGB channels
    assert sample['images'].shape[1] == 224  # Image height
    assert sample['images'].shape[2] == 224  # Image width
    assert sample['labels'].shape == (1,)  # Binary classification
    assert sample['timestamps'].shape == (1,)  # Single timestamp

def test_ind_dataset():
    """Test IND dataset loading and processing."""
    dataset = INDDataset(data_dir='data/processed/ind', split='train')
    
    # Test sample
    sample = dataset[0]
    assert isinstance(sample['text'], torch.Tensor)
    assert isinstance(sample['images'], torch.Tensor)
    assert isinstance(sample['labels'], torch.Tensor)
    assert isinstance(sample['timestamps'], torch.Tensor)
    
    # Test shapes
    assert sample['text'].shape[0] == 512
    assert sample['images'].shape[0] == 3
    assert sample['images'].shape[1] == 224
    assert sample['images'].shape[2] == 224
    assert sample['labels'].shape == (1,)
    assert sample['timestamps'].shape == (1,)

def test_covid_dataset():
    """Test COVID-19 dataset loading and processing."""
    dataset = COVIDDataset(data_dir='data/processed/covid', split='train')
    
    # Test sample
    sample = dataset[0]
    assert isinstance(sample['text'], torch.Tensor)
    assert isinstance(sample['images'], torch.Tensor)
    assert isinstance(sample['labels'], torch.Tensor)
    assert isinstance(sample['timestamps'], torch.Tensor)
    
    # Test shapes
    assert sample['text'].shape[0] == 512
    assert sample['images'].shape[0] == 3
    assert sample['images'].shape[1] == 224
    assert sample['images'].shape[2] == 224
    assert sample['labels'].shape == (1,)
    assert sample['timestamps'].shape == (1,) 