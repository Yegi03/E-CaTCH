import torch
import pytest
from models.ecatch import ECatchModel
from models.encoders.text_encoder import TextEncoder
from models.encoders.image_encoder import ImageEncoder

def test_full_pipeline():
    """Test the full E-CaTCH pipeline with dummy data."""
    # Initialize model
    model = ECatchModel(
        text_encoder='bert-base-uncased',
        image_encoder='resnet50',
        embed_dim=768,
        num_heads=8,
        lstm_hidden_dim=512,
        num_lstm_layers=2,
        dropout=0.1
    )
    
    # Create dummy batch
    batch_size = 4
    text = torch.randint(0, 1000, (batch_size, 512))  # BERT token IDs
    images = torch.randn(batch_size, 3, 224, 224)  # RGB images
    timestamps = torch.randn(batch_size, 10)  # Temporal features
    
    # Forward pass
    outputs = model(text, images, timestamps)
    
    # Check output structure
    assert isinstance(outputs, dict)
    assert 'logits' in outputs
    assert 'attention_weights' in outputs
    assert 'temporal_weights' in outputs
    
    # Check shapes
    assert outputs['logits'].shape == (batch_size, 1)  # Binary classification
    assert outputs['attention_weights']['text_to_image'].shape == (batch_size, 512, 768)
    assert outputs['attention_weights']['image_to_text'].shape == (batch_size, 512, 768)
    assert outputs['temporal_weights'].shape == (batch_size, 10)
    
    # Check value ranges
    assert torch.all(outputs['logits'] >= -100) and torch.all(outputs['logits'] <= 100)
    assert torch.all(outputs['temporal_weights'] >= 0) and torch.all(outputs['temporal_weights'] <= 1)
    
    # Check attention weights
    text_to_image = outputs['attention_weights']['text_to_image']
    image_to_text = outputs['attention_weights']['image_to_text']
    assert torch.all(text_to_image >= 0) and torch.all(text_to_image <= 1)
    assert torch.all(image_to_text >= 0) and torch.all(image_to_text <= 1)

def test_model_components():
    """Test individual model components."""
    # Initialize encoders
    text_encoder = TextEncoder(model_name='bert-base-uncased')
    image_encoder = ImageEncoder(model_name='resnet50')
    
    # Create dummy inputs
    batch_size = 4
    text = torch.randint(0, 1000, (batch_size, 512))
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Test text encoder
    text_features = text_encoder(text)
    assert text_features.shape == (batch_size, 768)
    
    # Test image encoder
    image_features = image_encoder(images)
    assert image_features.shape == (batch_size, 2048)

def test_temporal_features():
    """Test temporal feature handling."""
    model = ECatchModel(
        text_encoder='bert-base-uncased',
        image_encoder='resnet50',
        embed_dim=768,
        num_heads=8,
        lstm_hidden_dim=512,
        num_lstm_layers=2,
        dropout=0.1
    )
    
    # Create dummy batch with temporal features
    batch_size = 4
    text = torch.randint(0, 1000, (batch_size, 512))
    images = torch.randn(batch_size, 3, 224, 224)
    timestamps = torch.randn(batch_size, 10)
    
    # Forward pass with timestamps
    outputs = model(text, images, timestamps)
    
    # Check temporal weights
    assert 'temporal_weights' in outputs
    assert outputs['temporal_weights'].shape == (batch_size, 10)
    assert torch.all(outputs['temporal_weights'] >= 0)
    assert torch.all(outputs['temporal_weights'] <= 1)
    
    # Check temporal consistency
    weights = outputs['temporal_weights']
    for i in range(1, weights.shape[1]):
        # Weights should decrease over time (temporal decay)
        assert torch.all(weights[:, i] <= weights[:, i-1]) 