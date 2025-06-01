import torch
import pytest
from models.ecatch import ECatchModel
from models.encoders.text_encoder import TextEncoder
from models.encoders.image_encoder import ImageEncoder
from models.attention.intra_modal import IntraModalAttention
from models.attention.cross_modal import CrossModalAttention
from models.fusion.soft_gating import SoftGatingFusion
from models.temporal.trend_lstm import TrendLSTM

def test_text_encoder():
    """Test text encoder forward pass."""
    encoder = TextEncoder(model_name='bert-base-uncased', max_length=512)
    batch_size = 4
    
    # Create dummy input
    text = torch.randint(0, 1000, (batch_size, 512))
    
    # Forward pass
    outputs = encoder(text)
    
    # Check outputs
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, 768)  # BERT base hidden size

def test_image_encoder():
    """Test image encoder forward pass."""
    encoder = ImageEncoder(model_name='resnet50', pretrained=True)
    batch_size = 4
    
    # Create dummy input
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    outputs = encoder(images)
    
    # Check outputs
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, 2048)  # ResNet-50 output size

def test_intra_modal_attention():
    """Test intra-modal attention."""
    batch_size = 4
    seq_len = 10
    embed_dim = 768
    num_heads = 8
    
    attention = IntraModalAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    outputs = attention(x)
    
    # Check outputs
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, seq_len, embed_dim)

def test_cross_modal_attention():
    """Test cross-modal attention."""
    batch_size = 4
    seq_len = 10
    embed_dim = 768
    num_heads = 8
    
    attention = CrossModalAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Create dummy inputs
    text = torch.randn(batch_size, seq_len, embed_dim)
    image = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    outputs = attention(text, image)
    
    # Check outputs
    assert isinstance(outputs, dict)
    assert 'text_to_image' in outputs
    assert 'image_to_text' in outputs
    assert outputs['text_to_image'].shape == (batch_size, seq_len, embed_dim)
    assert outputs['image_to_text'].shape == (batch_size, seq_len, embed_dim)

def test_soft_gating_fusion():
    """Test soft gating fusion."""
    batch_size = 4
    seq_len = 10
    embed_dim = 768
    
    fusion = SoftGatingFusion(embed_dim=embed_dim)
    
    # Create dummy inputs
    text_to_image = torch.randn(batch_size, seq_len, embed_dim)
    image_to_text = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    outputs = fusion(text_to_image, image_to_text)
    
    # Check outputs
    assert isinstance(outputs, dict)
    assert 'fused' in outputs
    assert outputs['fused'].shape == (batch_size, seq_len, embed_dim)

def test_trend_lstm():
    """Test trend LSTM."""
    batch_size = 4
    seq_len = 10
    input_dim = 768
    hidden_dim = 512
    
    lstm = TrendLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)
    timestamps = torch.randn(batch_size, seq_len)
    
    # Forward pass
    outputs = lstm(x, timestamps)
    
    # Check outputs
    assert isinstance(outputs, dict)
    assert 'output' in outputs
    assert 'momentum' in outputs
    assert outputs['output'].shape == (batch_size, seq_len, hidden_dim)
    assert outputs['momentum'].shape == (batch_size, seq_len, hidden_dim)

def test_full_model():
    """Test full E-CaTCH model pipeline."""
    model = ECatchModel(
        text_encoder=TextEncoder(model_name='bert-base-uncased'),
        image_encoder=ImageEncoder(model_name='resnet50'),
        embed_dim=768,
        num_heads=8,
        lstm_hidden_dim=512,
        num_lstm_layers=2,
        dropout=0.1
    )
    
    batch_size = 4
    
    # Create dummy inputs
    text = torch.randint(0, 1000, (batch_size, 512))
    images = torch.randn(batch_size, 3, 224, 224)
    timestamps = torch.randn(batch_size, 10)
    
    # Forward pass
    outputs = model(text, images, timestamps)
    
    # Check outputs
    assert isinstance(outputs, dict)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, 1)  # Binary classification 