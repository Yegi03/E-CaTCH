import torch
import pytest
from models.ecatch import ECaTCH

@pytest.fixture
def config():
    return {
        'model': {
            'text_encoder': {
                'type': 'bert-base-uncased',
                'hidden_size': 768,
                'dropout': 0.1
            },
            'image_encoder': {
                'type': 'resnet50',
                'pretrained': True,
                'hidden_size': 2048,
                'dropout': 0.1
            },
            'attention': {
                'num_heads': 8,
                'dropout': 0.1
            },
            'fusion': {
                'hidden_size': 512,
                'dropout': 0.1
            }
        }
    }

def test_model_initialization(config):
    model = ECaTCH(config)
    assert isinstance(model, torch.nn.Module)

def test_model_forward_pass(config):
    model = ECaTCH(config)
    batch_size = 2
    seq_length = 512
    
    # Create dummy inputs
    text_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length)
    }
    image_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(text_input, image_input)
    
    # Check output shape
    assert output.shape == (batch_size, 2)  # Binary classification

def test_model_device_transfer(config):
    if torch.cuda.is_available():
        model = ECaTCH(config)
        model = model.cuda()
        
        # Check if model is on GPU
        assert next(model.parameters()).is_cuda

def test_model_gradient_flow(config):
    model = ECaTCH(config)
    batch_size = 2
    seq_length = 512
    
    # Create dummy inputs
    text_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length)
    }
    image_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(text_input, image_input)
    
    # Create dummy target
    target = torch.randint(0, 2, (batch_size,))
    
    # Compute loss and backward pass
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    loss.backward()
    
    # Check if gradients are computed
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any() 