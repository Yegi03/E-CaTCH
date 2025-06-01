import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, Tuple

class ImageEncoder(nn.Module):
    """ResNet-based image encoder that projects to a fixed dimension."""
    
    def __init__(self,
                 model_name: str = 'resnet152',
                 output_dim: int = 512,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        """
        Initialize image encoder.
        
        Args:
            model_name: Name of ResNet model ('resnet50', 'resnet101', 'resnet152')
            output_dim: Output dimension for projection
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze ResNet parameters
        """
        super().__init__()
        
        # Load pretrained ResNet
        if model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            self.resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection layer
        self.projection = nn.Linear(
            self.resnet.fc.in_features,  # 2048 for resnet152
            output_dim
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self,
                images: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Image tensor [batch_size, channels, height, width]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - features: Projected features [batch_size, output_dim]
                - backbone_features: Optional backbone features
        """
        # Get backbone features
        backbone_features = self.backbone(images)  # [batch_size, 2048, 1, 1]
        backbone_features = backbone_features.squeeze(-1).squeeze(-1)  # [batch_size, 2048]
        
        # Project to output dimension
        projected = self.projection(backbone_features)  # [batch_size, output_dim]
        
        # Apply layer normalization
        features = self.layer_norm(projected)
        
        result = {'features': features}
        
        # Add backbone features if requested
        if return_features:
            result['backbone_features'] = backbone_features
            
        return result
    
    def encode_image(self,
                    image: torch.Tensor,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convenience method to encode a single image.
        
        Args:
            image: Input image tensor [channels, height, width]
            device: Device to place tensors on
            
        Returns:
            Encoded features [1, output_dim]
        """
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        # Move to device if specified
        if device is not None:
            image = image.to(device)
        
        # Get features
        with torch.no_grad():
            outputs = self.forward(image)
            
        return outputs['features']
    
    def encode_batch(self,
                    images: torch.Tensor,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convenience method to encode a batch of images.
        
        Args:
            images: Input image tensor [batch_size, channels, height, width]
            device: Device to place tensors on
            
        Returns:
            Encoded features [batch_size, output_dim]
        """
        # Move to device if specified
        if device is not None:
            images = images.to(device)
        
        # Get features
        with torch.no_grad():
            outputs = self.forward(images)
            
        return outputs['features'] 