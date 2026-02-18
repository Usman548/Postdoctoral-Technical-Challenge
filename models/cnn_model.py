"""
CNN architectures for pneumonia classification.
Includes custom CNN, ResNet variants, and Vision Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CustomCNN(nn.Module):
    """
    Custom CNN architecture optimized for 28x28 medical images.
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        """
        Initialize custom CNN.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(CustomCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # 28->14
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # 14->7
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # Global pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TinyVisionTransformer(nn.Module):
    """
    Small Vision Transformer adapted for 28x28 images.
    Based on ViT architecture but scaled down.
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 img_size: int = 28,
                 patch_size: int = 4,
                 in_channels: int = 1,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        """
        Initialize Tiny Vision Transformer.
        
        Args:
            num_classes: Number of output classes
            img_size: Input image size
            patch_size: Size of image patches
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TinyVisionTransformer, self).__init__()
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification from class token
        x = self.norm(x[:, 0])
        x = self.dropout(x)
        x = self.head(x)
        
        return x

def create_model(model_name: str = 'custom', num_classes: int = 2, pretrained: bool = False, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for torchvision models)
        **kwargs: Additional arguments for model creation
    
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    if model_name == 'custom':
        # CustomCNN doesn't use pretrained
        return CustomCNN(num_classes=num_classes, **kwargs)
    
    elif model_name == 'resnet18':
        # Adapt ResNet18 for grayscale 28x28 images
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.resnet18(weights=weights)
        # Modify first layer for grayscale
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove maxpool to preserve spatial dimensions
        # Modify last layer for binary classification
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif model_name == 'vit' or model_name == 'vit-tiny':
        # ViT doesn't use pretrained in this implementation
        return TinyVisionTransformer(num_classes=num_classes, **kwargs)
    
    elif model_name == 'efficientnet' or model_name == 'efficientnet-b0':
        # Adapt EfficientNet-B0
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # Modify first layer for grayscale
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # Modify last layer for binary classification
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
