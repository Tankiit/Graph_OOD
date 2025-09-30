"""
WideResNet Implementation for Graph-based Spectral OOD Detection

This module implements WideResNet (WRN) architecture with modifications for OOD detection.
Reference: Wide Residual Networks (BMVC 2016)

Key features:
1. Wide residual blocks with dropout
2. Feature extraction capabilities for spectral analysis
3. Configurable depth and width
4. Support for different input sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    """Basic residual block for WideResNet"""
    
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class WideResNet(nn.Module):
    """
    WideResNet implementation
    
    Args:
        depth (int): Number of layers (should be 6n+4 for some n)
        widen_factor (int): Width multiplier for the network
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        input_size (int): Input image size (32 for CIFAR, 224 for ImageNet)
    """
    
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout_rate=0.0, input_size=32):
        super(WideResNet, self).__init__()
        
        self.depth = depth
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        
        # Calculate number of blocks per group
        assert (depth - 4) % 6 == 0, 'WideResNet depth should be 6n+4'
        n = (depth - 4) // 6
        
        # Calculate number of channels
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]
        
        # Initial convolution
        if input_size == 32:  # CIFAR
            self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, 
                                   padding=1, bias=False)
        else:  # ImageNet
            self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=7, stride=2, 
                                   padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(nStages[0])
        
        # Residual groups
        self.layer1 = self._make_layer(BasicBlock, nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(BasicBlock, nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(BasicBlock, nStages[3], n, stride=2, dropout_rate=dropout_rate)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nStages[3], num_classes)
        
        # Initialize in_planes for residual connections
        self.in_planes = nStages[0]
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        """Create a layer with multiple residual blocks"""
        layers = []
        layers.append(block(self.in_planes, planes, stride, dropout_rate))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1, dropout_rate))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.input_size != 32:  # ImageNet
            out = self.maxpool(out)
        
        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global pooling and classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
    
    def extract_features(self, x):
        """Extract features before the final classification layer"""
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.input_size != 32:  # ImageNet
            out = self.maxpool(out)
        
        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global pooling
        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        
        return features
    
    def get_feature_dim(self):
        """Get the dimension of extracted features"""
        if self.input_size == 32:  # CIFAR
            return 64 * self.widen_factor
        else:  # ImageNet
            return 64 * self.widen_factor
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'architecture': 'WideResNet',
            'depth': self.depth,
            'widen_factor': self.widen_factor,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'input_size': self.input_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.get_feature_dim()
        }
        
        return info


def wrn_28_10(num_classes=10, dropout_rate=0.0, input_size=32):
    """WRN-28-10: 28 layers with widen factor 10"""
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, 
                      dropout_rate=dropout_rate, input_size=input_size)


def wrn_40_4(num_classes=10, dropout_rate=0.0, input_size=32):
    """WRN-40-4: 40 layers with widen factor 4"""
    return WideResNet(depth=40, widen_factor=4, num_classes=num_classes, 
                      dropout_rate=dropout_rate, input_size=input_size)


def wrn_16_8(num_classes=10, dropout_rate=0.0, input_size=32):
    """WRN-16-8: 16 layers with widen factor 8"""
    return WideResNet(depth=16, widen_factor=8, num_classes=num_classes, 
                      dropout_rate=dropout_rate, input_size=input_size)


# Model registry for easy access
MODEL_REGISTRY = {
    'wrn_28_10': wrn_28_10,
    'wrn_40_4': wrn_40_4,
    'wrn_16_8': wrn_16_8,
}


def create_wrn_model(model_name, num_classes=10, dropout_rate=0.0, input_size=32):
    """
    Factory function to create WRN models
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate
        input_size (int): Input image size
    
    Returns:
        WideResNet: Configured model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown WRN model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](num_classes=num_classes, dropout_rate=dropout_rate, input_size=input_size)


if __name__ == '__main__':
    # Test the model
    model = wrn_28_10(num_classes=10, input_size=32)
    print("Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    features = model.extract_features(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Feature dimension: {model.get_feature_dim()}")
