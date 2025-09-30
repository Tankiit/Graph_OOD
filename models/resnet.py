"""
ResNet Implementation for Graph-based Spectral OOD Detection

This module implements ResNet architecture with modifications for OOD detection.
Reference: Deep Residual Learning for Image Recognition (CVPR 2016)

Key features:
1. Residual blocks with skip connections
2. Feature extraction capabilities for spectral analysis
3. Configurable depth and architecture variants
4. Support for different input sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for deeper ResNet"""
    
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet implementation
    
    Args:
        block: Type of residual block (BasicBlock or Bottleneck)
        num_blocks (list): Number of blocks in each layer
        num_classes (int): Number of output classes
        input_size (int): Input image size (32 for CIFAR, 224 for ImageNet)
    """
    
    def __init__(self, block, num_blocks, num_classes=10, input_size=32):
        super(ResNet, self).__init__()
        
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.input_size = input_size
        
        self.in_planes = 64
        
        # Initial convolution
        if input_size == 32:  # CIFAR
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:  # ImageNet
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
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
        out = self.layer4(out)
        
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
        out = self.layer4(out)
        
        # Global pooling
        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        
        return features
    
    def get_feature_dim(self):
        """Get the dimension of extracted features"""
        return 512 * self.block.expansion
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'architecture': 'ResNet',
            'block_type': self.block.__name__,
            'num_blocks': self.num_blocks,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.get_feature_dim()
        }
        
        return info


# Predefined ResNet architectures
def ResNet18(num_classes=10, input_size=32):
    """ResNet-18: 18 layers with basic blocks"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_size=input_size)


def ResNet34(num_classes=10, input_size=32):
    """ResNet-34: 34 layers with basic blocks"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, input_size=input_size)


def ResNet50(num_classes=10, input_size=32):
    """ResNet-50: 50 layers with bottleneck blocks"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, input_size=input_size)


def ResNet101(num_classes=10, input_size=32):
    """ResNet-101: 101 layers with bottleneck blocks"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, input_size=input_size)


def ResNet152(num_classes=10, input_size=32):
    """ResNet-152: 152 layers with bottleneck blocks"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, input_size=input_size)


# Model registry for easy access
MODEL_REGISTRY = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
}


def create_resnet_model(model_name, num_classes=10, input_size=32):
    """
    Factory function to create ResNet models
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        input_size (int): Input image size
    
    Returns:
        ResNet: Configured model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown ResNet model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](num_classes=num_classes, input_size=input_size)


# Convenience function for the main training script
def ResNet(depth=50, num_classes=10, input_size=32):
    """
    Create ResNet model by depth
    
    Args:
        depth (int): Model depth (18, 34, 50, 101, 152)
        num_classes (int): Number of output classes
        input_size (int): Input image size
    
    Returns:
        ResNet: Configured model
    """
    depth_map = {
        18: 'resnet18',
        34: 'resnet34',
        50: 'resnet50',
        101: 'resnet101',
        152: 'resnet152'
    }
    
    if depth not in depth_map:
        raise ValueError(f"Unsupported ResNet depth: {depth}. Available: {list(depth_map.keys())}")
    
    return create_resnet_model(depth_map[depth], num_classes=num_classes, input_size=input_size)


if __name__ == '__main__':
    # Test the models
    models_to_test = [
        ('ResNet18', ResNet18),
        ('ResNet34', ResNet34),
        ('ResNet50', ResNet50),
    ]
    
    for name, model_fn in models_to_test:
        print(f"\nTesting {name}:")
        model = model_fn(num_classes=10, input_size=32)
        print(f"Model info: {model.get_model_info()}")
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        features = model.extract_features(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Feature dimension: {model.get_feature_dim()}")
