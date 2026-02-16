"""
WideResNet Implementation from SAL Repository
This matches the exact architecture used in the SAL paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet implementation from SAL repository

    Args:
        depth: Number of layers (should be 6n+4 for some n)
        num_classes: Number of output classes
        widen_factor: Width multiplier for the network
        dropRate: Dropout rate
    """

    def __init__(self, depth, num_classes=10, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)

        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)

        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)

        self.nChannels = nChannels[3]

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def extract_features(self, x):
        """Extract features before the final classification layer"""
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out

    def intermediate_forward(self, x, layer_index):
        """Forward pass to intermediate layer (for compatibility)"""
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def feature_list(self, x):
        """Return features from multiple layers (for compatibility)"""
        out_list = []
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), out_list


def wrn_28_10(num_classes=10, dropout_rate=0.0):
    """WRN-28-10: 28 layers with widen factor 10"""
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=dropout_rate)


def wrn_40_2(num_classes=10, dropout_rate=0.0):
    """WRN-40-2: 40 layers with widen factor 2"""
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=dropout_rate)


def wrn_16_8(num_classes=10, dropout_rate=0.0):
    """WRN-16-8: 16 layers with widen factor 8"""
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=8, dropRate=dropout_rate)


# Model registry for easy access
MODEL_REGISTRY = {
    'wrn_28_10': wrn_28_10,
    'wrn_40_2': wrn_40_2,
    'wrn_16_8': wrn_16_8,
}


def create_wrn_model(model_name, num_classes=10, dropout_rate=0.0):
    """
    Factory function to create WRN models

    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        dropout_rate: Dropout rate

    Returns:
        WideResNet: Configured model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown WRN model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_name](num_classes=num_classes, dropRate=dropout_rate)


if __name__ == '__main__':
    # Test the model
    model = wrn_40_2(num_classes=10)
    print("Model created successfully!")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    features = model.extract_features(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Feature dimension: {model.nChannels}")
