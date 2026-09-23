import pytest
import torch
import torch.nn as nn
from src.models.base import BaseModelWrapper
from src.models.hook_manager import HookManager
from src.data.contrastive_dataset import ContrastiveDataset
from src.data.text_datasets import TextClassificationDataset


class MockSimpleNet(BaseModelWrapper):
    """Simple 3-layer neural network implementing BaseModelWrapper for unit tests."""

    def __init__(self, in_features: int = 16, hidden_dim: int = 32, num_classes: int = 2):
        super(BaseModelWrapper, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.layer0 = nn.Linear(in_features, hidden_dim)
        self.relu0 = nn.ReLU()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, num_classes)

        self.hook_manager = HookManager(self)

    def get_layer_names(self):
        return ["layer0", "layer1", "layer2"]

    def get_layer_module(self, layer_name: str) -> nn.Module:
        return self.hook_manager.get_submodule(layer_name)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h0 = self.relu0(self.layer0(x))
        h1 = self.relu1(self.layer1(h0))
        out = self.layer2(h1)
        return out

    def extract_representations(self, batch, layer_names=None, pooling="last", **kwargs):
        target_layers = layer_names or self.get_layer_names()
        if isinstance(batch, (list, str)):
            # Deterministic mock tensor from text strings
            texts = [batch] if isinstance(batch, str) else batch
            tensors = []
            for t in texts:
                torch.manual_seed(abs(hash(t)) % 10000)
                tensors.append(torch.randn(1, self.in_features))
            x = torch.cat(tensors, dim=0)
        elif isinstance(batch, dict) and "input_ids" in batch:
            x = batch["input_ids"].float()
        else:
            x = batch

        with torch.no_grad():
            with self.hook_manager.capture_activations(target_layers) as storage:
                _ = self.forward(x)

        return {name: storage[name][0] for name in target_layers}


@pytest.fixture
def mock_model():
    return MockSimpleNet()


@pytest.fixture
def sample_contrastive_dataset():
    return ContrastiveDataset.create_synthetic_ood_pairs(num_samples=16)


@pytest.fixture
def sample_classification_dataset():
    return TextClassificationDataset.create_synthetic_classification(num_samples_per_class=10, num_classes=2)
