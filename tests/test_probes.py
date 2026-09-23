import pytest
import torch
from src.models.probes import LinearProbe, MLPProbe, EnergyProbe, MahalanobisDetector
from src.training.probe_trainer import ProbeTrainer


def test_linear_probe():
    probe = LinearProbe(input_dim=16, num_classes=2)
    x = torch.randn(8, 16)
    out = probe(x)
    assert out.shape == (8, 2)
    dir_vec = probe.get_direction(class_idx=1, normalize=True)
    assert dir_vec.shape == (16,)
    assert abs(torch.norm(dir_vec, p=2).item() - 1.0) < 1e-4


def test_energy_probe():
    energy_probe = EnergyProbe(input_dim=16, num_classes=3, temperature=1.0)
    x = torch.randn(4, 16)
    energy = energy_probe.compute_energy(x)
    assert energy.shape == (4,)
    assert not torch.isnan(energy).any()


def test_mahalanobis_detector():
    torch.manual_seed(42)
    c0 = torch.randn(50, 8) + 3.0
    c1 = torch.randn(50, 8) - 3.0
    features = torch.cat([c0, c1], dim=0)
    labels = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])

    detector = MahalanobisDetector(num_classes=2, feature_dim=8)
    detector.fit(features, labels)

    # In-distribution test points
    test_id = torch.randn(10, 8) + 3.0
    # Far OOD test points
    test_ood = torch.randn(10, 8) * 10.0 + 50.0

    scores_id = detector.score(test_id)
    scores_ood = detector.score(test_ood)

    assert scores_id.mean() > scores_ood.mean()


def test_probe_trainer():
    trainer = ProbeTrainer(probe_type="linear", epochs=10)
    feats = torch.cat([torch.randn(30, 8) + 2.0, torch.randn(30, 8) - 2.0], dim=0)
    labels = torch.cat([torch.ones(30, dtype=torch.long), torch.zeros(30, dtype=torch.long)])

    probe, acc = trainer.train_layer_probe(feats, labels)
    assert acc > 0.8
