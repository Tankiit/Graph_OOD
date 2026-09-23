import pytest
import torch
from src.steering.base import SteeringVector
from src.steering.extractor import (
    MeanDiffExtractor,
    PCAExtractor,
    MassMeanExtractor,
    ProbeExtractor,
    extract_steering_vectors,
)


def test_steering_vector_serialization(tmp_path):
    vecs = {
        "layer0": torch.randn(32),
        "layer1": torch.randn(32)
    }
    sv = SteeringVector(vectors=vecs, concept="test_concept", method="mean_difference")
    save_file = tmp_path / "test_vec.pt"
    sv.save(save_file)

    loaded = SteeringVector.load(save_file)
    assert loaded.concept == "test_concept"
    assert loaded.method == "mean_difference"
    assert "layer0" in loaded
    assert torch.allclose(loaded["layer0"], vecs["layer0"])


def test_extractors_numerical():
    torch.manual_seed(42)
    pos_acts = torch.randn(50, 16) + 2.0
    neg_acts = torch.randn(50, 16) - 2.0

    # Mean difference
    caa = MeanDiffExtractor()
    v_caa = caa.extract_from_activations(pos_acts, neg_acts)
    assert v_caa.shape == (16,)
    assert (v_caa > 0).all()

    # PCA
    pca = PCAExtractor()
    v_pca = pca.extract_from_activations(pos_acts, neg_acts)
    assert v_pca.shape == (16,)

    # LDA
    lda = MassMeanExtractor()
    v_lda = lda.extract_from_activations(pos_acts, neg_acts)
    assert v_lda.shape == (16,)

    # Probe
    probe = ProbeExtractor(epochs=10)
    v_probe = probe.extract_from_activations(pos_acts, neg_acts)
    assert v_probe.shape == (16,)


def test_extract_steering_vectors_pipeline(mock_model, sample_contrastive_dataset):
    sv = extract_steering_vectors(
        model=mock_model,
        dataset=sample_contrastive_dataset,
        method="mean_difference",
        batch_size=4,
        normalize=True,
        show_progress=False
    )
    assert isinstance(sv, SteeringVector)
    assert "layer0" in sv
    assert "layer1" in sv
    # Check normalized unit length
    norm = torch.norm(sv["layer0"], p=2).item()
    assert abs(norm - 1.0) < 1e-4
