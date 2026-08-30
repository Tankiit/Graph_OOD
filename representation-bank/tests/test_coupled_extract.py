import numpy as np
import pytest
import torch

from repbank.coupled_extract import fraction_indices, verify_checksum


def test_fraction_indices():
    assert fraction_indices([0.2, 0.5, 0.8], 32) == [6, 16, 25]


def test_answer_logprob_indexing():
    # For token at absolute position p, its probability is stored at shifted p-1.
    logits = torch.full((1, 4, 5), -10.0)
    ids = torch.tensor([[0, 1, 2, 3]])
    for position in range(3):
        logits[0, position, ids[0, position + 1]] = 10.0
    shifted = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    answer_start = 2
    assert torch.all(shifted[0, answer_start - 1:] > -1e-5)


def test_verify_checksum_detects_change(tmp_path):
    path = tmp_path / "bank.npz"
    np.savez(path, x=np.ones(2))
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".npz.sha256").write_text(f"{digest}  bank.npz\n")
    verify_checksum(path)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksum(path)
