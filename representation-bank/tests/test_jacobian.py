import numpy as np

from repbank.jacobian import dispersion, fit_readout, paired_indices


def test_pair_selection_and_readout():
    rng = np.random.default_rng(3)
    pair_ids = np.repeat(np.arange(10), 2).astype(str)
    roles = np.tile([0, 1], 10)
    states = rng.normal(size=(20, 8)).astype(np.float32)
    states[roles == 1, 0] += 3
    correct, wrong = paired_indices(pair_ids, roles)
    assert len(correct) == len(wrong) == 10
    fitted = fit_readout(states, pair_ids, roles, epochs=100)
    assert fitted.train_pairs == 8
    assert fitted.heldout_pairs == 2


def test_dispersion_constant_field():
    report = dispersion(np.tile(np.array([[1.0, 0.0]]), (5, 1)))
    assert report["median_degrees"] == 0
    assert report["resultant_length"] == 1
