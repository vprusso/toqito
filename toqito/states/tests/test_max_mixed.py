"""Test max_mixed."""

import numpy as np

from toqito.states import max_mixed


def test_max_mixed_dim_2_full():
    """Generate full 2-dimensional maximally mixed state."""
    expected_res = 1 / 2 * np.array([[1, 0], [0, 1]])
    res = max_mixed(2, is_sparse=False)
    np.testing.assert_allclose(res, expected_res)


def test_max_mixed_dim_2_sparse():
    """Generate sparse 2-dimensional maximally mixed state."""
    expected_res = 1 / 2 * np.array([[1, 0], [0, 1]])
    res = max_mixed(2, is_sparse=True).toarray()
    np.testing.assert_allclose(res, expected_res)
