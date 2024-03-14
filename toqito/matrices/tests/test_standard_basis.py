"""Test standard_basis."""

import numpy as np

from toqito.matrices import standard_basis


def test_standard_basis_2_d():
    """Return the standard basis in 2 dimensions."""
    basis = standard_basis(2)
    expected_basis = [np.array([1, 0]).reshape(-1, 1), np.array([0, 1]).reshape(-1, 1)]

    assert np.allclose(basis[0], expected_basis[0])
    assert np.allclose(basis[1], expected_basis[1])


def test_standard_basis_2_d_flatten():
    """Return the standard basis in 2 dimensions flattened."""
    basis = standard_basis(2, flatten=True)
    expected_basis = [np.array([1, 0]), np.array([0, 1])]

    assert np.allclose(basis[0], expected_basis[0])
    assert np.allclose(basis[1], expected_basis[1])
