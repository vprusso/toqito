"""Test standard_basis."""

import numpy as np
import pytest

from toqito.matrices import standard_basis


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_standard_basis(dim):
    """Return the standard basis in the requested dimension."""
    basis = standard_basis(dim)
    expected_basis = [vector.reshape(-1, 1) for vector in np.eye(dim)]

    assert all(np.array_equal(actual, expected) for actual, expected in zip(basis, expected_basis))


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_standard_basis_flatten(dim):
    """Return the flattened standard basis in the requested dimension."""
    basis = standard_basis(dim, flatten=True)
    expected_basis = list(np.eye(dim))

    assert all(np.array_equal(actual, expected) for actual, expected in zip(basis, expected_basis))
