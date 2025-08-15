"""Test basis."""

import numpy as np
import pytest

from toqito.states import basis


@pytest.mark.parametrize(
    "dim, pos, expected_result",
    [
        # Test for `|0>`.
        (2, 0, np.array([[1], [0]])),
        # Test for `|1>`.
        (2, 1, np.array([[0], [1]])),
        # Test for `|0000>`.
        (4, 0, np.array([[1], [0], [0], [0]])),
    ],
)
def test_basis(dim, pos, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_array_equal(basis(dim, pos), expected_result)


@pytest.mark.parametrize(
    "dim, pos",
    [
        # Test for dim = pos.
        (4, 4),
        # Test for pos < 0.
        (4, -1),
    ],
)
def test_basis_invalid_dim(dim, pos):
    """Tests for invalid dimension inputs."""
    with np.testing.assert_raises(ValueError):
        basis(dim, pos)
