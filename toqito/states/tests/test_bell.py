"""Test bell."""

import numpy as np
import pytest

from toqito.states import bell

e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])


@pytest.mark.parametrize(
    "bell_idx, expected_result",
    [
        # 1/sqrt(2) * (|00> + |11>)
        (0, 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))),
        # 1/sqrt(2) * (|00> - |11>)
        (1, 1 / np.sqrt(2) * (np.kron(e_0, e_0) - np.kron(e_1, e_1))),
        # 1/sqrt(2) * (|01> + |10>)
        (2, 1 / np.sqrt(2) * (np.kron(e_0, e_1) + np.kron(e_1, e_0))),
        # 1/sqrt(2) * (|01> - |10>)
        (3, 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))),
    ],
)
def test_bell(bell_idx, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_array_equal(bell(bell_idx), expected_result)


@pytest.mark.parametrize(
    "bell_idx",
    [
        # Invalid index.
        (4),
        # Invalid index.
        (10),
    ],
)
def test_bell_invalid(bell_idx):
    """Ensures that an integer above 3 is error-checked."""
    with np.testing.assert_raises(ValueError):
        bell(bell_idx)
