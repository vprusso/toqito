"""Test domino."""

import numpy as np
import pytest

from toqito.states import domino

e_0, e_1, e_2 = np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])


@pytest.mark.parametrize(
    "idx, expected_result",
    [
        # Domino with index = 0.
        (0, np.kron(e_1, e_1)),
        # Domino with index = 1.
        (1, np.kron(e_0, 1 / np.sqrt(2) * (e_0 + e_1))),
        # Domino with index = 2.
        (2, np.kron(e_0, 1 / np.sqrt(2) * (e_0 - e_1))),
        # Domino with index = 3.
        (3, np.kron(e_2, 1 / np.sqrt(2) * (e_1 + e_2))),
        # Domino with index = 4.
        (4, np.kron(e_2, 1 / np.sqrt(2) * (e_1 - e_2))),
        # Domino with index = 5.
        (5, np.kron(1 / np.sqrt(2) * (e_1 + e_2), e_0)),
        # Domino with index = 6.
        (6, np.kron(1 / np.sqrt(2) * (e_1 - e_2), e_0)),
        # Domino with index = 7.
        (7, np.kron(1 / np.sqrt(2) * (e_0 + e_1), e_2)),
        # Domino with index = 8.
        (8, np.kron(1 / np.sqrt(2) * (e_0 - e_1), e_2)),
    ],
)
def test_domino(idx, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_array_equal(domino(idx), expected_result)


def test_domino_invalid_index():
    """Tests for invalid index input."""
    with np.testing.assert_raises(ValueError):
        domino(9)
