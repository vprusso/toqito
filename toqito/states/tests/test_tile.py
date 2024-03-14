"""Test tile."""

import numpy as np
import pytest

from toqito.states import tile

e_0, e_1, e_2 = np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])


@pytest.mark.parametrize(
    "tile_idx, expected_result",
    [
        # |\psi_0 \rangle = \frac{1}{\sqrt{2}} |0 \rangle \left(|0\rangle - |1\rangle \right).
        (0, 1 / np.sqrt(2) * np.kron(e_0, (e_0 - e_1))),
        # |\psi_1\rangle = \frac{1}{\sqrt{2}} \left(|0\rangle - |1\rangle \right) |2\rangle
        (1, 1 / np.sqrt(2) * np.kron((e_0 - e_1), e_2)),
        # |\psi_2\rangle = \frac{1}{\sqrt{2}} |2\rangle \left(|1\rangle - |2\rangle \right)
        (2, 1 / np.sqrt(2) * np.kron(e_2, (e_1 - e_2))),
        # |\psi_3\rangle = \frac{1}{\sqrt{2}} \left(|1\rangle - |2\rangle \right) |0\rangle
        (3, 1 / np.sqrt(2) * np.kron((e_1 - e_2), e_0)),
        # |\psi_4\rangle = \frac{1}{3} \left(|0\rangle + |1\rangle +
        # |2\rangle)\right) \left(|0\rangle + |1\rangle + |2\rangle.
        (4, 1 / 3 * np.kron((e_0 + e_1 + e_2), (e_0 + e_1 + e_2))),
    ],
)
def test_tile(tile_idx, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_array_equal(tile(tile_idx), expected_result)


@pytest.mark.parametrize(
    "tile_idx",
    [
        # Invalid idx.
        (5, 1 / np.sqrt(2) * np.kron(e_0, (e_0 - e_1))),
    ],
)
def test_tile_invalid(tile_idx):
    """Ensures that an integer above 4 is error-checked."""
    with np.testing.assert_raises(ValueError):
        tile(tile_idx)
