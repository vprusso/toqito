"""Test is_absolutely_k_incoherent."""

import numpy as np
import pytest

from toqito.matrix_props import is_absolutely_k_incoherent


@pytest.mark.parametrize(
    "mat, k, expected",
    [
        # 2x2 maximally mixed state.
        (np.eye(2) / 2, 1, True),
        # k == n (trivial).
        (np.eye(2) / 2, 2, True),
        # 2x2 pure state (non-maximally mixed density matrix).
        (np.array([[1, 0], [0, 0]]), 1, False),
        # k == n (trivial).
        (np.array([[1, 0], [0, 0]]), 2, True),
        # 2x2 non-density matrix (not PSD).
        (np.array([[1, 2], [2, 1]]), 1, False),
        # k == n (trivial).
        (np.array([[1, 2], [2, 1]]), 2, True),
        # 3x3 example matrix (as provided by the user).
        (np.array([[2, 1, 2], [1, 2, -1], [2, -1, 5]]), 1, False),
        # k == n (trivial).
        (np.array([[2, 1, 2], [1, 2, -1], [2, -1, 5]]), 3, True),
        # k > n (trivial).
        (np.array([[2, 1, 2], [1, 2, -1], [2, -1, 5]]), 4, True),
        # 3x3 maximally mixed state.
        (np.eye(3) / 3, 1, True),
        # Determined via eigenvalue condition.
        (np.eye(3) / 3, 2, True),
        # k == n (trivial).
        (np.eye(3) / 3, 3, True),
        # 4x4 custom density matrix: diag(0.5, 0.25, 0.125, 0.125).
        (np.diag([0.5, 0.25, 0.125, 0.125]), 1, False),
        (np.diag([0.5, 0.25, 0.125, 0.125]), 2, False),
        (np.diag([0.5, 0.25, 0.125, 0.125]), 3, True),
        # k == n (trivial).
        (np.diag([0.5, 0.25, 0.125, 0.125]), 4, True),
    ],
)
def test_is_absolutely_k_incoherent(mat, k, expected):
    """Test that is_absk_incoh returns the correct boolean value on valid inputs."""
    np.testing.assert_equal(is_absolutely_k_incoherent(mat, k), expected)


def test_is_absolutey_k_incoherent_non_square():
    """Test that passing a non-square matrix raises a ValueError."""
    # Construct a non-square matrix.
    mat = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        # np.linalg.eigvalsh in is_absolutely_k_incoherent will raise a ValueError for non-square matrices.
        is_absolutely_k_incoherent(mat, 1)
