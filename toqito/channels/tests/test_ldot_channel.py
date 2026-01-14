"""Tests for the local diagonal orthogonal twirl channel."""

import numpy as np
import pytest

from toqito.channels import ldot_channel


def test_ldot_channel_stable_on_diagonal() -> None:
    """Diagonal states should be fixed points of the LDOT channel."""
    diagonal_state = np.diag([1, 2, 3, 4]).astype(np.complex128)
    projected = ldot_channel(diagonal_state)
    assert np.allclose(projected, diagonal_state)


def test_ldot_channel_efficient_matches_bruteforce() -> None:
    """The efficient implementation should match the brute-force averaging."""
    rng = np.random.default_rng(0)
    mat = rng.random((4, 4)) + 1j * rng.random((4, 4))
    fast = ldot_channel(mat, efficient=True)
    slow = ldot_channel(mat, efficient=False)
    assert np.allclose(fast, slow)


def test_ldot_channel_raises_on_non_square() -> None:
    """The channel requires a square matrix as input."""
    with pytest.raises(ValueError, match="Input matrix must be square"):
        ldot_channel(np.ones((2, 3)))


def test_ldot_channel_raises_on_non_perfect_square_dimension() -> None:
    """Dimensions that are square but not bipartite squares should raise."""
    nearly_square = np.eye(3)
    with pytest.raises(ValueError, match="perfect square"):
        ldot_channel(nearly_square)
