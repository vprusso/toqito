"""Tests for validate_bipartite_dim."""

import numpy as np
import pytest

from toqito.matrix_ops import validate_bipartite_dim


def test_validate_bipartite_dim_none_perfect_square():
    """With dim=None and a perfect-square dimension, equal subsystems are inferred."""
    np.testing.assert_array_equal(validate_bipartite_dim(np.eye(9), None), np.array([3, 3]))


def test_validate_bipartite_dim_scalar():
    """A scalar dim is taken as the first subsystem dimension."""
    np.testing.assert_array_equal(validate_bipartite_dim(np.eye(6), 2), np.array([2, 3]))


def test_validate_bipartite_dim_pair():
    """A pair of dimensions whose product matches is returned as-is."""
    np.testing.assert_array_equal(validate_bipartite_dim(np.eye(6), [2, 3]), np.array([2, 3]))


@pytest.mark.parametrize(
    "rho, dim, msg",
    [
        (np.eye(6), None, "Cannot infer bipartite subsystem dimensions"),
        (np.eye(6), 4, "positive divisor"),
        (np.eye(6), 0, "positive divisor"),
        (np.eye(6), [2, 2, 2], "exactly two subsystem dimensions"),
        (np.eye(6), [2.0, 3.0], "integer subsystem dimensions"),
        (np.eye(6), [-2, -3], "must be positive"),
        (np.eye(6), [2, 2], "product of `dim`"),
    ],
)
def test_validate_bipartite_dim_invalid(rho, dim, msg):
    """Invalid dimension specifications raise a clear ValueError."""
    with pytest.raises(ValueError, match=msg):
        validate_bipartite_dim(rho, dim)
