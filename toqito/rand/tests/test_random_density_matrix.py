"""Test random_density_matrix."""

import numpy as np
import pytest

from toqito.matrix_props import is_density, is_positive_semidefinite
from toqito.rand import random_density_matrix


@pytest.mark.parametrize(
    "dim, is_real, k_param, distance_metric",
    [
        # Generate random non-real density matrix.
        (2, False, None, "haar"),
        # Generate random real density matrix.
        (2, True, None, "haar"),
        # Random non-real density matrix according to Bures metric.
        (2, False, None, "bures"),
        # Generate random non-real density matrix all params.
        (2, True, 2, "haar"),
    ],
)
def test_random_density(dim, is_real, k_param, distance_metric):
    """Test function works as expected for a valid input."""
    if k_param == dim:
        mat = random_density_matrix(dim=dim, is_real=is_real, k_param=k_param, distance_metric=distance_metric)
        np.testing.assert_equal(is_density(mat), True)


@pytest.mark.parametrize(
    "dim, is_real, distance_metric", [(2, False, "haar"), (2, True, "haar"), (3, False, "bures"), (3, True, "bures")]
)
def test_random_density_matrix(dim, is_real, distance_metric):
    """Test function output is real or complex."""
    dm = random_density_matrix(dim, is_real, distance_metric=distance_metric)

    # Check if the matrix is a valid density matrix
    assert is_density(dm), "Matrix should be a valid density matrix"

    # Check if the matrix is positive semidefinite
    assert is_positive_semidefinite(dm), "Matrix should be positive semidefinite"

    # Check if the trace is 1
    assert np.isclose(np.trace(dm), 1), "Trace of the matrix should be 1"

    # Check if the matrix is real or complex as expected
    if is_real:
        assert np.all(np.isreal(dm)), "Matrix should be real"
    else:
        assert np.any(np.iscomplex(dm)), "Matrix should be complex"
