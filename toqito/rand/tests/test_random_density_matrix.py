"""Test random_density_matrix."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

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


@pytest.mark.parametrize(
    "dim, is_real, k_param, distance_metric, expected",
    [
        # Generate random non-real density matrix.
        (
            2,
            False,
            None,
            "haar",
            np.array([[0.22986343 + 0.0j, 0.13365203 - 0.20624294j], [0.13365203 + 0.20624294j, 0.77013657 + 0.0j]]),
        ),
        # Generate random real density matrix.
        (2, True, None, "haar", np.array([[0.13871939, 0.33280544], [0.33280544, 0.86128061]])),
        # Random non-real density matrix according to Bures metric.
        (
            2,
            False,
            None,
            "bures",
            np.array([[0.24980937 + 0.0j, 0.19650908 - 0.22450506j], [0.19650908 + 0.22450506j, 0.75019063 + 0.0j]]),
        ),
        # Generate random non-real density matrix all params.
        (2, True, 2, "haar", np.array([[0.13871939, 0.33280544], [0.33280544, 0.86128061]])),
    ],
)
def test_seed(dim, is_real, k_param, distance_metric, expected):
    """Test that the function produces the expected output using a seed."""
    dm = random_density_matrix(dim, is_real, k_param=k_param, distance_metric=distance_metric, seed=124)
    assert_allclose(dm, expected)


@pytest.mark.parametrize("is_real", [False, True])
def test_random_density_matrix_mean_is_maximally_mixed(is_real):
    """Averaging Hilbert-Schmidt distributed states approaches I/d.

    With the previous uniform (rather than Gaussian) sampling of the Ginibre matrix the entries were biased toward
    positive values, so the average had large positive off-diagonal entries and did not converge to I/d.
    """
    dim, n_samples = 2, 4000
    acc = np.zeros((dim, dim), dtype=complex)
    for s in range(n_samples):
        acc += random_density_matrix(dim, is_real=is_real, seed=s)
    np.testing.assert_allclose(acc / n_samples, np.eye(dim) / dim, atol=0.02)
