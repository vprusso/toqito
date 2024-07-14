"""Test random_circulant_gram_matrix."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_equal

from toqito.rand.random_circulant_gram_matrix import random_circulant_gram_matrix


@pytest.mark.parametrize(
    "dim",
    [
        # Test with a matrix of dimension 2.
        2,
        # Test with a matrix of higher dimension.
        4,
        # Test with another higher dimension.
        5,
        # Test with yet another higher dimension.
        10,
    ],
)
def test_random_circulant_gram_matrix(dim):
    """Test for random_circulant_gram_matrix function."""
    # Generate a random circulant Gram matrix.
    circulant_matrix = random_circulant_gram_matrix(dim)

    # Ensure the matrix has the correct shape.
    assert_equal(circulant_matrix.shape, (dim, dim))

    # Check that the matrix is symmetric.
    assert_array_almost_equal(circulant_matrix, circulant_matrix.T)

    # Check that the matrix is real.
    assert_equal(np.isreal(circulant_matrix).all(), True)

    # Check that the matrix is positive semi-definite by verifying
    # all eigenvalues are non-negative.
    eigenvalues = np.linalg.eigvalsh(circulant_matrix)
    assert_array_almost_equal((eigenvalues >= 0), True)
