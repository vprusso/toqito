"""Test random_circulant_gram."""
import numpy as np
import pytest

from toqito.rand.random_circulant_gram import random_circulant_gram


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
def test_random_circulant_gram(dim):
    """Test for random_circulant_gram function."""
    # Generate a random circulant Gram matrix.
    circulant_matrix = random_circulant_gram(dim)

    # Ensure the matrix has the correct shape.
    assert circulant_matrix.shape == (dim, dim)

    # Check that the matrix is symmetric.
    assert np.allclose(circulant_matrix, circulant_matrix.T)

    # Check that the matrix is real.
    assert np.all(np.isreal(circulant_matrix))

    # Check that the matrix is positive semi-definite.
    # This is done by verifying all eigenvalues are non-negative.
    eigenvalues = np.linalg.eigvalsh(circulant_matrix)
    assert np.all(eigenvalues >= 0)
