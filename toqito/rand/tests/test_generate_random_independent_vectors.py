"""Test generate_random_independent_vectors."""

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises

from toqito.rand.generate_random_independent_vectors import generate_random_independent_vectors


@pytest.mark.parametrize(
    "num_vecs,dim",
    [
        # Test two two-dimensional vectors.
        (2, 2),
        # Test four four-dimensional vectors.
        (4, 4),
        # Test ten ten-dimensional vectors.
        (10, 10),
        # Test two five-dimensional vectors.
        (2, 5),
        # Test two ten-dimensional vectors.
        (2, 10),
    ],
)
def test_generate_random_independent_vectors(num_vecs, dim):
    """Test for generate_random_independent_vectors function."""
    linear_indep = generate_random_independent_vectors(num_vecs, dim)

    # verify the matrix has the correct dimensions
    assert_equal(linear_indep.shape, (dim, num_vecs))

    # Verify the matrix is real.
    assert_equal(np.isreal(linear_indep).all(), True)

    # verify the vectors are linearaly independent
    # by confirming the rank of the vector space
    # is equivalent to the number of vectors generated
    assert_equal(np.linalg.matrix_rank(linear_indep) == num_vecs, True)


@pytest.mark.parametrize(
    "n,m",
    [
        # Test with a matrix of dimension 2.
        (3, 2),
        # Test with a matrix of higher dimension.
        (20, 15),
    ],
)
def test_generate_random_independent_vectors_failure(n, m):
    """Test for generate_random_independent_vectors function; should fail."""
    assert_raises(ValueError, generate_random_independent_vectors, n, m)
