import numpy as np
import pytest
from numpy.testing import assert_equal

from toqito.matrix_props import is_density, is_positive_semidefinite

from toqito.rand.random_psd_operator import random_psd_operator

@pytest.mark.parametrize(
    "dim",
    [
        # Test with a matrix of dimension 2.
        2,
        # Test with a matrix of dimension 4.
        4,
        # Test with a matrix of dimension 5.
        5,
        # Test with a matrix of dimension 10.
        10,
    ],
)
def test_random_psd_operator(dim):
    """Test for random_psd_operator function."""
    # Generate a random positive semi-definite operator.
    rand_psd_operator = random_psd_operator(dim)

    # Ensure the matrix has the correct shape.
    assert_equal(rand_psd_operator.shape, (dim, dim))

    # Check if the matrix is a valid density matrix
    assert is_density(rand_psd_operator), "Matrix should be a valid density matrix"

    # Check if the trace is 1
    assert np.isclose(np.trace(rand_psd_operator), 1), "Trace of the matrix should be 1"

    # Check if the matrix is positive semidefinite
    assert is_positive_semidefinite(rand_psd_operator), "Matrix should be positive semidefinite"