"""Test random_psd_operator."""

import pytest
from numpy.testing import assert_equal

from toqito.matrix_props import is_positive_semidefinite
from toqito.rand import random_psd_operator


@pytest.mark.parametrize(
    "dim, is_real",
    [
        # Test with a matrix of dimension 2.
        (2, True),
        # Test with a matrix of dimension 4.
        (4, False),
        # Test with a matrix of dimension 5.
        (5, False),
        # Test with a matrix of dimension 10.
        (10, True)
    ]
)
def test_random_psd_operator(dim, is_real):
    """Test for random_psd_operator function."""
    # Generate a random positive semidefinite operator.
    rand_psd_operator = random_psd_operator(dim, is_real)

    # Ensure the matrix has the correct shape.
    assert_equal(rand_psd_operator.shape, (dim, dim))

    # Check if the matrix is positive semidefinite.
    assert is_positive_semidefinite(rand_psd_operator), "Matrix should be positive semidefinite"
