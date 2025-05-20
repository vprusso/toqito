"""Test generate_random_independent_vectors."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises

from toqito.rand.generate_random_independent_vectors import generate_random_independent_vectors
from toqito.matrix_props.is_linearly_independent import is_linearly_independent 


@pytest.mark.parametrize(
    "n,m",
    [
        # Test with a matrix of dimension 2.
        (2,2),
        # Test with a matrix of higher dimension.
        (4,4),
        # Test with yet another higher dimension.
        (10,10),
        # Test with yet another higher dimension.
        (2,5),
        (2,10),
    ],
)

def test_random_circulant_gram_matrix(n,m):
    linear_indep = generate_random_independent_vectors(n,m)
    
    
    # verify the matrix has the correct dimensions 
    assert_equal(linear_indep.shape, (n, m))


    # verify the matrix is real 
    assert_equal(np.isreal(linear_indep).all(), True)

    # verify the vectors are linearaly independent 
    # by checking that the corresponding matrix is invertible,
    # which is iff w the determinant being non-zero 
    assert_equal(np.linalg.det(linear_indep) == 0, False)


@pytest.mark.parametrize(
    "n,m",
    [
        # Test with a matrix of dimension 2.
        (3,2),
        # Test with a matrix of higher dimension.
        (20,15),
    ],
)

def test_random_circulant_gram_matrix_failure(n,m):
    assert_raises(ValueError,generate_random_independent_vectors,n,m)
    

