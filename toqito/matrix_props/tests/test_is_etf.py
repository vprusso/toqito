"""Test is_etf."""

import numpy as np
from toqito.matrix_props import is_etf

def test_is_etf_valid_etf():
    """Test that a known valid ETF matrix returns True."""
    mat_valid_etf = np.array([
        [1, -1/2, -1/2],
        [0,  np.sqrt(3)/2, -np.sqrt(3)/2]
    ])
    np.testing.assert_equal(is_etf(mat_valid_etf), True)

def test_is_etf_non_unit_norm_columns():
    """Test that a matrix with non-unit norm columns returns False."""
    mat_non_unit_norm = np.array([
        [2, 1,  3],
        [0, 2, -1]
    ])
    np.testing.assert_equal(is_etf(mat_non_unit_norm), False)

def test_is_etf_unit_norm_but_not_equiangular():
    """Test that a matrix whose columns have unit norm but are not equiangular returns False."""
    # Each column has unit norm but the off-diagonal of the Gram matrix isn't constant.
    mat_unit_but_not_equiangular = np.array([
        [1, 0, 1/np.sqrt(2)],
        [0, 1, 1/np.sqrt(2)]
    ])
    np.testing.assert_equal(is_etf(mat_unit_but_not_equiangular), False)

def test_is_etf_equiangular_but_not_tight():
    """Test that a matrix with equiangular columns but not tight frame returns False."""
    # Columns are identical, so off-diagonal inner product is 1 (hence equiangular),
    # but fails the tight frame condition A A* = (ncols/nrows) I.
    c1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    c2 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    mat_equiangular_not_tight = np.column_stack((c1, c2))
    np.testing.assert_equal(is_etf(mat_equiangular_not_tight), False)
