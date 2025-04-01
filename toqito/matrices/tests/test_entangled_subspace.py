"""Test entangled_subspace."""
import numpy as np
import pytest
from scipy import sparse

from toqito.matrices import entangled_subspace


def test_entangled_subspace_dim_2_local_dim_3():
    """Test entangled subspace with dim=2, local_dim=3."""
    dim, local_dim = 2, 3
    result = entangled_subspace(dim, local_dim)
    
    # Check dimensions
    assert result.shape == (9, 2)  # 9 = 3*3, product of local dimensions
    
    # Check that the columns are linearly independent (basis of subspace)
    rank = np.linalg.matrix_rank(result.toarray())
    assert rank == dim
    
    # Additional tests could be added to verify entanglement properties
    # of the resulting subspace vectors


def test_entangled_subspace_dim_3_unequal_local_dims():
    """Test entangled subspace with unequal local dimensions."""
    dim, local_dim = 3, [3, 4]
    result = entangled_subspace(dim, local_dim)
    
    # Check dimensions
    assert result.shape == (12, 3)  # 12 = 3*4, product of local dimensions
    
    # Check that the columns are linearly independent
    rank = np.linalg.matrix_rank(result.toarray())
    assert rank == dim


def test_entangled_subspace_with_r_2():
    """Test with higher entanglement parameter r=2."""
    dim, local_dim = 1, 4
    r = 2
    result = entangled_subspace(dim, local_dim, r)
    
    # Check dimensions
    assert result.shape == (16, 1)  # 16 = 4*4
    
    # Check that the column is non-zero
    assert np.linalg.norm(result.toarray()) > 0


def test_entangled_subspace_invalid_dim():
    """Test that an error is raised when dim is too large."""
    dim, local_dim, r = 10, 3, 1
    
    # (3-1)*(3-1) = 4, so dim=10 is too large
    with pytest.raises(ValueError):
        entangled_subspace(dim, local_dim, r)


def test_entangled_subspace_maximal_dim():
    """Test with the maximal possible dimension."""
    local_dim = 4
    r = 1
    max_dim = (local_dim - r) * (local_dim - r)  # 9
    result = entangled_subspace(max_dim, local_dim, r)
    
    # Check dimensions
    assert result.shape == (16, 9)  # 16 = 4*4
    
    # Check that the columns are linearly independent
    rank = np.linalg.matrix_rank(result.toarray())
    assert rank == max_dim
