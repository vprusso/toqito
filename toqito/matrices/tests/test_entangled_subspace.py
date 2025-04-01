"""Test entangled_subspace."""
import numpy as np
import pytest
from scipy import sparse

from toqito.matrices import entangled_subspace


@pytest.mark.parametrize(
    "dim, local_dim, r, expected_shape, expected_rank",
    [
        (2, 3, 1, (9, 2), 2),  # Test basic case
        (3, [3, 4], 1, (12, 3), 3),  # Test with unequal local dimensions
        (1, 4, 2, (16, 1), 1),  # Test with higher entanglement r=2
        (9, 4, 1, (16, 9), 9),  # Test with maximal dimension
    ],
)
def test_entangled_subspace(dim, local_dim, r, expected_shape, expected_rank):
    """Test entangled subspace with different parameters."""
    result = entangled_subspace(dim, local_dim, r)
    
    # Check dimensions
    assert result.shape == expected_shape
    
    # Check rank (linearly independent columns)
    rank = np.linalg.matrix_rank(result.toarray())
    assert rank == expected_rank
    
    # Check that the result is non-zero
    assert np.linalg.norm(result.toarray()) > 0


def test_entangled_subspace_invalid_dim():
    """Test that an error is raised when dim is too large."""
    dim, local_dim, r = 10, 3, 1
    
    # (3-1)*(3-1) = 4, so dim=10 is too large
    with pytest.raises(ValueError):
        entangled_subspace(dim, local_dim, r)


def test_entangled_subspace_default_r():
    """Test that default r=1 works correctly."""
    dim, local_dim = 2, 3
    # Call without specifying r
    result_with_default = entangled_subspace(dim, local_dim)
    # Call with explicit r=1
    result_with_explicit = entangled_subspace(dim, local_dim, 1)
    
    # Results should be identical
    np.testing.assert_array_equal(
        result_with_default.toarray(), result_with_explicit.toarray()
    )
