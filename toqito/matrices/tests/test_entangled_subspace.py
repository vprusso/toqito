"""Test entangled_subspace."""
import numpy as np
import pytest

from toqito.matrices import entangled_subspace


@pytest.mark.parametrize(
    "dim, local_dim, r, expected_shape, expected_rank",
    [
         # Test basic case.
        (2, 3, 1, (9, 2), 2),
         # Test with unequal local dimensions.
        (3, [3, 4], 1, (12, 3), 3),
        # Test with higher entanglement r = 2.
        (1, 4, 2, (16, 1), 1),
        # Test with maximal dimension.
        (9, 4, 1, (16, 9), 9),
        # Test with local_dim as a one-element list.
        (1, [4], 1, (16, 1), 1),
    ],
)
def test_entangled_subspace(dim, local_dim, r, expected_shape, expected_rank):
    """Test entangled subspace with different parameters."""
    result = entangled_subspace(dim, local_dim, r)

    # Check dimensions.
    assert result.shape == expected_shape

    # Check rank (linearly independent columns).
    rank = np.linalg.matrix_rank(result.toarray())
    assert rank == expected_rank

    # Check that the result is non-zero.
    assert np.linalg.norm(result.toarray()) > 0


def test_entangled_subspace_invalid_dim():
    """Test that an error is raised when dim is too large."""
    dim, local_dim, r = 10, 3, 1

    # (3-1)*(3-1) = 4, so dim=10 is too large.
    with pytest.raises(ValueError):
        entangled_subspace(dim, local_dim, r)


def test_entangled_subspace_default_r():
    """Test that default r=1 works correctly."""
    dim, local_dim = 2, 3
    # Call without specifying r.
    result_with_default = entangled_subspace(dim, local_dim)
    # Call with explicit r=1.
    result_with_explicit = entangled_subspace(dim, local_dim, 1)

    # Results should be identical.
    np.testing.assert_array_equal(
        result_with_default.toarray(), result_with_explicit.toarray()
    )

def test_negative_diagonal_branch():
    """Test the branch where j becomes negative."""
    # Choose values so j becomes negative at least once.
    entangled_subspace(1, [3, 4], 1)

def test_positive_diagonal_branch():
    """Test the branch where j becomes positive."""
    # Choose values so j becomes positive.
    entangled_subspace(1, [4, 3], 1)

def test_ct_early_exit():
    """Test the early exit condition where enough columns are found."""
    # Small dim to ensure early return.
    result = entangled_subspace(1, 4, 1)
    assert result.shape == (16, 1)

def test_entangled_subspace_loop_skip():
    """Test a case where the loops don't actually run."""
    # We need to create parameters where the upper loop bound is <= the lower bound
    # so the loops are skipped entirely.

    # For the loop range(1, m - r + 1), we need m - r + 1 <= 1.
    # This happens when m - r <= 0, or r >= m.

    # But we need to be careful not to violate the dimension check.
    # dim <= (local_dim[0] - r) * (local_dim[1] - r).

    # Setting up the edge case:
    r = 2  # Make r relatively large.
    local_dim = [3, 3]  # Keep local_dim reasonable.
    # m = min(local_dim) = 3.
    # m - r + 1 = 3 - 2 + 1 = 2, so range(1, 2) is just [1].

    # For j loop: range(r + 1 - local_dim[1], local_dim[0] - r).
    # That's range(2 + 1 - 3, 3 - 2) = range(0, 1).

    # Now make the "if k <= ell - r" condition fail.
    # For j=0, ell = min(3, 3) = 3.
    # We need k > ell - r, meaning 1 > 3 - 2, or 1 > 1, which is false.

    # So we need k = 2, but the loop only goes to k=1.
    # Let's create a different case where the j loop doesn't execute.

    # If we set r=2, local_dim=[3,2], then:
    # j loop is range(2+1-2, 3-2) = range(1, 1), which is empty.

    dim, local_dim, r = 1, [3, 2], 2

    # This is still valid: (3-2)*(2-2) = 1*0 = 0, but dim=1.
    # The function should return an empty matrix.

    with pytest.raises(ValueError):
        # This should raise an error because dim > (local_dim[0] - r) * (local_dim[1] - r).
        # 1 > (3-2)*(2-2) = 1 > 0.
        result = entangled_subspace(dim, local_dim, r)

    # Try with dim=0 instead.
    dim, local_dim, r = 0, [3, 2], 2
    result = entangled_subspace(dim, local_dim, r)

    assert result.shape == (6, 0)
