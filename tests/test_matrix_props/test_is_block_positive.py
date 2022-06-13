"""Test is_block_positive."""
import numpy as np
import pytest

from toqito.states import bell
from toqito.channels import choi, partial_transpose
from toqito.perms import swap, swap_operator
from toqito.matrix_props import is_block_positive


@pytest.mark.parametrize("d", [2, 3, 4])
def test_swap_operator_is_block_positive(d):
    """Test Swap is 1-block positive but not 2-block positive."""
    mat = swap_operator(d)
    np.testing.assert_equal(is_block_positive(mat), True)
    np.testing.assert_equal(is_block_positive(mat, k=2), False)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_choi_is_block_positive(d):
    """Test Choi map is 1-block positive but not 2-block positive."""
    mat = choi()
    np.testing.assert_equal(is_block_positive(mat), True)
    np.testing.assert_equal(is_block_positive(mat, k=2), False)


def test_is_block_positive():
    r"""
    Test that the positive linear map introduced in lemma 3 of:
        S. Bandyopadhyay, A. Cosentino, N. Johnston, V. Russo, J. Watrous, and N. Yu.
        Limitations on separable measurements by convex optimization.
        E-print: arXiv:1408.6981 [quant-ph], 2014.
    is block positive.
    """
    b0 = bell(0)
    b3 = bell(3)
    v0 = np.kron(b0, b0)
    y_mat = (
        np.kron(np.eye(4), b0 @ b0.T) / 2 + np.kron(b3 @ b3.T, partial_transpose(b3 @ b3.T, 1))
    ) / 3 - v0 @ v0.T / 4
    mat = swap(y_mat, [2, 3], [2, 2, 2, 2])
    np.testing.assert_equal(is_block_positive(mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
