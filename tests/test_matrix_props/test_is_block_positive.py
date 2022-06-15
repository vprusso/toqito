"""Test is_block_positive."""
import numpy as np
import pytest

from toqito.states import bell
from toqito.channels import choi, partial_transpose
from toqito.perms import swap, swap_operator
from toqito.matrix_props import is_block_positive


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_swap_operator_is_block_positive(dim):
    """Test Swap is 1-block positive but not 2-block positive."""
    mat = swap_operator(dim)
    np.testing.assert_equal(is_block_positive(mat), True)
    np.testing.assert_equal(is_block_positive(mat, k=2), False)


def test_choi_is_block_positive():
    """Test Choi map is 1-block positive but not 2-block positive."""
    mat = choi()
    np.testing.assert_equal(is_block_positive(mat, rtol=0.001), True)
    np.testing.assert_equal(is_block_positive(mat, k=2, rtol=0.001), False)


def test_is_block_positive():
    r"""
    Test that the positive linear map introduced in lemma 3 of:
        S. Bandyopadhyay, A. Cosentino, N. Johnston, V. Russo, J. Watrous, and N. Yu.
        Limitations on separable measurements by convex optimization.
        E-print: arXiv:1408.6981 [quant-ph], 2014.
    is block positive.
    """
    b_0 = bell(0)
    b_3 = bell(3)
    v_0 = np.kron(b_0, b_0)
    y_mat = (
        np.kron(np.eye(4), b_0 @ b_0.T) / 2
        + np.kron(b_3 @ b_3.T, partial_transpose(b_3 @ b_3.T, 1))
    ) / 3 - v_0 @ v_0.T / 4
    mat = swap(y_mat, [2, 3], [2, 2, 2, 2])
    np.testing.assert_equal(is_block_positive(mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
