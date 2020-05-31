"""Test np_array_as_expr."""
import cvxpy
import numpy as np

from toqito.helper import np_array_as_expr


def test_np_array_as_expr():
    """Ensure return type is CVX object."""
    test_input_mat = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

    res_mat = np_array_as_expr(test_input_mat)
    np.testing.assert_equal(isinstance(res_mat, cvxpy.atoms.affine.vstack.Vstack), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
