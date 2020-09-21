"""Test cvx_kron."""
import cvxpy
import numpy as np

from toqito.helper import cvx_kron


def test_cvx_kron():
    """Ensure return type is CVX object."""
    rho = cvxpy.Variable((4, 4))
    res_mat = cvx_kron(rho, np.identity(4))
    np.testing.assert_equal(isinstance(res_mat, cvxpy.atoms.affine.vstack.Vstack), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
