"""Test npa_constraints."""
from collections import defaultdict
import numpy as np
import cvxpy

from toqito.helper import npa_constraints


# see Table 1. from NPA paper [arXiv:0803.4290].
def test_cglmp_inequality():
    """Test Collins-Gisin-Linden-Massar-Popescu inequality."""
    dim = 3
    (a_in, b_in) = (2, 2)
    (a_out, b_out) = (dim, dim)

    mat = defaultdict(cvxpy.Variable)
    for x_in in range(a_in):
        for y_in in range(b_in):
            mat[x_in, y_in] = cvxpy.Variable((a_out, b_out),
                                    name='M(a, b | {}, {})'.format(x_in, y_in))

    i_b = 0
    for k in range(dim // 2):
        tmp = 0
        for a_val in range(a_out):
            for b_val in range(b_out):
                if a_val == np.mod(b_val + k, dim):
                    tmp += mat[0, 0][a_val, b_val]
                    tmp += mat[1, 1][a_val, b_val]

                if b_val == np.mod(a_val + k + 1, dim):
                    tmp += mat[1, 0][a_val, b_val]

                if b_val == np.mod(a_val + k, dim):
                    tmp += mat[0, 1][a_val, b_val]

                if a_val == np.mod(b_val - k - 1, dim):
                    tmp -= mat[0, 0][a_val, b_val]
                    tmp -= mat[1, 1][a_val, b_val]

                if b_val == np.mod(a_val - k, dim):
                    tmp -= mat[1, 0][a_val, b_val]

                if b_val == np.mod(a_val - k - 1, dim):
                    tmp -= mat[0, 1][a_val, b_val]

        i_b += (1 - 2 * k / (dim - 1)) * tmp

    k = 2
    npa = npa_constraints(mat, k)
    objective = cvxpy.Maximize(i_b)
    problem = cvxpy.Problem(objective, npa)
    val = problem.solve()
    np.testing.assert_equal(np.allclose(val, 2.914, atol=1e-3), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
