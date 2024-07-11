"""Test npa_constraints."""

from collections import defaultdict

import cvxpy
import numpy as np
import pytest

from toqito.helper import npa_constraints


def test_gen_words_intermediate_hierarchy():
    """Intermediate NPA hierarchy."""
    # Create a mock assemblage with cvxpy Variables
    referee_dim = 1
    a_out = 2
    b_out = 2
    assemblage = {
        (0, 0): cvxpy.Variable((referee_dim * a_out, referee_dim * b_out)),
        (0, 1): cvxpy.Variable((referee_dim * a_out, referee_dim * b_out)),
        (1, 0): cvxpy.Variable((referee_dim * a_out, referee_dim * b_out)),
        (1, 1): cvxpy.Variable((referee_dim * a_out, referee_dim * b_out))
    }

    # Use a hierarchy level string that includes an intermediate level
    k = "1+ab+bb"

    # Call npa_constraints, which internally calls _gen_words
    constraints = npa_constraints(assemblage, k)

    # Assert that constraints were generated (exact number may vary)
    assert len(constraints) > 0


def cglmp_inequality(dim: int) -> tuple[dict[tuple[int, int], cvxpy.Variable], cvxpy.Expression]:
    """Collins-Gisin-Linden-Massar-Popescu inequality."""
    (a_in, b_in) = (2, 2)
    (a_out, b_out) = (dim, dim)

    mat = defaultdict(cvxpy.Variable)
    for x_in in range(a_in):
        for y_in in range(b_in):
            mat[x_in, y_in] = cvxpy.Variable((a_out, b_out), name=f"M(a, b | {x_in}, {y_in})")

    i_b = cvxpy.Constant(0)
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

    return mat, i_b


@pytest.mark.parametrize("k", [2, "1+ab+aab+baa"])
def test_cglmp_inequality(k):
    """Test Collins-Gisin-Linden-Massar-Popescu inequality.

    See Table 1. from NPA paper :cite:`Navascues_2008_AConvergent`.
    """
    dim = 3
    mat, i_b = cglmp_inequality(dim)
    npa = npa_constraints(mat, k)
    objective = cvxpy.Maximize(i_b)
    problem = cvxpy.Problem(objective, npa)
    val = problem.solve()
    assert pytest.approx(val, 1e-3) == 2.914


@pytest.mark.parametrize("k, expected_size", [("1+a", 9), ("1+ab", 25)])
def test_cglmp_dimension(k, expected_size):
    """Test matrix size in Collins-Gisin-Linden-Massar-Popescu inequality.

    See Table 1. from NPA paper :cite:`Navascues_2008_AConvergent`.
    """
    dim = 3
    mat, i_b = cglmp_inequality(dim)
    npa = npa_constraints(mat, k)
    objective = cvxpy.Maximize(i_b)
    problem = cvxpy.Problem(objective, npa)
    r_size = 0
    for variable in problem.variables():
        if variable.name() == "R":
            r_size = variable.shape[0]
    assert r_size == expected_size
