"""Compute the constraints on a spectrum for it to be absolutely PPT."""

import cvxpy as cp
import numpy as np


def abs_ppt_constraints(
    eigs: np.ndarray | cp.Variable, p: int, max_constraints: int = 33_592, use_check: bool = False
) -> list[np.ndarray | cp.Expression]:
    r"""Return the constraint matrices for the spectrum to be absolutely PPT [@hildebrand2007positive].

    The returned matrices are constructed from the provided eigenvalues `eigs`, and they must all be positive
    semidefinite for the spectrum to be absolutely PPT.


    !!! Note
        The function does not always return the optimal number of constraint matrices.
        There are some redundant constraint matrices [@johnston2014counting].

        * With `use_checks=False`, the number of matrices returned starting from \(p=1\) is
          \([0, 1, 2, 12, 286, 33592, 23178480, \ldots]\).
        * With `use_checks=True`, the number of matrices returned starting from \(p=1\) is
          \([0, 1, 2, 10, 114, 2612, 108664, \ldots]\).

        However, the optimal number of matrices starting from \(p=1\) is given by
        \([0, 1, 2, 10, 114, 2608, 107498]\).

    !!! Note
        This function accepts a `cvxpy` Variable as input for `eigs`. The function
        will return the assembled constraint matrices as a list of `cvxpy` Expressions.
        These can be used with `cvxpy` to optimize over the space of absolutely PPT matrices.

        The user must impose the ordering
        \(\text{eigs}[0] \geq \text{eigs}[1] \geq \cdots \geq \text{eigs}[-1] \geq 0\), along with the
        positive semidefinite constraint on each returned matrix separately.

        It is recommended to set `use_check=True` for this use case to minimize the number of
        constraint equations in the problem.

    This function is adapted from QETLAB [@qetlablink].

    Args:
        eigs: A list of eigenvalues.
        p: The dimension of the smaller subsystem in the bipartite system.
        max_constraints: The maximum number of constraint matrices to compute. (default: 33,592)
        use_check: Use the "criss-cross" ordering check described in [@johnston2014counting] to reduce the number of
            constraint matrices. (default: `False`)

    Returns:
        A list of `max_constraints` constraint matrices which must be positive semidefinite for an absolutely PPT
        spectrum.

    Raises:
        TypeError: If `eigs` is not a `numpy` ndarray or a `cvxpy` Variable.

    Examples:
        We can compute the constraint matrices for a random density matrix:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.rand import random_density_matrix
        from toqito.state_props import abs_ppt_constraints

        rho = random_density_matrix(9)  # assumed to act on a 3 x 3 bipartite system
        eigs = np.linalg.eigvalsh(rho)
        constraints = abs_ppt_constraints(eigs, 3)
        for i, cons in enumerate(constraints, 1):
            print(f"Constraint {i}:")
            print(cons)
        ```

    """
    if isinstance(eigs, np.ndarray):
        eigs = np.sort(eigs)[::-1]
    elif isinstance(eigs, cp.Variable):
        pass
    else:
        raise TypeError("mat must be a numpy ndarray or a cvxpy Variable")

    # Hard-code matrices for p = 1, 2.
    if p == 1:
        return []
    if p == 2:
        add_index = np.array([[-1, -2], [-2, -3]], dtype=np.int32)
        sub_index = np.array([[-1, 0], [0, -3]], dtype=np.int32)
        diag = np.diag if isinstance(eigs, np.ndarray) else cp.diag
        return [eigs[add_index] - eigs[sub_index] + 2 * diag(eigs[np.diag(add_index)])]

    p_plus = p * (p + 1) // 2
    order_matrix = np.zeros((p, p), dtype=np.int32)
    available = np.ones(p_plus, dtype=bool)
    constraints = []

    # The first two elements of the first row and the last two elements of the last column are fixed.
    order_matrix[0, 0] = 1
    order_matrix[0, 1] = 2
    order_matrix[-1, -1] = p_plus
    order_matrix[-2, -1] = p_plus - 1

    def _fill_matrix(row: int, col: int, l_lim: int) -> None:
        r"""Construct all valid orderings by backtracking. Processes order matrix in row major order.

        A valid ordering has rows and columns of the upper triangle + diagonal in ascending order.
        """
        # If we already constructed enough constraints, exit.
        if len(constraints) == max_constraints:
            return
        col_plus = col * (col + 1) // 2
        # We check numbers in [l_lim, u_lim].
        # u_lim is calculated by considering how many numbers are definitely not admissible for
        # the current location, which is the number of locations to the lower-right of the
        # current position (directly below and directly to the right included).
        u_lim = min(row * (p - col) + col_plus + 1, p_plus - 2)
        for k in range(l_lim, u_lim + 1):
            # If k is available, try it.
            if available[k]:
                order_matrix[row, col] = k
                available[k] = False
                # If placing this k was valid, then we proceed.
                # We only check the ascending column condition because the rows are
                # ascending by construction.
                # A simple explanation: We set l_lim to be greater than the last set number
                # and we are setting elements in row major order.
                if row == 0 or order_matrix[row - 1, col] < order_matrix[row, col]:
                    if row == p - 2 and col == p - 2:
                        # We already placed the last two elements of the last column, so
                        # we have completed the matrix.
                        # Now we create a constraint matrix out of this order matrix.
                        if not use_check or _check_cross(order_matrix, p):
                            constraints.append(_create_constraint(eigs, order_matrix, p))
                    elif col == p - 1:
                        # We finished the current row, so head to the next row.
                        # Also reset l_lim: It will automatically be set to a valid value
                        # by the column ordering check.
                        _fill_matrix(row + 1, row + 1, 3)
                    else:
                        # We are not done with the current row, so head to the next column.
                        # Set l_lim to be greater than the current number to maintain the
                        # row ordering condition.
                        _fill_matrix(row, col + 1, k + 1)
                available[k] = True

    def _check_cross(order_matrix: np.ndarray, p: int) -> bool:
        r"""Check if the order matrix satisfies the "criss-cross" check in [@johnston2014counting].

        This is a vectorized reformulation of the six-nested ``O(p^6)`` loop. Because only the upper
        triangle (plus diagonal) of ``order_matrix`` is populated, every ``order_matrix[min(a, b)][max(a, b)]``
        access is a lookup into the symmetric completion ``sym`` of ``order_matrix``. The check fails iff there
        exists a triple of index pairs ``(j, k), (m, n), (x, g)`` in the loop ranges such that all three
        comparisons hold simultaneously; numpy broadcasting evaluates every such combination at once.
        """
        # Symmetric completion: sym[a, b] == order_matrix[min(a, b)][max(a, b)] since the lower triangle is 0.
        sym = order_matrix + order_matrix.T - np.diag(np.diag(order_matrix))

        # Loop ranges, matching the original nested loops exactly.
        j = np.arange(0, p - 3)  # range(p - 3)
        k = np.arange(2, p)  # range(2, p)
        m = np.arange(0, p - 2)  # range(p - 2)
        n = np.arange(1, p)  # range(1, p)
        x = np.arange(0, p - 1)  # range(p - 1)
        g = np.arange(1, p)  # range(1, p)

        # Any empty range means there are no triples to check, so the matrix trivially passes.
        if min(len(j), len(k), len(m), len(n), len(x), len(g)) == 0:
            return True

        # Pairwise value blocks; reshape onto distinct broadcast axes (j, k, m, n, x, g).
        s_jk = sym[np.ix_(j, k)].reshape(len(j), len(k), 1, 1, 1, 1)
        s_mn = sym[np.ix_(m, n)].reshape(1, 1, len(m), len(n), 1, 1)
        s_ng = sym[np.ix_(n, g)].reshape(1, 1, 1, len(n), 1, len(g))
        s_kx = sym[np.ix_(k, x)].reshape(1, len(k), 1, 1, len(x), 1)
        s_jg = sym[np.ix_(j, g)].reshape(len(j), 1, 1, 1, 1, len(g))
        s_mx = sym[np.ix_(m, x)].reshape(1, 1, len(m), 1, len(x), 1)

        violation = (s_jk > s_mn) & (s_ng > s_kx) & (s_jg < s_mx)
        return not violation.any()

    def _create_constraint(eigs: np.ndarray, order_matrix: np.ndarray, p: int) -> np.ndarray:
        r"""Return constraint matrix from order matrix."""
        add_index = -np.where(order_matrix, order_matrix, order_matrix.T)
        renum_su_tri = np.unique(order_matrix - np.diag(np.diag(order_matrix)), return_inverse=True)[1].reshape(p, p)
        sub_index = renum_su_tri + renum_su_tri.T - 1 + np.diag(np.diag(add_index) + 1)
        diag = np.diag if isinstance(eigs, np.ndarray) else cp.diag
        return eigs[add_index] - eigs[sub_index] + 2 * diag(eigs[np.diag(add_index)])

    # We already set the first two elements of the first row, so start from the third element.
    # We also used 1 and 2 already in order_matrix, so we start checking numbers from 3.
    _fill_matrix(0, 2, 3)

    return constraints
