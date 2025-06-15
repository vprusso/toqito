"""Compute the constraints on a spectrum for it to be absolutely PPT."""

import numpy as np


def abs_ppt_constraints(eigs: np.ndarray, p: int, max_constraints: int = 33_592) -> list[np.ndarray]:
    r"""Return the constraint matrices for the spectrum to be absolutely PPT :cite:`Hildebrand_2007_AbsPPT`.

    The returned constraint matrices must all be positive semidefinite for the spectrum to be absolutely PPT.

    .. note::
        The above statement is not strictly true since it is known that the function does not always return the
        optimal number of constraint matrices. There are some redundant constraint matrices.
        :cite:`Johnston_2014_Orderings`

    This function is adapted from QETLAB :cite:`QETLAB_link`.

    Examples
    ==========
    We can compute the constraint matrices for a random density matrix:

    .. jupyter-execute::

        import numpy as np
        from toqito.rand import random_density_matrix
        from toqito.state_props import abs_ppt_constraints
        rho = random_density_matrix(9) # assumed to act on a 3 x 3 bipartite system
        eigs = np.linalg.eigvalsh(rho)
        constraints = abs_ppt_constraints(eigs, 3)
        for i, cons in enumerate(constraints, 1):
            print(f"Constraint {i}:")
            print(cons)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param eigs: A list of eigenvalues.
    :param p: The dimension of the smaller subsystem in the bipartite system.
    :param max_constraints: The maximum number of constraint matrices to compute. By default, this is
                            equal to :math:`33592` which is an upper bound on the optimal number of
                            constraint matrices which must be computed for :math:`p \leq 6`
                            :cite:`Johnston_2014_Orderings`.
    :return: A list of :code:`max_constraints` constraint matrices which must be positive
             semidefinite for an absolutely PPT spectrum.

    """
    # Sort eigenvalues in non-increasing order
    eigs = np.sort(eigs)[::-1]

    # Hard-code matrices for p = 1, 2
    if p == 1:
        return []
    if p == 2:
        return [np.array([[2 * eigs[-1], eigs[-2] - eigs[0]], [eigs[-2] - eigs[0], 2 * eigs[-3]]])]

    p_plus = p * (p + 1) // 2
    order_matrix = np.zeros((p, p), dtype=np.int32)
    available = np.ones(p_plus, dtype=bool)
    constraints = []

    # The first two elements of the first row and the last two elements of the last column are fixed
    order_matrix[0, 0] = 1
    order_matrix[0, 1] = 2
    order_matrix[-1, -1] = p_plus
    order_matrix[-2, -1] = p_plus - 1

    def _fill_matrix(row: int, col: int, l_lim: int) -> None:
        r"""Construct all valid orderings by backtracking. Processes order_matrix in row major order.

        A valid ordering has rows and columns of the upper triangle + diagonal in ascending order.
        """
        # If we already constructed enough constraints, exit
        if len(constraints) == max_constraints:
            return
        col_plus = col * (col + 1) // 2
        # We check numbers in [l_lim, u_lim]
        # u_lim is calculated by considering how many numbers are definitely not admissible for
        # the current location, which is the number of locations to the lower-right of the
        # current position (directly below and directly to the right included)
        u_lim = min(row * (p - col) + col_plus + 1, p_plus - 2)
        for k in range(l_lim, u_lim + 1):
            # If k is available, try it
            if available[k]:
                order_matrix[row, col] = k
                available[k] = False
                # If placing this k was valid, then we proceed
                # We only check the ascending column condition because the rows are
                # ascending by construction
                # A simple explanation: We set l_lim to be greater than the last set number
                # and we are setting elements in row major order
                if row == 0 or order_matrix[row - 1, col] < order_matrix[row, col]:
                    if row == p - 2 and col == p - 2:
                        # We already placed the last two elements of the last column, so
                        # we have completed a valid matrix
                        # Now we create a constraint matrix out of this order matrix
                        constraints.append(_create_constraint(eigs, order_matrix, p))
                    elif col == p - 1:
                        # We finished the current row, so head to the next row
                        # Also reset l_lim: It will automatically be set to a valid value
                        # by the column ordering check
                        _fill_matrix(row + 1, row + 1, 3)
                    else:
                        # We are not done with the current row, so head to the next column
                        # Set l_lim to be greater than the current number to maintain the
                        # row ordering condition
                        _fill_matrix(row, col + 1, k + 1)
                available[k] = True

    def _create_constraint(eigs: np.ndarray, order_matrix: np.ndarray, p: int) -> np.ndarray:
        r"""Return constraint matrix from order matrix."""
        constraint_matrix = np.zeros((p, p))
        # The elements of the upper triangle + diagonal of the constraint matrix are placed by
        # the rule constraint_matrix[row, col] = eigs[-order_matrix[row, col]] (col >= row)
        upper_inds = np.triu_indices(p)
        constraint_matrix[upper_inds] = eigs[-order_matrix[upper_inds]]
        # The elements of the lower triangle are placed in the same order as that of the
        # transposed upper triangle
        strictly_upper_inds = np.triu_indices(p, 1)
        # First we need to translate the elements of the upper triangle from [0, p(p-1)/2)
        renumbered_upper_triangle = np.unique(order_matrix[strictly_upper_inds], return_inverse=True)[1]
        # Then we can directly place everything by indexing the transposed constraint matrix
        # The rule is given by constraint_matrix[col, row] = -eigs[renumbered_upper_triangle[row, col]] (col > row)
        constraint_matrix.T[strictly_upper_inds] = -eigs[renumbered_upper_triangle]
        return constraint_matrix + constraint_matrix.T

    # We already set the first two elements of the first row, so start from the third element
    # We also used 1 and 2 already in order_matrix, so we start checking numbers from 3
    _fill_matrix(0, 2, 3)

    return constraints
