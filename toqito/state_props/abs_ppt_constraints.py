"""Compute the constraints on a spectrum for it to be absolutely PPT."""

import numpy as np

def abs_ppt_constraints(eigs: np.ndarray, p: int, max_constraints: int) -> list[np.ndarray]:
    r"""Return the constraint matrices for the spectrum given by :code:`eigs` to be absolutely PPT.

        This function is adapted from QETLAB.

        Examples
        ==========
        Demonstrate how the function works with expected output.

        .. jupyter-execute::

            import numpy as np
            x = np.array([[1, 2], [3, 4]])
            print(x)

        References
        ==========
        .. bibliography::
            :filter: docname in docnames

        :param eigs: A list of eigenvalues in non-increasing order.
        :param p: The dimension of the smaller subsystem in the bipartite system.
        :param max_constraints: The maximum number of constraint matrices to compute.
        :return: A list of `max_constraints` constraint matrices which must be positive
                 semidefinite for an absolutely PPT spectrum.

    """
    p_p = p * (p + 1) // 2
    X = (p_p + 1) * np.ones((p, p), dtype=np.int32)
    num_pool = np.ones((p_p, 1), dtype=np.int32)
    constraints = []

    # Hard-code matrices for p = 1, 2
    if p == 1:
        return []
    if p == 2:
        return [np.array([[2 * eigs[-1], eigs[-2] - eigs[0]],
                          [eigs[-2] - eigs[0], 2 * eigs[-3]]])]

    X[0, 0] = 1
    X[0, 1] = 2
    X[-1, -1] = p_p
    X[-2, -1] = p_p - 1

    def _fill_matrix(i: int, j: int, l_lim: int) -> None:
        r"""Construct all valid orderings by backtracking."""
        if len(constraints) == max_constraints:
            return
        j_p = j * (j + 1) // 2
        u_lim = min(i * (p - j) + j_p + 1, p_p - 2)
        for k in range(l_lim, u_lim + 1):
            if num_pool[k] == 1:
                X[i, j] = k
                num_pool[k] = 0
                if i == 0 or X[i - 1, j] < X[i, j]:
                    if i == p - 2 and j == p - 2:
                        constraints.append(_create_constraint(eigs, X, p))
                    elif j == p - 1:
                        _fill_matrix(i + 1, i + 1, 3)
                    else:
                        _fill_matrix(i, j + 1, k + 1)
                num_pool[k] = 1
        X[i, j] = p_p + 1

    _fill_matrix(0, 2, 3)

    return constraints

def _create_constraint(eigs: np.ndarray, X: np.ndarray, p: int) -> np.ndarray:
    r"""Return constraint matrix from order matrix."""
    L = np.zeros((p, p))
     # Set upper triangle + diagonal
    upper_inds = np.triu_indices(p)
    L[upper_inds] = eigs[-X[upper_inds]]
    strictly_upper_inds = np.triu_indices(p, 1)
    # Set lower triangle
    L.T[strictly_upper_inds] = -eigs[np.unique(X[strictly_upper_inds], return_inverse=True)[1]]
    return L + L.T
