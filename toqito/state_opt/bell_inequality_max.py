"""Computes the upper bound for a given bipartite Bell inequality."""

from itertools import combinations

import cvxpy as cp
import numpy as np

from toqito.matrix_ops import partial_transpose
from toqito.perms import permutation_operator, swap


def bell_inequality_max(
    joint_coe: np.ndarray,
    a_coe: np.ndarray,
    b_coe: np.ndarray,
    a_val: np.ndarray,
    b_val: np.ndarray,
    solver_name: str = "SCS",
) -> float:
    r"""Return the upper bound for the maximum violation(Tsirelson Bound) for a given bipartite Bell inequality.

    This computes the upper bound for the maximum value of a given bipartite Bell inequality using an SDP.
    The method is from :footcite:`Navascues_2014_Characterization` and the implementation is based on
    :footcite:`QETLAB_link`. This is useful for various tasks in device independent quantum information processing.

    The function formulates the problem as a SDP problem in the following format for the :math:`W`-state.

    .. math::

        \begin{multline}
        \max \operatorname{tr}\!\Bigl( W \cdot \sum_{a,b,x,y} B^{xy}_{ab}\, M^x_a \otimes N^y_b \Bigr),\\[1ex]
        \text{s.t.} \quad \operatorname{tr}(W) = 1,\quad W \ge 0,\\[1ex]
        W^{T_P} \ge 0,\quad \text{for all bipartitions } P.
        \end{multline}


    Examples
    =======


    Consider the I3322 Bell inequality from :footcite:`Collins_2004`.

    .. math::

        \begin{aligned}
        I_{3322} &= P(A_1 = B_1) + P(B_1 = A_2) + P(A_2 = B_2) + P(B_2 = A_3) \\
                 &\quad - P(A_1 = B_2) - P(A_2 = B_3) - P(A_3 = B_1) - P(A_3 = B_3) \\
                 &\le 2
        \end{aligned}

    The individual and joint coefficents and measurement values are encoded as matrices.
    The upper bound can then be found in :code:`|toqito⟩` as follows.

    .. jupyter-execute::

        import numpy as np
        from toqito.state_opt import bell_inequality_max

        joint_coe = np.array([
            [1, 1, -1],
            [1, 1, 1],
            [-1, 1, 0],
        ])
        a_coe = np.array([0, -1, 0])
        b_coe = np.array([-1, -2, 0])
        a_val = np.array([0, 1])
        b_val = np.array([0, 1])

        result = bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val)
        print(f"Bell inequality maximum value: {result:.3f}")

    References
    ==========
    .. footbibliography::


    :raises ValueError: If `a_val` or `b_val` are not length 2.
    :param joint_coe: The coefficents for terms containing both A and B.
    :param a_coe: The coefficent for terms only containing A.
    :param b_coe: The coefficent for terms only containing B.
    :param a_val: The value of each measurement outcome for A.
    :param b_val: The value of each measurement outcome for B.
    :param solver_name: The solver used.
    :return: The upper bound for the maximum violation of the Bell inequality.

    """
    m, _ = joint_coe.shape

    # Ensure the input vectors are column vectors.
    a_val = a_val.reshape(-1, 1)
    b_val = b_val.reshape(-1, 1)
    a_coe = a_coe.reshape(-1, 1)
    b_coe = b_coe.reshape(-1, 1)

    # Check if vectors a_val and b_val have only two elements.
    if len(a_val) != 2 or len(b_val) != 2:
        raise ValueError("This script is only capable of handling Bell inequalities with two outcomes.")

    tot_dim = 2 ** (2 * m + 2)
    obj_mat = np.zeros((tot_dim, tot_dim), dtype=float)

    # Nested loops to compute the objective matrix.
    for a in range(2):  # a = 0 to 1
        for b in range(2):  # b = 0 to 1
            # Indices below are adjusted to account for Python-MATLAB difference.
            for x in range(1, m + 1):  # x = 1 to m (1-indexed in MATLAB, hence the range adjustment)
                for y in range(1, m + 1):  # y = 1 to m
                    b_coeff = joint_coe[x - 1, y - 1] * a_val[a, 0] * b_val[b, 0]
                    if y == 1:
                        b_coeff += a_coe[x - 1, 0] * a_val[a, 0]
                    if x == 1:
                        b_coeff += b_coe[y - 1, 0] * b_val[b, 0]

                    # Construct Alice and Bob's extended measurement operators.
                    perm_x = [x if i == 0 else (0 if i == x else i) for i in range(m + 1)]
                    perm_y = [y if i == 0 else (0 if i == y else i) for i in range(m + 1)]
                    M = a * np.eye(2 ** (m + 1)) + ((-1) ** a) * permutation_operator([2] * (m + 1), perm_x, 0, 1)
                    N = b * np.eye(2 ** (m + 1)) + ((-1) ** b) * permutation_operator([2] * (m + 1), perm_y, 0, 1)

                    # Adding the result of the tensor product to the objective matrix.
                    obj_mat += b_coeff * np.kron(M, N)

    # Symmetrize the matrix to avoid numerical issues.
    obj_mat = (obj_mat + obj_mat.T) / 2
    aux_mat = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    # Construct the SDP problem.
    W = cp.Variable((2 ** (2 * m), 2 ** (2 * m)), symmetric=True)

    # Dimension boost W to the same dimension as obj_mat.
    M = swap(W, [2, m + 1], [2] * (2 * m))
    X = swap(cp.kron(M, aux_mat), [m + 1, 2 * m + 1], [2] * (2 * m + 2))
    Z = swap(X, [m + 2, 2 * m + 1], [2] * (2 * m + 2))

    objective = cp.Maximize(cp.trace(Z @ obj_mat))

    # Define the constraints.
    constraints = [cp.trace(W) == 1, W >> 0]

    # Adding PPT constraints.
    for sz in range(1, m + 1):
        # Generate all combinations of indices from 1 to 2*m-1 of size sz.
        for ppt_partition in combinations(range(1, 2 * m - 1), sz):
            # Convert to 0-based indexing for Python.
            ppt_partition_updated = [x - 1 for x in ppt_partition]
            # Partial transpose on the partition, ensuring it's positive semidefinite.
            pt_matrix = partial_transpose(W, ppt_partition_updated, [4] + [2] * (2 * (m - 1)))
            constraints.append(pt_matrix >> 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver_name, verbose=False)

    return prob.value
