"""Computes max values for Bell inequalities (General and Qubit-specific)."""

import numbers
import re
import warnings
from itertools import combinations

import cvxpy as cp
import numpy as np

from toqito.helper.bell_notation_conversions import (
    cg_to_fc,
    cg_to_fp,
    fc_to_cg,
    fp_to_cg,
    fp_to_fc,
)
from toqito.matrix_ops import partial_transpose
from toqito.perms import permutation_operator, swap
from toqito.state_opt.npa_hierarchy import bell_npa_constraints


def _integer_digits(number: int, base: int, digits: int) -> np.ndarray:
    """Convert an integer to a fixed-length array of its digits in a given base."""
    dits = np.zeros(digits, dtype=int)
    temp_number = number
    for i in range(digits):
        dits[digits - 1 - i] = temp_number % base
        temp_number //= base
    return dits


def _cg_to_fp_cp(p_cg_var: cp.Variable, desc: list[int]) -> list[cp.Expression]:
    """Generate cp expressions for full probabilities from a CG variable."""
    oa, ob, ia, ib = desc
    fp_expressions = []

    def _cg_row_index(a: int, x: int) -> int:
        return 1 + a + x * (oa - 1)

    def _cg_col_index(b: int, y: int) -> int:
        return 1 + b + y * (ob - 1)

    for x in range(ia):
        for y in range(ib):
            if oa > 1 and ob > 1:
                for a in range(oa - 1):
                    for b in range(ob - 1):
                        row_idx = _cg_row_index(a, x)
                        col_idx = _cg_col_index(b, y)
                        fp_expressions.append(p_cg_var[row_idx, col_idx])

            if oa > 1:
                for a in range(oa - 1):
                    row_idx = _cg_row_index(a, x)
                    cg_a_marg = p_cg_var[row_idx, 0]
                    sum_b = 0
                    if ob > 1:
                        sum_b = cp.sum([p_cg_var[row_idx, _cg_col_index(b_prime, y)] for b_prime in range(ob - 1)])
                    fp_expressions.append(cg_a_marg - sum_b)

            if ob > 1:
                for b in range(ob - 1):
                    col_idx = _cg_col_index(b, y)
                    cg_b_marg = p_cg_var[0, col_idx]
                    sum_a = 0
                    if oa > 1:
                        sum_a = cp.sum([p_cg_var[_cg_row_index(a_prime, x), col_idx] for a_prime in range(oa - 1)])
                    fp_expressions.append(cg_b_marg - sum_a)

            sum_a_marg = 0
            if oa > 1:
                sum_a_marg = cp.sum([p_cg_var[_cg_row_index(a, x), 0] for a in range(oa - 1)])

            sum_b_marg = 0
            if ob > 1:
                sum_b_marg = cp.sum([p_cg_var[0, _cg_col_index(b, y)] for b in range(ob - 1)])

            sum_ab_joint = 0
            if oa > 1 and ob > 1:
                sum_ab_joint = cp.sum(
                    [p_cg_var[_cg_row_index(a, x), _cg_col_index(b, y)] for a in range(oa - 1) for b in range(ob - 1)]
                )

            fp_expressions.append(p_cg_var[0, 0] - sum_a_marg - sum_b_marg + sum_ab_joint)

    return fp_expressions


def bell_inequality_max(
    coefficients: np.ndarray,
    desc: list[int],
    notation: str,
    mtype: str = "classical",
    k: int | str = 1,
    tol: float = 1e-8,
) -> float:
    r"""Compute the maximum value of a Bell inequality.

    Calculates the maximum value achievable for a given Bell inequality under classical, quantum,
    or no-signalling assumptions.

    The maximum classical and no-signalling values are computed exactly. The maximum quantum value
    is upper bounded using the NPA (Navascués-Pironio-Acín) hierarchy :cite:``Navascues_2008_AConvergent``.

    Examples
    ==========

    The CHSH inequality in Full Correlator (FC) notation.
    The classical maximum is 2, the quantum maximum (Tsirelson's bound) is :math:`2\sqrt{2}`,
    and the no-signalling maximum is 4.

    .. math::
        \langle A_1 B_1 \rangle + \langle A_1 B_2 \rangle + \langle A_2 B_1 \rangle - \langle A_2 B_2 \rangle \le V

    Represented by the coefficient matrix:

    .. math::
        M_{FC} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 1 & -1 \end{pmatrix}

    .. jupyter-execute::

        import numpy as np
        from toqito.state_opt.bell_inequality_max import bell_inequality_max
        M_chsh_fc = np.array([[0, 0, 0], [0, 1, 1], [0, 1, -1]])
        desc_chsh = [2, 2, 2, 2]
        bell_inequality_max(M_chsh_fc, desc_chsh, 'fc', 'classical')
        bell_inequality_max(M_chsh_fc, desc_chsh, 'fc', 'quantum', tol=1e-7)
        bell_inequality_max(M_chsh_fc, desc_chsh, 'fc', 'nosignal', tol=1e-9)

    ==========

    The CHSH inequality in Collins-Gisin (CG) notation.
    The classical maximum is 0, the quantum maximum is :math:`1/\sqrt{2} - 1/2`,
    and the no-signalling maximum is 1/2.

    .. math::
        p(00|11)+p(00|12)+p(00|21)-p(00|22)-p_A(0|1)-p_B(0|1) \le V

    Represented by the coefficient matrix:

    .. math::
        M_{CG} = \begin{pmatrix} 0 & -1 & 0 \\ -1 & 1 & 1 \\ 0 & 1 & -1 \end{pmatrix}

    .. jupyter-execute::

        import numpy as np
        from toqito.state_opt.bell_inequality_max import bell_inequality_max
        M_chsh_cg = np.array([[0, -1, 0], [-1, 1, 1], [0, 1, -1]])
        desc_chsh = [2, 2, 2, 2]
        bell_inequality_max(M_chsh_cg, desc_chsh, 'cg', 'classical')
        bell_inequality_max(M_chsh_cg, desc_chsh, 'cg', 'quantum', tol=1e-7)
        bell_inequality_max(M_chsh_cg, desc_chsh, 'cg', 'nosignal', tol=1e-9)

    ==========

    The I3322 inequality in Collins-Gisin (CG) notation.
    Classical max = 1, No-signalling max = 2. Quantum value is between 1 and 2.

    .. jupyter-execute::

        import numpy as np
        from toqito.state_opt.bell_inequality_max import bell_inequality_max
        M_i3322_cg = np.array([[0, 1, 0, 0], [1, -1, -1, -1], [0, -1, -1, 1], [0, -1, 1, 0]])
        desc_i3322 = [2, 2, 3, 3]
        bell_inequality_max(M_i3322_cg, desc_i3322, 'cg', 'classical')
        bell_inequality_max(M_i3322_cg, desc_i3322, 'cg', 'quantum', k=1, tol=1e-7)
        bell_inequality_max(M_i3322_cg, desc_i3322, 'cg', 'quantum', k='1+ab', tol=1e-7)
        bell_inequality_max(M_i3322_cg, desc_i3322, 'cg', 'nosignal', tol=1e-9)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param ``coefficients``: A matrix or tensor specifying the Bell inequality coefficients in either
                             full probability (FP), full correlator (FC), or Collins-Gisin (CG) notation.
    :param ``desc``: A list [:math:`oa`, :math:`ob`, :math:`ma`, :math:`mb`]
                     describing the number of outputs for Alice (:math:`oa`) and Bob (:math:`ob`),
                     and the number of inputs for Alice (:math:`ma`) and Bob (:math:`mb`).
    :param ``notation``: A string ('fp', 'fc', or 'cg') indicating the notation of the ``coefficients``.
    :param ``mtype``: The type of theory to maximize over ('classical', 'quantum', or 'nosignal').
                      Defaults to 'classical'. Note: 'quantum' computes an upper bound via NPA hierarchy.
    :param ``k``: The level of the NPA hierarchy to use if ``mtype='quantum'``. Can be an integer (e.g., 1, 2)
                  or a string specifying intermediate levels (e.g., '1+ab', '1+aab'). Defaults to 1.
                  Higher levels yield tighter bounds but require more computation. Ignored if ``mtype`` is
                  not 'quantum'.
    :param ``tol``: Tolerance for numerical comparisons and solver precision. Defaults to ``1e-8``.
    :return: The maximum value (or quantum upper bound) of the Bell inequality.
    :raises ValueError: If the input ``notation`` is invalid.
    :raises ValueError: If the input ``mtype`` is invalid.
    :raises ValueError: If notation conversion fails (e.g., 'fc' for non-binary outputs).
    :raises ValueError: If the NPA level ``k`` is invalid.
    :raises ValueError: If generating NPA constraints fails.
    :raises cp.error.SolverError: If the cp solver fails.

    """
    oa, ob, ma, mb = desc
    mtype_low = mtype.lower()
    notation_low = notation.lower()

    if notation_low not in ["fp", "fc", "cg"]:
        raise ValueError(f"Invalid notation: {notation}. Must be 'fp', 'fc', or 'cg'.")

    bmax = None
    problem = None

    if mtype_low == "nosignal":
        if notation_low == "fc" and (oa != 2 or ob != 2):
            raise ValueError(
                "Notation conversion failed: 'fc' notation is only supported for binary outputs (oa=2, ob=2)."
            )
        try:
            if notation_low == "cg":
                M = coefficients
            elif notation_low == "fp":
                M = fp_to_cg(coefficients, behavior=False)
            else:
                M = fc_to_cg(coefficients, behavior=False)
        except ValueError as e:
            raise ValueError(f"Notation conversion failed: {e}") from e

        cg_dim = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
        if M.shape != cg_dim:
            raise ValueError(
                f"Coefficient shape {M.shape} incompatible with desc={desc} and CG notation (expected {cg_dim})."
            )

        p_cg = cp.Variable(cg_dim, name="p_cg")

        objective = cp.Maximize(cp.sum(cp.multiply(M, p_cg)))

        constraints = [p_cg[0, 0] == 1]
        fp_expressions = _cg_to_fp_cp(p_cg, desc)

        constraints += [expr >= -tol for expr in fp_expressions]

        problem = cp.Problem(objective, constraints)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            bmax = problem.solve(solver=cp.SCS, eps=tol, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: Solver status for 'nosignal': {problem.status}")
            if problem.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
                bmax = -np.inf
            elif problem.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:
                bmax = np.inf

    elif mtype_low == "quantum":
        if notation_low == "fc" and (oa != 2 or ob != 2):
            raise ValueError(
                "Notation conversion failed: 'fc' notation is only supported for binary outputs (oa=2, ob=2)."
            )

        if not isinstance(k, (str, numbers.Integral)) or (isinstance(k, numbers.Integral) and k < 0):
            raise ValueError(f"Invalid NPA level k={k}. Must be a non-negative integer or a valid string level.")
        if isinstance(k, str):
            # Use regex to validate the string format: digits optionally followed by '+' and 'a's/'b's
            if not re.fullmatch(r"\d+(\+[ab]+)?", k):
                raise ValueError(
                    f"Invalid NPA level k='{k}'. String format must be an integer (e.g., '1', '2') "
                    f"optionally followed by '+' and a sequence of 'a's and 'b's (e.g., '1+ab', '2+aab')."
                )

        try:
            if notation_low == "cg":
                M = coefficients
            elif notation_low == "fp":
                M = fp_to_cg(coefficients, behavior=False)
            else:
                M = fc_to_cg(coefficients, behavior=False)
        except ValueError as e:
            raise ValueError(f"Notation conversion failed: {e}") from e

        cg_dim = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
        if M.shape != cg_dim:
            raise ValueError(
                f"Coefficient shape {M.shape} incompatible with desc={desc} and CG notation (expected {cg_dim})."
            )

        p_var = cp.Variable(cg_dim, name="p_cg_quantum")

        objective = cp.Maximize(cp.sum(cp.multiply(M, p_var)))

        constraints = [p_var[0, 0] == 1]
        try:
            npa_constraints = bell_npa_constraints(p_var, desc, k)
            constraints += npa_constraints
        except ValueError as e:
            raise ValueError(f"Error generating NPA constraints: {e}") from e

        problem = cp.Problem(objective, constraints)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            bmax = problem.solve(solver=cp.SCS, eps=tol, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: Solver status for 'quantum' k={k}: {problem.status}")
            if problem.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
                bmax = -np.inf
            elif problem.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:
                bmax = np.inf

    elif mtype_low == "classical":
        current_oa, current_ob, current_ma, current_mb = oa, ob, ma, mb

        if current_oa == 2 and current_ob == 2:
            expected_fc_shape = (ma + 1, mb + 1)
            expected_cg_shape = (ma * (oa - 1) + 1, mb * (ob - 1) + 1)
            expected_fp_shape = (oa, ob, ma, mb)

            try:
                if notation_low == "fc":
                    if coefficients.shape != expected_fc_shape:
                        raise ValueError(
                            f"FC coefficient shape {coefficients.shape} incompatible "
                            f"with desc={desc} (expected {expected_fc_shape})."
                        )
                    M = coefficients
                elif notation_low == "fp":
                    if coefficients.shape != expected_fp_shape:
                        raise ValueError(
                            f"FP coefficient shape {coefficients.shape} incompatible "
                            f"with desc={desc} (expected {expected_fp_shape})."
                        )
                    M = fp_to_fc(coefficients, behavior=False)
                else:  # notation_low == "cg"
                    if coefficients.shape != expected_cg_shape:
                        raise ValueError(
                            f"CG coefficient shape {coefficients.shape} incompatible "
                            f"with desc={desc} (expected {expected_cg_shape})."
                        )
                    M = cg_to_fc(coefficients, behavior=False)
            except ValueError as e:
                raise ValueError(f"Notation conversion failed for binary scenario: {e}") from e

            if current_ma == 0 or current_mb == 0:
                if current_ma == 0:
                    bmax = M[0, 0] + np.sum(np.abs(M[0, 1:]))
                else:  # current_mb == 0
                    bmax = M[0, 0] + np.sum(np.abs(M[1:, 0]))
                return float(bmax)

            if current_ma < current_mb:
                M = M.T
                current_ma, current_mb = mb, ma

            b_marginal = M[0, 1:]
            a_marginal = M[1:, 0]
            correlations = M[1:, 1:]
            bmax = -np.inf

            num_bob_strategies = 1 << current_mb
            for b_idx in range(num_bob_strategies):
                b_digits = _integer_digits(b_idx, 2, current_mb)
                b_vec = 1 - 2 * b_digits
                temp_bmax = b_marginal @ b_vec + np.sum(np.abs(a_marginal + correlations @ b_vec))
                bmax = max(bmax, temp_bmax)

            bmax += M[0, 0]

        else:
            if notation_low == "fc":
                raise ValueError(
                    "Notation conversion failed for non-binary scenario: "
                    "'fc' notation is only supported for binary outputs (oa=2, ob=2)."
                )

            expected_cg_shape = (ma * (oa - 1) + 1, mb * (ob - 1) + 1)
            expected_fp_shape = (oa, ob, ma, mb)

            if current_ma == 0 or current_mb == 0:
                return 0.0

            try:
                if notation_low == "fp":
                    if coefficients.shape != expected_fp_shape:
                        raise ValueError(
                            f"FP coefficient shape {coefficients.shape} incompatible "
                            f"with desc={desc} (expected {expected_fp_shape})."
                        )
                    M_fp = coefficients
                elif notation_low == "cg":
                    if coefficients.shape != expected_cg_shape:
                        raise ValueError(
                            f"CG coefficient shape {coefficients.shape} incompatible "
                            f"with desc={desc} (expected {expected_cg_shape})."
                        )
                    M_fp = cg_to_fp(coefficients, desc, behavior=False)
            except ValueError as e:
                raise ValueError(f"Notation conversion failed for non-binary scenario: {e}") from e

            num_a_strats = current_oa**current_ma
            num_b_strats = current_ob**current_mb

            if num_a_strats < num_b_strats:
                M_fp = np.transpose(M_fp, (1, 0, 3, 2))
                current_oa, current_ob = ob, oa
                current_ma, current_mb = mb, ma

            M_perm = np.transpose(M_fp, (0, 2, 1, 3))

            bob_dim_size = current_ob * current_mb
            alice_dim_size = current_oa * current_ma
            M_reshaped = M_perm.reshape(alice_dim_size, bob_dim_size)

            bmax = -np.inf
            num_bob_strategies = current_ob**current_mb
            bob_offset = current_ob * np.arange(current_mb)

            for b_idx in range(num_bob_strategies):
                b_digits = _integer_digits(b_idx, current_ob, current_mb)
                bob_indices_for_sum = b_digits + bob_offset

                Ma = np.sum(M_reshaped[:, bob_indices_for_sum], axis=1)

                Ma_reshaped = Ma.reshape(current_oa, current_ma)
                max_a_for_x = np.max(Ma_reshaped, axis=0)

                temp_bmax = np.sum(max_a_for_x)
                bmax = max(bmax, temp_bmax)

    else:
        raise ValueError(f"Invalid mtype: {mtype}. Must be 'classical', 'quantum', or 'nosignal'.")

    if bmax is None or np.isnan(bmax):
        return -np.inf

    if np.isinf(bmax):
        return float(bmax)

    return float(bmax)


def bell_inequality_max_qubits(
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
        from toqito.state_opt.bell_inequality_max import bell_inequality_max_qubits

        joint_coe = np.array([
            [1, 1, -1],
            [1, 1, 1],
            [-1, 1, 0],
        ])
        a_coe = np.array([0, -1, 0])
        b_coe = np.array([-1, -2, 0])
        a_val = np.array([0, 1])
        b_val = np.array([0, 1])

        result = bell_inequality_max_qubits(joint_coe, a_coe, b_coe, a_val, b_val)
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
