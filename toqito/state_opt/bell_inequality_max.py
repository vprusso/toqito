"""Computes max values for Bell inequalities (General and Qubit-specific)."""

import numbers
import re
import warnings
from itertools import combinations

import cvxpy as cp
import numpy as np

from toqito.matrix_ops import partial_transpose
from toqito.perms import permutation_operator, swap
from toqito.state_opt.npa_hierarchy import bell_npa_constraints


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
    is upper bounded using the NPA (Navascués-Pironio-Acín) hierarchy
    `Navascues_2008_AConvergent`[@navascues2008convergent].

    Args:
        coefficients: A matrix or tensor specifying the Bell inequality coefficients in either
                             full probability (FP), full correlator (FC), or Collins-Gisin (CG) notation.
        desc: A list [\(oa\), \(ob\), \(ma\), \(mb\)]
                     describing the number of outputs for Alice (\(oa\)) and Bob (\(ob\)),
                     and the number of inputs for Alice (\(ma\)) and Bob (\(mb\)).
        notation: A string ('fp', 'fc', or 'cg') indicating the notation of the ``coefficients``.
        mtype: The type of theory to maximize over ('classical', 'quantum', or 'nosignal').
                      Defaults to 'classical'. Note: 'quantum' computes an upper bound via NPA hierarchy.
        k: The level of the NPA hierarchy to use if ``mtype='quantum'``. Can be an integer (e.g., 1, 2)
                  or a string specifying intermediate levels (e.g., '1+ab', '1+aab'). Defaults to 1.
                  Higher levels yield tighter bounds but require more computation. Ignored if ``mtype`` is
                  not 'quantum'.
        tol: Tolerance for numerical comparisons and solver precision. Defaults to ``1e-8``.

    Returns:
        The maximum value (or quantum upper bound) of the Bell inequality.

    Raises:
        ValueError: If the input ``notation`` is invalid.
        ValueError: If the input ``mtype`` is invalid.
        ValueError: If notation conversion fails (e.g., 'fc' for non-binary outputs).
        ValueError: If the NPA level ``k`` is invalid.
        ValueError: If generating NPA constraints fails.
        cp.error.SolverError: If the cp solver fails.

    Examples:
        The CHSH inequality in Full Correlator (FC) notation.
        The classical maximum is 2, the quantum maximum (Tsirelson's bound) is \(2\sqrt{2}\),
        and the no-signalling maximum is 4.

        \[
            \langle A_1 B_1 \rangle + \langle A_1 B_2 \rangle + \langle A_2 B_1 \rangle - \langle A_2 B_2 \rangle \le V
        \]

        Represented by the coefficient matrix:

        \[
            M_{FC} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 1 & -1 \end{pmatrix}
        \]

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import bell_inequality_max
        M_chsh_fc = np.array([[0, 0, 0], [0, 1, 1], [0, 1, -1]])
        desc_chsh = [2, 2, 2, 2]
        bell_inequality_max(M_chsh_fc, desc_chsh, 'fc', 'classical')
        bell_inequality_max(M_chsh_fc, desc_chsh, 'fc', 'quantum', tol=1e-7)
        print(bell_inequality_max(M_chsh_fc, desc_chsh, 'fc', 'nosignal', tol=1e-9))
        ```


        The CHSH inequality in Collins-Gisin (CG) notation.
        The classical maximum is 0, the quantum maximum is \(1/\sqrt{2} - 1/2\),
        and the no-signalling maximum is 1/2.

        \[
            p(00|11)+p(00|12)+p(00|21)-p(00|22)-p_A(0|1)-p_B(0|1) \le V
        \]

        Represented by the coefficient matrix:

        \[
            M_{CG} = \begin{pmatrix} 0 & -1 & 0 \\ -1 & 1 & 1 \\ 0 & 1 & -1 \end{pmatrix}
        \]

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import bell_inequality_max
        M_chsh_cg = np.array([[0, -1, 0], [-1, 1, 1], [0, 1, -1]])
        desc_chsh = [2, 2, 2, 2]
        bell_inequality_max(M_chsh_cg, desc_chsh, 'cg', 'classical')
        bell_inequality_max(M_chsh_cg, desc_chsh, 'cg', 'quantum', tol=1e-7)
        print(bell_inequality_max(M_chsh_cg, desc_chsh, 'cg', 'nosignal', tol=1e-9))
        ```

        The I3322 inequality in Collins-Gisin (CG) notation.
        Classical max = 1, No-signalling max = 2. Quantum value is between 1 and 2.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import bell_inequality_max
        M_i3322_cg = np.array([[0, 1, 0, 0], [1, -1, -1, -1], [0, -1, -1, 1], [0, -1, 1, 0]])
        desc_i3322 = [2, 2, 3, 3]
        bell_inequality_max(M_i3322_cg, desc_i3322, 'cg', 'classical')
        bell_inequality_max(M_i3322_cg, desc_i3322, 'cg', 'quantum', k=1, tol=1e-7)
        bell_inequality_max(M_i3322_cg, desc_i3322, 'cg', 'quantum', k='1+ab', tol=1e-7)
        print(bell_inequality_max(M_i3322_cg, desc_i3322, 'cg', 'nosignal', tol=1e-9))
        ```

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
                M = _fp_to_cg(coefficients, behavior=False)
            else:
                M = _fc_to_cg(coefficients, behavior=False)
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
                M = _fp_to_cg(coefficients, behavior=False)
            else:
                M = _fc_to_cg(coefficients, behavior=False)
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
                    M = _fp_to_fc(coefficients, behavior=False)
                else:  # notation_low == "cg"
                    if coefficients.shape != expected_cg_shape:
                        raise ValueError(
                            f"CG coefficient shape {coefficients.shape} incompatible "
                            f"with desc={desc} (expected {expected_cg_shape})."
                        )
                    M = _cg_to_fc(coefficients, behavior=False)
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
                    M_fp = _cg_to_fp(coefficients, desc, behavior=False)
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
    The method is from [@navascues2014characterization] and the implementation is based on
    [@qetlablink]. This is useful for various tasks in device independent quantum information processing.

    The function formulates the problem as a SDP problem in the following format for the \(W\)-state.

    \[
        \begin{multline}
        \max \operatorname{tr}\!\Bigl( W \cdot \sum_{a,b,x,y} B^{xy}_{ab}\, M^x_a \otimes N^y_b \Bigr),\\[1ex]
        \text{s.t.} \quad \operatorname{tr}(W) = 1,\quad W \ge 0,\\[1ex]
        W^{T_P} \ge 0,\quad \text{for all bipartitions } P.
        \end{multline}
    \]


    Args:
        joint_coe: The coefficients for terms containing both A and B.
        a_coe: The coefficient for terms only containing A.
        b_coe: The coefficient for terms only containing B.
        a_val: The value of each measurement outcome for A.
        b_val: The value of each measurement outcome for B.
        solver_name: The solver used.

    Returns:
        The upper bound for the maximum violation of the Bell inequality.

    Raises:
        ValueError: If `a_val` or `b_val` are not length 2.

    Examples:
        Consider the I3322 Bell inequality from [@collins2004relevant].

        \[
            \begin{aligned}
            I_{3322} &= P(A_1 = B_1) + P(B_1 = A_2) + P(A_2 = B_2) + P(B_2 = A_3) \\
                     &\quad - P(A_1 = B_2) - P(A_2 = B_3) - P(A_3 = B_1) - P(A_3 = B_3) \\
                     &\le 2
            \end{aligned}
        \]

        The individual and joint coefficients and measurement values are encoded as matrices.
        The upper bound can then be found in `|toqito⟩` as follows.

        ```python exec="1" source="above" result="text"
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
        ```

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


def _cg_to_fc(cg_mat: np.ndarray, behavior: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behavior from Collins-Gisin (CG) to Full Correlator (FC) notation.

    The Collins-Gisin (CG) notation for a Bell functional or behavior is represented by a matrix:

    \[
    \text{CG} =
    \begin{pmatrix}
        K      & p_B(0|1) & p_B(0|2) & \dots \\
        p_A(0|1) & p(00|11) & p(00|12) & \dots \\
        p_A(0|2) & p(00|21) & p(00|22) & \dots \\
        \vdots   & \vdots   & \vdots   & \ddots
    \end{pmatrix}
    \]

    The Full Correlator (FC) notation is represented by:

    \[
    \text{FC} =
    \begin{pmatrix}
        K      & \langle B_1 \rangle & \langle B_2 \rangle & \dots \\
        \langle A_1 \rangle & \langle A_1 B_1 \rangle & \langle A_1 B_2 \rangle & \dots \\
        \langle A_2 \rangle & \langle A_2 B_1 \rangle & \langle A_2 B_2 \rangle & \dots \\
        \vdots   & \vdots      & \vdots      & \ddots
    \end{pmatrix}
    \]

    This function converts between these two notations.

    Args:
        cg_mat: The matrix in Collins-Gisin notation.
        behavior: If True, assume input is a behavior (default: False, assume functional).

    Returns:
        The matrix in Full Correlator notation.

    !!! Note
        This function is adapted from the QETLAB MATLAB package function ``CG2FC``.

    Examples:
        Consider the CHSH inequality in CG notation for a functional:

        \[
        \text{CHSH}_{CG} =
        \begin{pmatrix}
            0 & 0 & 0 \\
            0 & 1 & -1 \\
            0 & -1 & 1
        \end{pmatrix}
        \]

        Converting to FC notation:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _cg_to_fc as cg_to_fc

        chsh_cg = np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]])
        print(cg_to_fc(chsh_cg))
        ```

        Consider a behavior (probability distribution) in CG notation:

        \[
        P_{CG} =
        \begin{pmatrix}
            1 & 0.5 & 0.5 \\
            0.5 & 0.25 & 0.25 \\
            0.5 & 0.25 & 0.25
        \end{pmatrix}
        \]

        Converting to FC notation:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _cg_to_fc as cg_to_fc
        p_cg = np.array([[1, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
        print(cg_to_fc(p_cg, behavior=True))
        ```

    """
    ia = cg_mat.shape[0] - 1
    ib = cg_mat.shape[1] - 1

    fc_mat = np.zeros((ia + 1, ib + 1))

    a_vec = cg_mat[1:, 0]
    b_vec = cg_mat[0, 1:]
    c_mat = cg_mat[1:, 1:]

    if not behavior:
        fc_mat[0, 0] = cg_mat[0, 0] + np.sum(a_vec) / 2 + np.sum(b_vec) / 2 + np.sum(c_mat) / 4
        fc_mat[1:, 0] = a_vec / 2 + np.sum(c_mat, axis=1) / 4
        fc_mat[0, 1:] = b_vec / 2 + np.sum(c_mat, axis=0) / 4
        fc_mat[1:, 1:] = c_mat / 4
    else:
        fc_mat[0, 0] = 1
        fc_mat[1:, 0] = 2 * a_vec - 1
        fc_mat[0, 1:] = 2 * b_vec - 1
        fc_mat[1:, 1:] = np.ones((ia, ib)) - 2 * a_vec[:, np.newaxis] - 2 * b_vec[np.newaxis, :] + 4 * c_mat

    return fc_mat


def _fc_to_cg(fc_mat: np.ndarray, behavior: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behavior from Full Correlator (FC) to Collins-Gisin (CG) notation.

    The Full Correlator (FC) notation is represented by:

    \[
    \text{FC} =
    \begin{pmatrix}
        K      & \langle B_1 \rangle & \langle B_2 \rangle & \dots \\
        \langle A_1 \rangle & \langle A_1 B_1 \rangle & \langle A_1 B_2 \rangle & \dots \\
        \langle A_2 \rangle & \langle A_2 B_1 \rangle & \langle A_2 B_2 \rangle & \dots \\
        \vdots   & \vdots      & \vdots      & \ddots
    \end{pmatrix}
    \]

    The Collins-Gisin (CG) notation for a Bell functional or behavior is represented by a matrix:

    \[
    \text{CG} =
    \begin{pmatrix}
        K      & p_B(0|1) & p_B(0|2) & \dots \\
        p_A(0|1) & p(00|11) & p(00|12) & \dots \\
        p_A(0|2) & p(00|21) & p(00|22) & \dots \\
        \vdots   & \vdots   & \vdots   & \ddots
    \end{pmatrix}
    \]

    This function converts between these two notations.

    Args:
        fc_mat: The matrix in Full Correlator notation.
        behavior: If True, assume input is a behavior (default: False, assume functional).

    Returns:
        The matrix in Collins-Gisin notation.

    !!! Note
        This function is adapted from the QETLAB MATLAB package function ``FC2CG``.

    Examples:
        Consider the CHSH inequality in FC notation for a functional:

        \[
        \text{CHSH}_{FC} =
        \begin{pmatrix}
            0 & 0 & 0 \\
            0 & 1/4 & -1/4 \\
            0 & -1/4 & 1/4
        \end{pmatrix}
        \]

        Converting to CG notation:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _fc_to_cg as fc_to_cg
        chsh_fc = np.array([[0, 0, 0], [0, 0.25, -0.25], [0, -0.25, 0.25]])
        print(fc_to_cg(chsh_fc))
        ```

        Consider a behavior (correlation matrix) in FC notation:

        \[
        P_{FC} =
        \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 0 & 0 \\
            0 & 0 & 0
        \end{pmatrix}
        \]

        Converting to CG notation:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _fc_to_cg as fc_to_cg
        p_fc = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        print(fc_to_cg(p_fc, behavior=True))
        ```

    """
    ia = fc_mat.shape[0] - 1
    ib = fc_mat.shape[1] - 1

    cg_mat = np.zeros((ia + 1, ib + 1))

    a_vec = fc_mat[1:, 0]
    b_vec = fc_mat[0, 1:]
    c_mat = fc_mat[1:, 1:]

    if not behavior:
        cg_mat[0, 0] = fc_mat[0, 0] + np.sum(c_mat) - np.sum(a_vec) - np.sum(b_vec)
        cg_mat[1:, 0] = 2 * a_vec - 2 * np.sum(c_mat, axis=1)
        cg_mat[0, 1:] = 2 * b_vec - 2 * np.sum(c_mat, axis=0)
        cg_mat[1:, 1:] = 4 * c_mat
    else:
        cg_mat[0, 0] = 1
        cg_mat[1:, 0] = (1 + a_vec) / 2
        cg_mat[0, 1:] = (1 + b_vec) / 2
        cg_mat[1:, 1:] = (np.ones((ia, ib)) + a_vec[:, np.newaxis] + b_vec[np.newaxis, :] + c_mat) / 4

    return cg_mat


def _cg_to_fp(cg_mat: np.ndarray, desc: list[int], behavior: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behavior from Collins-Gisin (CG) to Full Probability (FP) notation.

    The Collins-Gisin (CG) notation for a Bell functional or behavior is represented by a matrix
    (see :func:`cg_to_fc`). The Full Probability (FP) notation represents the full probability
    distribution \(V(a, b, x, y) = P(a, b | x, y)\), the probability of Alice getting outcome
    \(a\) (0 to oa-1) and Bob getting outcome \(b\) (0 to ob-1) given inputs \(x\)
    (0 to ia-1) and \(y\) (0 to ib-1). This is stored as a 4D numpy array with indices
    `V[a, b, x, y]`.

    This function converts from CG to FP notation.

    Args:
        cg_mat: The matrix in Collins-Gisin notation.
        desc: A list [\(oa\), \(ob\), \(ia\), \(ib\)] describing the number of outputs
                  (\(oa\), \(ob\)) and inputs (\(ia\), \(ib\)).
        behavior: If True, assume input is a behavior (default: False, assume functional).

    Returns:
        The probability tensor \(V[a, b, x, y]\) in Full Probability notation.

    !!! Note
        This function is adapted from the QETLAB MATLAB package function ``CG2FP``.

    Examples:
        Consider the CHSH inequality functional in CG notation:

        \[
        \text{CHSH}_{CG} =
        \begin{pmatrix}
            0 & 0 & 0 \\
            0 & 1 & -1 \\
            0 & -1 & 1
        \end{pmatrix}
        \]

        Converting to FP notation (desc = [2, 2, 2, 2]):

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _cg_to_fp as cg_to_fp
        chsh_cg = np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]])
        desc = [2, 2, 2, 2] # oa, ob, ia, ib
        print(cg_to_fp(chsh_cg, desc))
        ```

        Consider a behavior (probability distribution) in CG notation (desc = [2, 2, 2, 2]):

        \[
        P_{CG} =
        \begin{pmatrix}
            1 & 0.5 & 0.5 \\
            0.5 & 0.25 & 0.25 \\
            0.5 & 0.25 & 0.25
        \end{pmatrix}
        \]

        Converting to FP notation:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _cg_to_fp as cg_to_fp
        p_cg = np.array([[1, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
        desc = [2, 2, 2, 2]
        print(cg_to_fp(p_cg, desc, behavior=True))
        ```

    """
    oa, ob, ia, ib = desc
    v_mat = np.zeros((oa, ob, ia, ib))

    def aindex(a: int, x: int) -> int:
        r"""CG matrix row index for Alice's outcome \(a\) (0..\(oa-2\)) and input \(x\) (0..\(ia-1\)).

        Returns 1-based index.
        """
        return 1 + a + x * (oa - 1)

    def bindex(b: int, y: int) -> int:
        r"""CG matrix col index for Bob's outcome \(b\) (0..\(ob-2\)) and input \(y\) (0..\(ib-1\)).

        Returns 1-based index.
        """
        return 1 + b + y * (ob - 1)

    if not behavior:
        # Functional case logic
        k_term = cg_mat[0, 0] / (ia * ib) if ia > 0 and ib > 0 else 0
        for x in range(ia):
            for y in range(ib):
                # Fill V[a, b, x, y] for a < oa-1, b < ob-1
                for a in range(oa - 1):
                    a_term = cg_mat[aindex(a, x), 0] / ib if ib > 0 else 0
                    for b in range(ob - 1):
                        b_term = cg_mat[0, bindex(b, y)] / ia if ia > 0 else 0
                        v_mat[a, b, x, y] = k_term + a_term + b_term + cg_mat[aindex(a, x), bindex(b, y)]
                # Fill V[a, ob-1, x, y] for a < oa-1 (last column for Bob)
                for a in range(oa - 1):
                    a_term = cg_mat[aindex(a, x), 0] / ib if ib > 0 else 0
                    v_mat[a, ob - 1, x, y] = k_term + a_term
                # Fill V[oa-1, b, x, y] for b < ob-1 (last row for Alice)
                for b in range(ob - 1):
                    b_term = cg_mat[0, bindex(b, y)] / ia if ia > 0 else 0
                    v_mat[oa - 1, b, x, y] = k_term + b_term
                # Fill V[oa-1, ob-1, x, y] (bottom-right corner)
                v_mat[oa - 1, ob - 1, x, y] = k_term

    else:
        for x in range(ia):
            for y in range(ib):
                # Calculate slices for CG matrix corresponding to non-last outcomes
                # Need 1-based indices for slicing cg_mat
                start_row_a = aindex(0, x)
                end_row_a = aindex(oa - 2, x) + 1 if oa > 1 else start_row_a
                slice_a = slice(start_row_a, end_row_a)

                start_col_b = bindex(0, y)
                end_col_b = bindex(ob - 2, y) + 1 if ob > 1 else start_col_b
                slice_b = slice(start_col_b, end_col_b)

                # Get corresponding submatrix or default to zeros/scalars if outputs=1
                cg_sub_mat = cg_mat[slice_a, slice_b] if oa > 1 and ob > 1 else np.array([[]])
                cg_a_marg = cg_mat[slice_a, 0] if oa > 1 else np.array([])
                cg_b_marg = cg_mat[0, slice_b] if ob > 1 else np.array([])

                # V(0..oa-2, 0..ob-2, x, y) = p(a,b|xy)
                if oa > 1 and ob > 1:
                    v_mat[0 : oa - 1, 0 : ob - 1, x, y] = cg_sub_mat

                # V(0..oa-2, ob-1, x, y) = pA(a|x) - sum_{b'=0..ob-2} p(a,b'|xy)
                if oa > 1:
                    sum_b = np.sum(cg_sub_mat, axis=1) if ob > 1 else np.zeros(oa - 1)
                    v_mat[0 : oa - 1, ob - 1, x, y] = cg_a_marg - sum_b

                # V(oa-1, 0..ob-2, x, y) = pB(b|y) - sum_{a'=0..oa-2} p(a',b|xy)
                if ob > 1:
                    sum_a = np.sum(cg_sub_mat, axis=0) if oa > 1 else np.zeros(ob - 1)
                    v_mat[oa - 1, 0 : ob - 1, x, y] = cg_b_marg - sum_a

                # V(oa-1, ob-1, x, y) = 1 - sum pA(a|x) - sum pB(b|y) + sum p(ab|xy)
                sum_a_marg = np.sum(cg_a_marg)
                sum_b_marg = np.sum(cg_b_marg)
                sum_ab_joint = np.sum(cg_sub_mat)
                v_mat[oa - 1, ob - 1, x, y] = (
                    cg_mat[0, 0]  # Should be 1 for behavior
                    - sum_a_marg
                    - sum_b_marg
                    + sum_ab_joint
                )

    return v_mat


def _fc_to_fp(fc_mat: np.ndarray, behavior: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behavior from Full Correlator (FC) to Full Probability (FP) notation.

    Assumes binary outcomes (\(oa=2\), \(ob=2\)) corresponding to physical values +1 and -1.
    The FP tensor indices \(a, b = 0, 1\) correspond to outcomes \(+1, -1\) respectively.

    The Full Correlator (FC) notation is represented by a matrix (see :func:`.fc_to_cg`).
    The Full Probability (FP) notation represents the full probability distribution
    \(V(a, b, x, y) = P(\text{out}_A=a', \text{out}_B=b' | x, y)\),
    where \(a=0 \rightarrow a'=+1\), \(a=1 \rightarrow a'=-1\) (similarly for \(b\)),
    stored as a 4D numpy array \(V[a, b, x, y]\).

    This function converts from FC to FP notation.

    Args:
        fc_mat: The matrix in Full Correlator notation.
        behavior: If True, assume input is a behavior (default: False, assume functional).

    Returns:
        The probability tensor \(V[a, b, x, y]\) in Full Probability notation (oa=2, ob=2).

    !!! Note
        This function is adapted from the QETLAB MATLAB package function ``FC2FP`` [@QETLAB].
        For `behavior=True`, it applies the standard formula relating probabilities to correlators:
        \(P(a', b' | x, y) = (1 + a'\langle A_x \rangle + b'\langle B_y \rangle +\)
        \(a'b'\langle A_x B_y \rangle) / 4\),
        where \(a', b' \in \{+1, -1\}\).
        Crucially, it uses the values \(\langle A_x \rangle\) and \(\langle B_y \rangle\) directly
        from the input ``fc_mat``. If this input matrix was generated using a convention where these
        entries represent *averaged* marginal correlators (like the output of ``fp_to_fc(..., behavior=True)``),
        the resulting FP tensor might not represent a valid probability distribution (e.g., entries could be negative).

    Examples:
        Consider the CHSH inequality functional in FC notation:

        \[
        \text{CHSH}_{FC} =
        \begin{pmatrix}
            0 & 0 & 0 \\
            0 & 1/4 & -1/4 \\
            0 & -1/4 & 1/4
        \end{pmatrix}
        \]

        Converting to FP notation:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _fc_to_fp as fc_to_fp
        chsh_fc = np.array([[0, 0, 0], [0, 0.25, -0.25], [0, -0.25, 0.25]])
        print(fc_to_fp(chsh_fc))
        ```

        Consider a behavior (correlation matrix) in FC notation (e.g., from PR box):
        Note: This FC matrix corresponds to the PR box *after* applying ``fp_to_fc(pr_box, behavior=True)``,
        which uses the QETLAB convention of averaging marginal correlators.

        \[
        P_{FC} =
        \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 1/\sqrt{2} & 1/\sqrt{2} \\
            0 & 1/\sqrt{2} & -1/\sqrt{2}
        \end{pmatrix}
        \]

        Converting to FP notation:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _fc_to_fp as fc_to_fp
        p_fc = np.array([[1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)], [0, 1/np.sqrt(2), -1/np.sqrt(2)]])
        print(fc_to_fp(p_fc, behavior=True))
        ```

    """
    ia = fc_mat.shape[0] - 1
    ib = fc_mat.shape[1] - 1
    # Assumes oa=2, ob=2 based on FC notation structure
    oa, ob = 2, 2
    v_mat = np.zeros((oa, ob, ia, ib))

    if not behavior:
        # Functional case logic
        k_term = fc_mat[0, 0] / (ia * ib) if ia > 0 and ib > 0 else 0
        for x in range(ia):
            ax_term = fc_mat[1 + x, 0] / ib if ib > 0 else 0
            for y in range(ib):
                by_term = fc_mat[0, 1 + y] / ia if ia > 0 else 0
                axby_term = fc_mat[1 + x, 1 + y]
                # V[0,0,x,y] = P(++,xy) coefficient
                v_mat[0, 0, x, y] = k_term + ax_term + by_term + axby_term
                # V[0,1,x,y] = P(+-,xy) coefficient
                v_mat[0, 1, x, y] = k_term + ax_term - by_term - axby_term
                # V[1,0,x,y] = P(-+,xy) coefficient
                v_mat[1, 0, x, y] = k_term - ax_term + by_term - axby_term
                # V[1,1,x,y] = P(--,xy) coefficient
                v_mat[1, 1, x, y] = k_term - ax_term - by_term + axby_term
    else:
        for x in range(ia):
            ax_val = fc_mat[1 + x, 0]
            for y in range(ib):
                by_val = fc_mat[0, 1 + y]
                axby_val = fc_mat[1 + x, 1 + y]
                # V[0,0,x,y] = P(++,xy) = (1 + <Ax> + <By> + <AxBy>)/4
                v_mat[0, 0, x, y] = 1 + ax_val + by_val + axby_val
                # V[0,1,x,y] = P(+-,xy) = (1 + <Ax> - <By> - <AxBy>)/4
                v_mat[0, 1, x, y] = 1 + ax_val - by_val - axby_val
                # V[1,0,x,y] = P(-+,xy) = (1 - <Ax> + <By> - <AxBy>)/4
                v_mat[1, 0, x, y] = 1 - ax_val + by_val - axby_val
                # V[1,1,x,y] = P(--,xy) = (1 - <Ax> - <By> + <AxBy>)/4
                v_mat[1, 1, x, y] = 1 - ax_val - by_val + axby_val
        v_mat = v_mat / 4

    return v_mat


def _fp_to_cg(v_mat: np.ndarray, behavior: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behavior from Full Probability (FP) to Collins-Gisin (CG) notation.

    The Full Probability (FP) notation represents the full probability distribution
    \(V(a, b, x, y) = P(a, b | x, y)\), where \(a\) (0 to \(oa-1\)), \(b\) (0 to \(ob-1\)) are
    outcomes and \(x\) (0 to \(ia-1\)), \(y\)  (0 to \(ib-1\)) are inputs. It's stored as a 4D
    numpy array \(V[a, b, x, y]\). The Collins-Gisin (CG) notation for a Bell functional or
    behavior is represented by a Collins-Gisin matrix.

    This function converts from FP to CG notation.

    Args:
        v_mat: The probability tensor \(V[a, b, x, y]\) in Full Probability notation.
        behavior: If True, assume input is a behavior (default: False, assume functional).

    Returns:
        The matrix in Collins-Gisin notation.

    !!! Note
        This function is adapted from the QETLAB MATLAB package function ``FP2CG``.
        For ``behavior=True``, it uses the QETLAB convention for calculating marginal probabilities,
        summing over the other party's outcomes for a *fixed* input setting of the other party
        (\(y=0\) for Alice's marginal \(p_A(a|x)\), \(x=0\) for Bob's marginal \(p_B(b|y)\)).

    Examples:
        Consider the CHSH inequality functional in FP notation:
        (Here V represents coefficients, not probabilities)

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _fp_to_cg as fp_to_cg
        chsh_fp = np.zeros((2, 2, 2, 2))
        chsh_fp[0, 0, 0, 0] = 1
        chsh_fp[0, 0, 0, 1] = -1
        chsh_fp[0, 0, 1, 0] = -1
        chsh_fp[0, 0, 1, 1] = 1
        print(fp_to_cg(chsh_fp))
        ```

        Consider a behavior (probability distribution) in FP notation (standard PR box):

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _fp_to_cg as fp_to_cg
        pr_box = np.zeros((2, 2, 2, 2))
        pr_box[0, 0, 0, 0] = 0.5 # p(0,0|0,0)
        pr_box[1, 1, 0, 0] = 0.5 # p(1,1|0,0)
        pr_box[0, 0, 0, 1] = 0.5 # p(0,0|0,1)
        pr_box[1, 1, 0, 1] = 0.5 # p(1,1|0,1)
        pr_box[0, 0, 1, 0] = 0.5 # p(0,0|1,0)
        pr_box[1, 1, 1, 0] = 0.5 # p(1,1|1,0)
        pr_box[0, 1, 1, 1] = 0.5 # p(0,1|1,1)
        pr_box[1, 0, 1, 1] = 0.5 # p(1,0|1,1)
        print(fp_to_cg(pr_box, behavior=True))
        ```

    """
    oa, ob, ia, ib = v_mat.shape

    alice_pars = max(0, ia * (oa - 1)) + 1 if oa > 0 else 0
    bob_pars = max(0, ib * (ob - 1)) + 1 if ob > 0 else 0

    if alice_pars == 0 or bob_pars == 0:
        if behavior:
            raise ValueError("behavior case requires non-zero outputs (oa>0, ob>0).")
        cg_mat = np.zeros((alice_pars, bob_pars))
        return cg_mat

    cg_mat = np.zeros((alice_pars, bob_pars))

    def _cg_row_index(a: int, x: int) -> int:
        r"""Calculate 0-based CG matrix row index for Alice.

        Outcome \(a\) (0..\(oa-2\)) and input \(x\) (0..\(ia-1\)).
        """
        return 1 + a + x * (oa - 1)

    def _cg_col_index(b: int, y: int) -> int:
        r"""Calculate 0-based CG matrix col index for Bob.

        Outcome \(b\) (0..\(ob-2\)) and input \(y\) (0..\(ib-1\)).
        """
        return 1 + b + y * (ob - 1)

    if not behavior:
        # Functional case logic
        cg_mat[0, 0] = np.sum(v_mat[oa - 1, ob - 1, :, :])

        if oa > 1:
            for a in range(oa - 1):
                for x in range(ia):
                    cg_mat[_cg_row_index(a, x), 0] = np.sum(v_mat[a, ob - 1, x, :] - v_mat[oa - 1, ob - 1, x, :])

        if ob > 1:
            for b in range(ob - 1):
                for y in range(ib):
                    cg_mat[0, _cg_col_index(b, y)] = np.sum(v_mat[oa - 1, b, :, y] - v_mat[oa - 1, ob - 1, :, y])

        if oa > 1 and ob > 1:
            for a in range(oa - 1):
                for b in range(ob - 1):
                    for x in range(ia):
                        for y in range(ib):
                            row_idx_0based = _cg_row_index(a, x)
                            col_idx_0based = _cg_col_index(b, y)
                            cg_mat[row_idx_0based, col_idx_0based] = (
                                v_mat[a, b, x, y]
                                - v_mat[a, ob - 1, x, y]
                                - v_mat[oa - 1, b, x, y]
                                + v_mat[oa - 1, ob - 1, x, y]
                            )

    else:
        cg_mat[0, 0] = 1.0  # Set K=1 for behavior

        if oa > 1 and ib > 0:
            for x in range(ia):
                for a in range(oa - 1):
                    target_row_0based = _cg_row_index(a, x)
                    cg_mat[target_row_0based, 0] = np.sum(v_mat[a, :, x, 0])
        elif oa > 1 and ib == 0:
            pass  # Already initialized to 0

        if ob > 1 and ia > 0:
            for y in range(ib):
                for b in range(ob - 1):
                    target_col_0based = _cg_col_index(b, y)
                    cg_mat[0, target_col_0based] = np.sum(v_mat[:, b, 0, y])
        elif ob > 1 and ia == 0:
            pass

        if oa > 1 and ob > 1:
            for x in range(ia):
                for y in range(ib):
                    for a in range(oa - 1):
                        target_row_0based = _cg_row_index(a, x)
                        for b in range(ob - 1):
                            target_col_0based = _cg_col_index(b, y)
                            cg_mat[target_row_0based, target_col_0based] = v_mat[a, b, x, y]

    return cg_mat


def _fp_to_fc(v_mat: np.ndarray, behavior: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behavior from Full Probability (FP) to Full Correlator (FC) notation.

    Assumes binary outcomes (\(oa=2\), \(ob=2\)). The FP tensor indices \(a, b = 0, 1\)
    correspond to physical outcomes \(+1, -1\) respectively.

    The Full Probability (FP) notation represents the full probability distribution
    \(V(a, b, x, y) = P(\text{out}_A=a', \text{out}_B=b' | x, y)\), where
    \(a=0 \rightarrow a'=+1\), \(a=1 \rightarrow a'=-1\) (similarly for \(b\)),
    stored as a 4D numpy array \(V[a, b, x, y]\).
    The Full Correlator (FC) notation is represented by a matrix.

    This function converts from FP to FC notation.

    Args:
        v_mat: The probability tensor \(V[a, b, x, y]\)
                          in Full Probability notation (:math:`oa=2`, :math:`ob=2`).
        behavior: If True, assume input is a behavior (default: False, assume functional).

    Returns:
        The matrix in Full Correlator notation.

    !!! Note
        This function is adapted from the QETLAB MATLAB package function ``FP2FC``.
        For ``behavior=True``, it calculates the *average* marginal correlators \(\langle A_x \rangle\)
        and \(\langle B_y \rangle\) by summing over the other party's inputs
        and dividing by the number of inputs (\(ib\) or \(ia\)).
        The joint correlators \(\langle A_x B_y \rangle\) are calculated directly for each (\(x\), \(y\)).

    Examples:
        Consider the CHSH inequality functional in FP notation:
        (Here V represents coefficients, not probabilities)

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _fp_to_fc as fp_to_fc, fc_to_fp
        chsh_fc = np.array([[0, 0, 0], [0, 0.25, -0.25], [0, -0.25, 0.25]])
        chsh_fp = fc_to_fp(chsh_fc)
        print(fp_to_fc(chsh_fp))
        ```

        Consider a behavior (probability distribution) in FP notation (standard PR box):

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt.bell_inequality_max import _fp_to_fc as fp_to_fc
        pr_box = np.zeros((2, 2, 2, 2))
        pr_box[0, 0, 0, 0] = 0.5 # p(0,0|0,0)
        pr_box[1, 1, 0, 0] = 0.5 # p(1,1|0,0)
        pr_box[0, 0, 0, 1] = 0.5 # p(0,0|0,1)
        pr_box[1, 1, 0, 1] = 0.5 # p(1,1|0,1)
        pr_box[0, 0, 1, 0] = 0.5 # p(0,0|1,0)
        pr_box[1, 1, 1, 0] = 0.5 # p(1,1|1,0)
        pr_box[0, 1, 1, 1] = 0.5 # p(0,1|1,1)
        pr_box[1, 0, 1, 1] = 0.5 # p(1,0|1,1)
        print(fp_to_fc(pr_box, behavior=True))
        ```

    """
    oa, ob, ia, ib = v_mat.shape

    if oa != 2 or ob != 2:
        raise ValueError("FP to FC conversion currently only supports binary outcomes (oa=2, ob=2).")

    fc_mat = np.zeros((1 + ia, 1 + ib))

    fc_mat[0, 0] = np.sum(v_mat)  # K' = sum(V), used for functional case

    for x in range(ia):
        fc_mat[x + 1, 0] = np.sum(v_mat[0, :, x, :]) - np.sum(v_mat[1, :, x, :])

    for y in range(ib):
        fc_mat[0, 1 + y] = np.sum(v_mat[:, 0, :, y]) - np.sum(v_mat[:, 1, :, y])

    # Calculate E[AxBy] for each (x,y) -> FC[x+1, y+1] component
    for x in range(ia):
        for y in range(ib):
            fc_mat[x + 1, y + 1] = v_mat[0, 0, x, y] - v_mat[0, 1, x, y] - v_mat[1, 0, x, y] + v_mat[1, 1, x, y]

    if not behavior:
        fc_mat = fc_mat / 4
    else:
        fc_mat[0, 0] = 1
        if ib > 0:
            fc_mat[1:, 0] = fc_mat[1:, 0] / ib
        else:
            # If no Bob inputs, average marginal <Ax> is 0.
            fc_mat[1:, 0] = 0
        if ia > 0:
            fc_mat[0, 1:] = fc_mat[0, 1:] / ia
        else:
            # If no Alice inputs, average marginal <By> is 0.
            fc_mat[0, 1:] = 0

    return fc_mat
