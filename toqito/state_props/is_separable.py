"""Checks if a quantum state is a separable state."""

import numpy as np
from scipy.linalg import orth

from toqito.channel_ops import partial_channel
from toqito.channels import partial_trace, partial_transpose
from toqito.channels.realignment import realignment
from toqito.matrix_props.is_positive_semidefinite import is_positive_semidefinite
from toqito.matrix_props.trace_norm import trace_norm
from toqito.perms.swap import swap
from toqito.perms.swap_operator import swap_operator
from toqito.state_props.has_symmetric_extension import has_symmetric_extension
from toqito.state_props.in_separable_ball import in_separable_ball
from toqito.state_props.schmidt_rank import schmidt_rank
from toqito.states.max_entangled import max_entangled


def is_separable(state: np.ndarray, dim: None | int | list[int] = None, level: int = 2, tol: float = 1e-8) -> bool:
    r"""Determine if a given state (given as a density matrix) is a separable state [WikiSepSt]_.

    A quantum state :math:`\rho \in \text{D}(\mathcal{H}_A \otimes \mathcal{H}_B)` is called
    separable if it can be written as a convex combination of product states. If a state is not
    separable, it is called entangled.

    Overview
    ==========
    This function implements several criteria to determine separability:

    1. **Input validation**: Checks if the state is positive semi-definite (PSD), square, and trace-normalized to 1.
    2. **Trivial cases**: Returns True if a subsystem is 1D.
    3. **Pure states**: Uses Schmidt rank to determine separability.
    4. **Gurvits-Barnum ball**: Checks if the state lies within a separable ball [1]_.
    5. **PPT criterion (Peres-Horodecki)** [2]_ [3]_:
       - If PPT and :math:`MN \le 6`, the state is separable [3]_.
       - Operational criterion for low-rank PPT states [4]_:
         - If :math:`\text{rank}(\rho) \le \max(\text{dim}_A, \text{dim}_B)`.
         - If :math:`\text{rank}(\rho) + \text{rank}(\rho^{T_A}) \le 2 MN - M - N + 2`.
       - Range criterion variant [5]_.
       - Rank-1 perturbation of identity [6]_.
    6. **3x3 rank-4 PPT check**: Applies necessary and sufficient conditions [7]_.
    7. **Reduction criterion**: Based on [8]_.
    8. **Realignment/CCNR criteria**: Uses matrix realignment [9]_ [10]_.
    9. **2xN specific checks**: For PPT states [11]_ [12]_.
    10. **Decomposable maps**: Applies Ha-Kye [13]_ and Breuer-Hall [14]_ [15]_.
    11. **Symmetric extension hierarchy**: Implements DPS hierarchy [16]_.

    Examples
    ==========
    Consider the following separable (by construction) state:

    .. math::
        \rho = \rho_1 \otimes \rho_2.
        \rho_1 = \frac{1}{2} \left(
            |0 \rangle \langle 0| + |0 \rangle \langle 1| + |1 \rangle \langle 0| + |1 \rangle \langle 1| \right)
        \rho_2 = \frac{1}{2} \left( |0 \rangle \langle 0| + |1 \rangle \langle 1| \right)

    The resulting density matrix will be:

    .. math::
        \rho =  \frac{1}{4} \begin{pmatrix}
                1 & 0 & 1 & 0 \\
                0 & 1 & 0 & 1 \\
                1 & 0 & 1 & 0 \\
                0 & 1 & 0 & 1
                \end{pmatrix} \in \text{D}(\mathcal{X}).

    We provide the input as a density matrix :math:`\rho`.

    On the other hand, a random density matrix will be an entangled state (a separable state).

    .. jupyter-execute::

        import numpy as np
        from toqito.rand.random_density_matrix import random_density_matrix
        from toqito.state_props.is_separable import is_separable
        rho_separable = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        is_separable(rho_separable)

    .. jupyter-execute::

        rho_not_separable = np.array([[ 0.13407875+0.j        , -0.08263926-0.17760437j,
                -0.0135111 -0.12352182j,  0.0368423 -0.05563985j],
            [-0.08263926+0.17760437j,  0.53338542+0.j        ,
                0.19782968-0.04549732j,  0.11287093+0.17024249j],
            [-0.0135111 +0.12352182j,  0.19782968+0.04549732j,
                0.21254612+0.j        , -0.00875865+0.11144344j],
            [ 0.0368423 +0.05563985j,  0.11287093-0.17024249j,
                -0.00875865-0.11144344j,  0.11998971+0.j        ]])
        is_separable(rho_not_separable)

    References
    ==========
    .. [WikiSepSt] Wikipedia: Separable state
       https://en.wikipedia.org/wiki/Separable_state
    .. [1] Gurvits, L. (2002). Classical deterministic complexity of Edmonds'
       problem and quantum entanglement.
       arXiv:quant-ph/0201022.
    .. [2] Peres, A. (1996). Separability criterion for density matrices.
       Physical Review Letters, 77(8), 1413.
    .. [3] Horodecki, M., Horodecki, P., & Horodecki, R. (1996).
       Separability of mixed states: necessary and sufficient conditions.
       Physics Letters A, 223(1-2), 1-8.
    .. [4] Horodecki, M., Horodecki, P., & Horodecki, R. (2000).
       Operational criterion for the separability of low-rank density matrices.
       Physical Review Letters, 84(15), 3494.
    .. [5] Horodecki, P. (1997). Separability criterion and
       inseparable mixed states with positive partial transposition.
       Physics Letters A, 232(5), 333-339.
    .. [6] Vidal, G., & Tarrach, R. (1999). Robustness of entanglement.
       Physical Review A, 59(1), 141.
    .. [7] Breuer, H. P. (2005). Optimal entanglement criterion for mixed
       quantum states.
       Physical Review Letters, 97(8), 080501.
    .. [8] Horodecki, M., & Horodecki, P. (1999). Reduction criterion of
       separability and limits for a class of distillation protocols.
       Physical Review A, 59(6), 4206.
    .. [9] Chen, K., & Wu, L. A. (2003). A matrix realignment method for
       recognizing entanglement.
       Quantum Information & Computation, 3(3), 193-202.
    .. [10] Zhang, C. J., et al. (2008). Entanglement detection beyond the
       computable cross-norm or realignment criterion.
       Physical Review A, 77(6), 062302.
    .. [11] Johnston, N. (2013). Separability from spectrum for qubit-qudit
       states.
       Physical Review A, 88(6), 062330.
    .. [12] Hildebrand, R. (2005). Positive partial transpose from spectra.
       Physical Review A, 72(4), 042321.
    .. [13] Ha, K. C., & Kye, S. H. (2011). Positive maps and entanglement in qudits.
       Physical Review A, 84(2), 022314.
    .. [14] Breuer, H. P. (2006). Optimal entanglement criterion for mixed
       quantum states.
       Physical Review Letters, 97(8), 080501.
    .. [15] Hall, W. (2006). A new criterion for indecomposability of positive maps.
       Journal of Physics A: Mathematical and General, 39(46), 14119.
    .. [16] Doherty, A. C., Parrilo, P. A., & Spedalieri, F. M. (2004).
       Complete family of separability criteria.
       Physical Review A, 69(2), 022308.

    .. bibliography::
        :filter: docname in docnames
    :raises ValueError: If dimension is not specified correctly or input is invalid.
    :param state: The density matrix to check.
    :param dim: The dimension of the input state, e.g., [dim_A, dim_B]. Optional; inferred if None.
    :param level: The level for symmetric extensions (default: 2).
    :param tol: Numerical tolerance (default: 1e-8).
    :return: True if separable, False if entangled or inconclusive by implemented checks.

    """
    # --- 1. Input Validation, Normalization, Dimension Setup ---
    if not isinstance(state, np.ndarray):
        raise TypeError("Input state must be a NumPy array.")
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("Input state must be a square matrix.")

    if np.issubdtype(state.dtype, np.complexfloating):
        _machine_eps = np.finfo(state.real.dtype).eps
    elif np.issubdtype(state.dtype, np.floating):
        _machine_eps = np.finfo(state.dtype).eps
    else:
        _machine_eps = np.finfo(float).eps

    state_len = state.shape[0]

    if not is_positive_semidefinite(state, atol=tol, rtol=tol):
        raise ValueError("Checking separability of non-positive semidefinite matrix is invalid.")

    trace_state_val = np.trace(state)
    current_state = state.copy()

    if state_len > 0 and abs(trace_state_val) < tol:
        if np.any(
            np.abs(current_state)
            > 100 * _machine_eps * max(1, np.max(np.abs(current_state)) if current_state.size > 0 else 1)
        ):
            raise ValueError("Trace of the input state is close to zero, but state is not zero matrix.")

    if abs(trace_state_val - 1) > tol:
        if abs(trace_state_val) > 100 * _machine_eps:
            current_state = current_state / trace_state_val
        elif state_len > 0 and np.any(np.abs(current_state) > tol):  # pragma: no cover (Hard to hit with PSD)
            raise ValueError(
                "State has numerically insignificant trace but significant elements; cannot normalize reliably."
            )

    # Dimension processing
    temp_dim_param = dim
    if temp_dim_param is None:
        if state_len == 0:
            dims_arr_val = np.array([0, 0])
        elif state_len == 1:
            dims_arr_val = np.array([1, 1])
        else:
            dim_A_guess = int(np.round(np.sqrt(state_len)))
            # if dim_A_guess == 0 and state_len > 0: # pragma: no cover (dim_A_guess unlikely 0 if state_len > 0)
            #     dim_A_guess = 1
            if dim_A_guess > 0 and state_len % dim_A_guess == 0:  # Simplified from state_len > 0 here
                dims_arr_val = np.array([dim_A_guess, state_len / dim_A_guess])
                if (
                    np.abs(dims_arr_val[1] - np.round(dims_arr_val[1])) >= 2 * state_len * _machine_eps
                    and state_len > 0
                ):
                    found_factor = False
                    for dA_try in range(2, int(np.sqrt(state_len)) + 1):
                        if state_len % dA_try == 0:
                            dims_arr_val = np.array([dA_try, state_len // dA_try])
                            found_factor = True
                            break
                    if not found_factor:
                        dims_arr_val = np.array([1, state_len])
                else:
                    dims_arr_val[1] = np.round(dims_arr_val[1])
            elif state_len > 0:
                found_factor = False
                for dA_try in range(2, int(np.sqrt(state_len)) + 1):
                    if state_len % dA_try == 0:
                        dims_arr_val = np.array([dA_try, state_len // dA_try])
                        found_factor = True
                        break
                if not found_factor:
                    dims_arr_val = np.array([1, state_len])
            else:
                dims_arr_val = np.array([0, 0])
        dims_list = [int(d) for d in dims_arr_val]
    elif isinstance(temp_dim_param, int):
        if temp_dim_param <= 0:
            if state_len == 0 and temp_dim_param == 0:
                dims_list = [0, 0]
            else:
                raise ValueError(
                    "Integer `dim` (interpreted as dim_A) must be positive "
                    + "for non-empty states or zero for empty states."
                )
        elif state_len == 0 and temp_dim_param != 0:
            raise ValueError(f"Cannot apply positive dimension {temp_dim_param} to zero-sized state.")
        elif state_len > 0 and temp_dim_param > 0 and state_len % temp_dim_param != 0:
            raise ValueError("The parameter `dim` must evenly divide the length of the state.")
        else:
            dims_list = [int(temp_dim_param), int(np.round(state_len / temp_dim_param))]
    elif isinstance(temp_dim_param, list) and len(temp_dim_param) == 2:
        if not all(isinstance(d, (int, np.integer)) and d >= 0 for d in temp_dim_param):
            raise ValueError("Dimensions in list must be non-negative integers.")
        if temp_dim_param[0] * temp_dim_param[1] != state_len:
            if (temp_dim_param[0] == 0 or temp_dim_param[1] == 0) and state_len != 0:
                raise ValueError("Non-zero state with zero-dim subsystem is inconsistent.")
            raise ValueError("Product of list dimensions must equal state length.")
        dims_list = [int(d) for d in temp_dim_param]
    else:
        raise ValueError("`dim` must be None, an int, or a list of two non-negative integers.")

    dA, dB = dims_list[0], dims_list[1]
    if (dA == 0 or dB == 0) and state_len != 0:
        raise ValueError("Non-zero state with zero-dim subsystem is inconsistent.")

    if state_len == 0:
        return True

    state_rank = np.linalg.matrix_rank(current_state, tol=tol)
    min_dim_val, max_dim_val = min(dA, dB), max(dA, dB)
    prod_dim_val = dA * dB

    if prod_dim_val == 0 and state_len > 0:
        raise ValueError("Zero product dimension for non-empty state is inconsistent.")
    if prod_dim_val > 0 and prod_dim_val != state_len:
        raise ValueError(f"Internal dimension calculation error: prod_dim {prod_dim_val} != state_len {state_len}")

    # --- 2. Trivial Cases for Separability ---
    if min_dim_val == 1:
        return True

    if state_rank == 1:
        s_rank = schmidt_rank(current_state, dims_list)
        return bool(s_rank == 1)

    if in_separable_ball(current_state):
        return True

    # --- PPT (Peres-Horodecki) Criterion ---
    rho_pt_B = partial_transpose(current_state, sys=1, dim=dims_list)
    is_state_ppt = is_positive_semidefinite(rho_pt_B, atol=tol, rtol=tol)

    if not is_state_ppt:
        return False

    # ----- From here on, the state is known to be PPT -----

    if prod_dim_val <= 6:
        return True

    # --- Specific N&S Check for 3x3 Rank 4 PPT States (Plucker/Breuer) ---
    if dA == 3 and dB == 3 and state_rank == 4:
        q_orth_basis = orth(current_state)
        if q_orth_basis.shape[1] < 4:
            pass
        else:
            p_np_arr = np.zeros((6, 7, 8, 9), dtype=complex)
            for j_breuer in range(6, 0, -1):
                for k_breuer in range(7, 0, -1):
                    for n_breuer in range(8, 0, -1):
                        for m_breuer in range(9, 0, -1):
                            if j_breuer < k_breuer < n_breuer < m_breuer:
                                selected_rows = [idx - 1 for idx in [j_breuer, k_breuer, n_breuer, m_breuer]]
                                if all(0 <= r_idx < q_orth_basis.shape[0] for r_idx in selected_rows):
                                    sub_matrix = q_orth_basis[selected_rows, :]
                                    if sub_matrix.shape[0] == 4 and sub_matrix.shape[1] == 4:
                                        try:
                                            p_np_arr[j_breuer - 1, k_breuer - 1, n_breuer - 1, m_breuer - 1] = (
                                                np.linalg.det(sub_matrix)
                                            )
                                        except np.linalg.LinAlgError:
                                            p_np_arr[j_breuer - 1, k_breuer - 1, n_breuer - 1, m_breuer - 1] = np.nan

            def get_p(t_tuple):
                # try:
                val = p_np_arr[t_tuple[0] - 1, t_tuple[1] - 1, t_tuple[2] - 1, t_tuple[3] - 1]
                return val if not np.isnan(val) else 0.0
                # except IndexError:
                #     return 0.0

            F_det_matrix_elements = [
                [
                    get_p((1, 2, 4, 5)),
                    get_p((1, 3, 4, 6)),
                    get_p((2, 3, 5, 6)),
                    get_p((1, 2, 4, 6)) + get_p((1, 3, 4, 5)),
                    get_p((1, 2, 5, 6)) + get_p((2, 3, 4, 5)),
                    get_p((1, 3, 5, 6)) + get_p((2, 3, 4, 6)),
                ],
                [
                    get_p((1, 2, 7, 8)),
                    get_p((1, 3, 7, 9)),
                    get_p((2, 3, 8, 9)),
                    get_p((1, 2, 7, 9)) + get_p((1, 3, 7, 8)),
                    get_p((1, 2, 8, 9)) + get_p((2, 3, 7, 8)),
                    get_p((1, 3, 8, 9)) + get_p((2, 3, 7, 9)),
                ],
                [
                    get_p((4, 5, 7, 8)),
                    get_p((4, 6, 7, 9)),
                    get_p((5, 6, 8, 9)),
                    get_p((4, 5, 7, 9)) + get_p((4, 6, 7, 8)),
                    get_p((4, 5, 8, 9)) + get_p((5, 6, 7, 8)),
                    get_p((4, 6, 8, 9)) + get_p((5, 6, 7, 9)),
                ],
                [
                    get_p((1, 2, 4, 8)) - get_p((1, 2, 5, 7)),
                    get_p((1, 3, 4, 9)) - get_p((1, 3, 6, 7)),
                    get_p((2, 3, 5, 9)) - get_p((2, 3, 6, 8)),
                    get_p((1, 2, 4, 9)) - get_p((1, 2, 6, 7)) + get_p((1, 3, 4, 8)) - get_p((1, 3, 5, 7)),
                    get_p((1, 2, 5, 9)) - get_p((1, 2, 6, 8)) + get_p((2, 3, 4, 8)) - get_p((2, 3, 5, 7)),
                    get_p((1, 3, 5, 9)) - get_p((1, 3, 6, 8)) + get_p((2, 3, 4, 9)) - get_p((2, 3, 6, 7)),
                ],
                [
                    get_p((1, 4, 5, 8)) - get_p((2, 4, 5, 7)),
                    get_p((1, 4, 6, 9)) - get_p((3, 4, 6, 7)),
                    get_p((2, 5, 6, 9)) - get_p((3, 5, 6, 8)),
                    get_p((1, 4, 5, 9)) - get_p((2, 4, 6, 7)) + get_p((1, 4, 6, 8)) - get_p((3, 4, 5, 7)),
                    get_p((1, 5, 6, 8)) - get_p((2, 5, 6, 7)) + get_p((2, 4, 5, 9)) - get_p((3, 4, 5, 8)),
                    get_p((1, 5, 6, 9)) - get_p((3, 4, 6, 8)) + get_p((2, 4, 6, 9)) - get_p((3, 5, 6, 7)),
                ],
                [
                    get_p((1, 5, 7, 8)) - get_p((2, 4, 7, 8)),
                    get_p((1, 6, 7, 9)) - get_p((3, 4, 7, 9)),
                    get_p((2, 6, 8, 9)) - get_p((3, 5, 8, 9)),
                    get_p((1, 5, 7, 9)) - get_p((2, 4, 7, 9)) + get_p((1, 6, 7, 8)) - get_p((3, 4, 7, 8)),
                    get_p((1, 5, 8, 9)) - get_p((2, 4, 8, 9)) + get_p((2, 6, 7, 8)) - get_p((3, 5, 7, 8)),
                    get_p((1, 6, 8, 9)) - get_p((3, 4, 8, 9)) + get_p((2, 6, 7, 9)) - get_p((3, 5, 7, 9)),
                ],
            ]
            try:
                F_det_val = np.linalg.det(np.array(F_det_matrix_elements, dtype=complex))
                return bool(abs(F_det_val) < max(tol**2, _machine_eps ** (3 / 4)))
            except np.linalg.LinAlgError:
                pass

    # --- Horodecki Operational Criterion for Low-Rank PPT States (General) ---
    if state_rank <= max_dim_val:
        return True

    rho_pt_A = partial_transpose(current_state, sys=0, dim=dims_list)
    rank_pt_A = np.linalg.matrix_rank(rho_pt_A, tol=tol)
    threshold_horodecki = 2 * prod_dim_val - dA - dB + 2
    if state_rank + rank_pt_A <= threshold_horodecki:
        return True

    # --- Other Sufficient Conditions for PPT states & Entanglement Witnesses ---
    rho_A_marginal = partial_trace(current_state, sys=[1], dim=dims_list)
    rho_B_marginal = partial_trace(current_state, sys=[0], dim=dims_list)

    # if state_rank <= np.linalg.matrix_rank(rho_A_marginal, tol=tol)
    # or state_rank <= np.linalg.matrix_rank( # pragma: no cover (logically problematic branch)
    #     rho_B_marginal, tol=tol
    # ):
    #     return True

    op_reduct1 = np.kron(np.eye(dA), rho_B_marginal) - current_state
    op_reduct2 = np.kron(rho_A_marginal, np.eye(dB)) - current_state
    if not (
        is_positive_semidefinite(op_reduct1, atol=tol, rtol=tol)
        and is_positive_semidefinite(op_reduct2, atol=tol, rtol=tol)
    ):
        return False

    if trace_norm(realignment(current_state, dims_list)) > 1 + tol:
        return False

    tr_rho_A_sq = np.real(np.trace(rho_A_marginal @ rho_A_marginal))
    tr_rho_B_sq = np.real(np.trace(rho_B_marginal @ rho_B_marginal))
    val_A = max(0, 1 - tr_rho_A_sq)
    val_B = max(0, 1 - tr_rho_B_sq)
    bound_zhang = np.sqrt(val_A * val_B) if (val_A * val_B >= 0) else 0
    if trace_norm(realignment(current_state - np.kron(rho_A_marginal, rho_B_marginal), dims_list)) > bound_zhang + tol:
        return False

    try:
        try:
            lam = np.linalg.eigvalsh(current_state)[::-1]
        except np.linalg.LinAlgError:
            lam = np.sort(np.real(np.linalg.eigvals(current_state)))[::-1]

        if len(lam) == prod_dim_val and prod_dim_val > 1:
            if lam[1] - lam[prod_dim_val - 1] < (tol**2 + 2 * _machine_eps):
                return True
    except np.linalg.LinAlgError:
        pass

    if min_dim_val == 2 and prod_dim_val > 0:
        state_t_2xn, d_N_val, sys_to_pt_for_hildebrand_map, dim_for_hildebrand_map = (
            current_state,
            max_dim_val,
            1,
            [2, max_dim_val],
        )
        if dA != 2 and dB == 2:
            state_t_2xn = swap(current_state, sys=[0, 1], dim=dims_list)
        elif dA == 2:
            pass
        else:
            state_t_2xn = None

        if state_t_2xn is not None:
            current_lam_2xn = lam
            if state_t_2xn is not current_state:  # If swap occurred
                try:
                    current_lam_2xn = np.linalg.eigvalsh(state_t_2xn)[::-1]
                except np.linalg.LinAlgError:
                    current_lam_2xn = np.sort(np.real(np.linalg.eigvals(state_t_2xn)))[::-1]

            # Ensure indices are valid for current_lam_2xn
            if (
                len(current_lam_2xn) >= 2 * d_N_val
                and (2 * d_N_val - 1) < len(current_lam_2xn)
                and (2 * d_N_val - 2) >= 0
                and (2 * d_N_val - 3) >= 0
            ):
                if (current_lam_2xn[0] - current_lam_2xn[2 * d_N_val - 2]) ** 2 <= 4 * current_lam_2xn[
                    2 * d_N_val - 3
                ] * current_lam_2xn[2 * d_N_val - 1] + tol**2:
                    return True

            A_block, B_block, C_block = (
                state_t_2xn[:d_N_val, :d_N_val],
                state_t_2xn[:d_N_val, d_N_val : 2 * d_N_val],
                state_t_2xn[d_N_val : 2 * d_N_val, d_N_val : 2 * d_N_val],
            )
            if B_block.size > 0 and np.linalg.matrix_rank(B_block - B_block.conj().T, tol=tol) <= 1:
                return True  # is_state_ppt is True
            if A_block.size > 0 and B_block.size > 0 and C_block.size > 0:
                X_2n_ppt_check = np.vstack(
                    (
                        np.hstack(((5 / 6) * A_block - C_block / 6, B_block)),
                        np.hstack((B_block.conj().T, (5 / 6) * C_block - A_block / 6)),
                    )
                )
                if is_positive_semidefinite(X_2n_ppt_check, atol=tol, rtol=tol) and is_positive_semidefinite(
                    partial_transpose(X_2n_ppt_check, sys=sys_to_pt_for_hildebrand_map, dim=dim_for_hildebrand_map),
                    atol=tol,
                    rtol=tol,
                ):  # Check PPT of this map
                    return True
                try:
                    eig_A_real, eig_C_real = np.real(np.linalg.eigvals(A_block)), np.real(np.linalg.eigvals(C_block))
                    if eig_A_real.size > 0 and eig_C_real.size > 0 and B_block.size > 0:
                        if np.linalg.norm(B_block) ** 2 <= np.min(eig_A_real) * np.min(eig_C_real) + tol**2:
                            return True
                except np.linalg.LinAlgError:
                    pass

    if dA == 3 and dB == 3:
        phi_me3 = max_entangled(3, False, False)
        phi_proj3 = phi_me3 @ phi_me3.conj().T
        for t_raw_ha in np.arange(0.1, 1.0, 0.1):  # Avoid t=0
            t_iter_ha = t_raw_ha
            for j_ha_loop in range(2):  # Iterate for t and 1/t
                if j_ha_loop == 1:
                    # if abs(t_raw_ha) < _machine_eps: # pragma: no cover (t_raw_ha always >= 0.1)
                    #     break  # Should not happen with arange
                    t_iter_ha = 1 / t_raw_ha
                denom_ha = 1 - t_iter_ha + t_iter_ha**2
                if abs(denom_ha) < _machine_eps:  # pragma: no cover (denom_ha = 1-t+t^2 > 0)
                    continue  # Avoid division by zero
                a, b, c = (1 - t_iter_ha) ** 2 / denom_ha, t_iter_ha**2 / denom_ha, 1 / denom_ha
                Phi_map_ha = np.diag([a + 1, c, b, b, a + 1, c, c, b, a + 1]) - phi_proj3
                if not is_positive_semidefinite(
                    partial_channel(current_state, Phi_map_ha, sys=1, dim=dims_list), atol=tol, rtol=tol
                ):
                    return False

    for p_idx_bh in range(2):  # Breuer-Hall Loop
        current_dim_bh = dims_list[p_idx_bh]
        if current_dim_bh > 0 and current_dim_bh % 2 == 0:
            phi_me_bh = max_entangled(current_dim_bh, False, False)
            phi_proj_bh = phi_me_bh @ phi_me_bh.conj().T
            half_dim_bh = current_dim_bh // 2
            diag_U_elems_bh = np.concatenate([np.ones(half_dim_bh), -np.ones(half_dim_bh)])
            U_bh_kron_part = np.fliplr(np.diag(diag_U_elems_bh))
            U_for_phi_constr = np.kron(np.eye(current_dim_bh), U_bh_kron_part)
            Phi_bh_map = (
                np.eye(current_dim_bh**2)
                - phi_proj_bh
                - U_for_phi_constr @ swap_operator(current_dim_bh) @ U_for_phi_constr.conj().T
            )
            mapped_state_bh = partial_channel(current_state, Phi_bh_map, sys=p_idx_bh, dim=dims_list)
            is_psd_after_bh = is_positive_semidefinite(mapped_state_bh, atol=tol, rtol=tol)

            if not is_psd_after_bh:
                return False

    if level >= 2:
        for k_actual_level_check in range(2, level + 1):
            # try:
            if has_symmetric_extension(rho=current_state, level=k_actual_level_check, dim=dims_list, tol=tol):
                return True
            # except ImportError:
            #     break
            # except Exception:
            #     pass
    elif level == 1 and is_state_ppt:
        return True  # If only k_level=1 SES requested and state is PPT

    return False
