"""Checks if a quantum state is a separable state."""

from itertools import product

import numpy as np
from scipy.linalg import orth

from toqito.channel_ops.partial_channel import partial_channel
from toqito.channels.realignment import realignment
from toqito.matrix_ops import partial_trace
from toqito.matrix_ops import partial_transpose as pt_func
from toqito.matrix_props.is_positive_semidefinite import is_positive_semidefinite
from toqito.matrix_props.trace_norm import trace_norm
from toqito.perms.swap import swap
from toqito.perms.swap_operator import swap_operator
from toqito.state_props.has_symmetric_extension import has_symmetric_extension
from toqito.state_props.in_separable_ball import in_separable_ball
from toqito.state_props.is_ppt import is_ppt
from toqito.state_props.schmidt_rank import schmidt_rank
from toqito.states.max_entangled import max_entangled

EPS = np.finfo(float).eps


# Helper functions (assuming these are correct from previous versions)
def _check_pure_and_product(state: np.ndarray, dim: list[int]) -> bool:
    return schmidt_rank(state, dim=dim) == 1


def _check_horodecki_operational(
    rho: np.ndarray, rho_pt_A: np.ndarray, dims: list[int], rank_rho: int, tol: float
) -> bool:
    dA, dB = dims[0], dims[1]
    if rank_rho <= max(dA, dB):
        return True
    rank_rho_pt_A = np.linalg.matrix_rank(rho_pt_A, tol)
    threshold = 2 * dA * dB - dA - dB + 2
    return (rank_rho + rank_rho_pt_A) <= threshold


def _check_reduction(rho: np.ndarray, dims: list[int], rtol_param: float, atol_param: float) -> bool:
    dA, dB = dims[0], dims[1]
    rho_A = partial_trace(rho, sys=1, dim=dims)
    rho_B = partial_trace(rho, sys=0, dim=dims)
    I_A, I_B = np.eye(dA), np.eye(dB)
    op1, op2 = np.kron(I_A, rho_B) - rho, np.kron(rho_A, I_B) - rho
    return is_positive_semidefinite(op1, rtol=rtol_param, atol=atol_param) and is_positive_semidefinite(
        op2, rtol=rtol_param, atol=atol_param
    )


def _check_realignment_ccnr(rho: np.ndarray, dims: list[int], tol: float) -> bool:
    R_rho = realignment(rho, dim=dims)
    return trace_norm(R_rho) <= 1 + tol


# Main function
def is_separable(state: np.ndarray, dim: None | int | list[int] = None, level: int = 2, tol: float = 1e-8) -> bool:
    r"""Determine if a given state (given as a density matrix) is a separable state :cite:`WikiSepSt`.

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
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If dimension is not specified.
    :param state: The matrix to check.
    :param dim: The dimension of the input.
    :param level: The level up to which to search for the symmetric extensions.
    :param tol: Numerical tolerance used.
    :return: :code:`True` if :code:`rho` is separabale and :code:`False` otherwise.

    """
    # 1. Input validation and preprocessing
    if not isinstance(state, np.ndarray):
        raise TypeError("Input state must be a NumPy array.")
    if state.ndim != 2:
        raise ValueError("Input state must be a 2D matrix.")
    state_len = state.shape[0]
    if state.shape[1] != state_len:
        raise ValueError("Input state must be a square matrix.")
    if not is_positive_semidefinite(state, rtol=tol, atol=tol):
        raise ValueError("Input state must be a positive semidefinite matrix.")
    trace_state = np.trace(state)
    if abs(trace_state) < tol:
        raise ValueError("Trace of the input state is close to zero.")
    if abs(trace_state - 1) > tol:
        state = state / trace_state

    temp_dim_param = dim
    if temp_dim_param is None:
        temp_dim_param = int(np.round(np.sqrt(state_len)))
    if isinstance(temp_dim_param, int):
        if temp_dim_param <= 0:
            raise ValueError("Integer `dim` must be positive.")
        if state_len % temp_dim_param != 0:
            if dim is None and state_len > 1 and all(state_len % i != 0 for i in range(2, int(np.sqrt(state_len)) + 1)):
                raise ValueError(f"State dimension {state_len} is prime > 1; system cannot be bipartite.")
            raise ValueError(f"Integer `dim` ({temp_dim_param}) must evenly divide state dimension ({state_len}).")
        dims_arr = np.array([temp_dim_param, state_len // temp_dim_param])
    elif isinstance(temp_dim_param, list) and len(temp_dim_param) == 2:
        dims_arr = np.array(temp_dim_param)
        if not all(isinstance(d, (int, np.integer)) and d > 0 for d in dims_arr):
            raise ValueError("Dimensions in `dim` list must be positive integers.")
        if dims_arr[0] * dims_arr[1] != state_len:
            raise ValueError("Product of dimensions in `dim` list must equal state dimension.")
    else:
        raise ValueError("`dim` must be an int, a list of two positive ints, or None.")

    dims_list = [int(d) for d in dims_arr]
    dA, dB = dims_list[0], dims_list[1]
    min_d, prod_dim = min(dA, dB), dA * dB
    state_rank = np.linalg.matrix_rank(state, tol=tol)

    # Section 1: Quick Conclusive Separability Checks
    if min_d == 1:
        return True
    if _check_pure_and_product(state, dims_list):
        return True
    if in_separable_ball(state):
        return True

    # Section 2: PPT Criterion (Primary Necessary Condition)
    ppt_status = is_ppt(state, sys=2, dim=dims_list, tol=tol)
    if not ppt_status:
        return False  # ENTANGLED by Peres-Horodecki

    # At this point, state IS PPT.

    # Section 3: Conclusive N&S Criteria for specific PPT states
    if prod_dim <= 6:
        return True  # SEPARABLE (PPT sufficient for 2x2, 2x3)

    if state_rank == 4 and dA == 3 and dB == 3:  # Chen/Johnston 3x3 rank-4 PPT N&S test
        p_arr = np.zeros((prod_dim - 3, prod_dim - 2, prod_dim - 1, prod_dim))
        q_orth = orth(state)
        idx_range = list(range(1, prod_dim + 1))
        for j, k, l_var, m_var in product(idx_range, idx_range, idx_range, idx_range):
            if j < k < l_var < m_var:
                if (
                    j - 1 < p_arr.shape[0]
                    and k - 1 < p_arr.shape[1]
                    and l_var - 1 < p_arr.shape[2]
                    and m_var - 1 < p_arr.shape[3]
                ):
                    p_arr[j - 1, k - 1, l_var - 1, m_var - 1] = np.linalg.det(
                        q_orth[[j - 1, k - 1, l_var - 1, m_var - 1], :]
                    )

        F_det_matrix_elements = [  # Using elements as defined in the original code
            [
                p_arr[0, 1, 3, 4],
                p_arr[0, 2, 3, 5],
                p_arr[1, 2, 4, 5],
                p_arr[0, 1, 3, 5] + p_arr[0, 2, 3, 4],
                p_arr[0, 1, 4, 5] + p_arr[1, 2, 3, 4],
                p_arr[0, 2, 4, 5] + p_arr[1, 2, 3, 5],
            ],
            [
                p_arr[0, 1, 6, 7],
                p_arr[0, 2, 6, 8],
                p_arr[1, 2, 7, 8],
                p_arr[0, 1, 6, 8] + p_arr[0, 2, 6, 7],
                p_arr[0, 1, 7, 8] + p_arr[1, 2, 6, 7],
                p_arr[0, 2, 7, 8] + p_arr[1, 2, 6, 8],
            ],
            [
                p_arr[3, 4, 6, 7],
                p_arr[3, 5, 6, 8],
                p_arr[4, 5, 7, 8],
                p_arr[3, 4, 6, 8] + p_arr[3, 5, 6, 7],
                p_arr[3, 4, 7, 8] + p_arr[4, 5, 6, 7],
                p_arr[3, 5, 7, 8] + p_arr[4, 5, 6, 8],
            ],
            [
                p_arr[0, 1, 3, 7] - p_arr[0, 1, 4, 6],
                p_arr[0, 2, 3, 8] - p_arr[0, 2, 5, 6],
                p_arr[1, 2, 4, 8] - p_arr[1, 2, 5, 7],
                p_arr[0, 1, 3, 8] - p_arr[0, 1, 5, 6] + p_arr[0, 2, 3, 7] - p_arr[0, 2, 4, 6],
                p_arr[0, 1, 4, 8] - p_arr[0, 1, 5, 7] + p_arr[1, 2, 3, 7] - p_arr[1, 2, 4, 6],
                p_arr[0, 2, 4, 8] - p_arr[0, 2, 5, 7] + p_arr[1, 2, 3, 8] - p_arr[1, 2, 5, 6],
            ],
            [
                p_arr[0, 3, 4, 7] - p_arr[1, 3, 4, 6],
                p_arr[0, 3, 5, 8] - p_arr[2, 3, 5, 6],
                p_arr[1, 4, 5, 8] - p_arr[2, 4, 5, 7],
                p_arr[0, 3, 4, 8] - p_arr[1, 3, 5, 6] + p_arr[0, 3, 5, 7] - p_arr[2, 3, 4, 6],
                p_arr[0, 4, 5, 7] - p_arr[1, 4, 5, 6] + p_arr[1, 3, 4, 8] - p_arr[2, 3, 4, 7],
                p_arr[0, 4, 5, 8] - p_arr[2, 3, 5, 7] + p_arr[1, 3, 5, 8] - p_arr[2, 4, 5, 6],
            ],
            [
                p_arr[0, 4, 6, 7] - p_arr[1, 3, 6, 7],
                p_arr[0, 5, 6, 8] - p_arr[2, 3, 6, 8],
                p_arr[1, 5, 7, 8] - p_arr[2, 4, 7, 8],
                p_arr[0, 4, 6, 8] - p_arr[1, 3, 6, 8] + p_arr[0, 5, 6, 7] - p_arr[2, 3, 6, 7],
                p_arr[0, 4, 7, 8] - p_arr[1, 3, 7, 8] + p_arr[1, 5, 6, 7] - p_arr[2, 4, 6, 7],
                p_arr[0, 5, 7, 8] - p_arr[2, 3, 7, 8] + p_arr[1, 5, 6, 8] - p_arr[2, 4, 6, 8],
            ],
        ]
        F_det_val = np.linalg.det(np.array(F_det_matrix_elements, dtype=complex))  # Ensure complex dtype for det
        return abs(F_det_val) < max(tol**2, EPS ** (3 / 4))  # Returns True if separable, False if entangled

    # Section 4: Strong Necessary Conditions for Entanglement (for PPT states)
    if not _check_reduction(state, dims_list, rtol_param=tol, atol_param=tol):
        return False
    if not _check_realignment_ccnr(state, dims_list, tol=tol):
        return False

    rho_A_marg = partial_trace(state, sys=1, dim=dims_list)
    rho_B_marg = partial_trace(state, sys=0, dim=dims_list)
    tr_rho_A_sq = np.trace(rho_A_marg @ rho_A_marg)
    tr_rho_B_sq = np.trace(rho_B_marg @ rho_B_marg)
    bound_zhang = np.sqrt(max(0, (1 - tr_rho_A_sq) * (1 - tr_rho_B_sq)))  # Ensure non-negative under sqrt
    if trace_norm(realignment(state - np.kron(rho_A_marg, rho_B_marg), dim=dims_list)) > bound_zhang + tol:
        return False

    # Positive Maps (Ha for 3x3, Breuer-Hall for even dimensions)
    if dA == 3 and dB == 3:  # Ha maps
        phi_me3 = max_entangled(3, is_normalized=False, is_sparse=False)
        phi_proj3 = phi_me3 @ phi_me3.conj().T
        for t_param_ha in np.arange(0, 1.0, 0.1):  # Iterate through map parameters
            # ... (Ha map construction as before)
            t_map_val = t_param_ha
            for j_iter_ha in range(2):  # Covers t and 1/t if t > 0
                if j_iter_ha == 1:
                    if t_param_ha > EPS:
                        t_map_val = 1 / t_param_ha
                    else:
                        break
                denom_ha = 1 - t_map_val + t_map_val**2
                if abs(denom_ha) < EPS:
                    continue  # Avoid division by zero if somehow
                a_ha = (1 - t_map_val) ** 2 / denom_ha
                b_ha = t_map_val**2 / denom_ha
                c_ha = 1 / denom_ha
                Phi_ha_choi = np.diag([a_ha + 1, c_ha, b_ha, b_ha, a_ha + 1, c_ha, c_ha, b_ha, a_ha + 1]) - phi_proj3
                if not is_positive_semidefinite(
                    partial_channel(state, Phi_ha_choi, sys=1, dim=dims_list), rtol=tol, atol=tol
                ):
                    return False  # ENTANGLED

    for p_sys_idx in range(2):  # Breuer-Hall maps
        dim_p_bh = dims_list[p_sys_idx]
        if dim_p_bh % 2 == 0:
            # ... (Breuer-Hall map construction as before)
            phi_me_bh = max_entangled(dim_p_bh, is_normalized=False, is_sparse=False)
            phi_proj_bh = phi_me_bh @ phi_me_bh.conj().T
            diag_elems_bh = np.concatenate([np.ones(dim_p_bh // 2), -np.ones(dim_p_bh // 2)])
            J_antisym_bh = np.fliplr(np.diag(diag_elems_bh))
            U_op_bh = np.kron(np.eye(dim_p_bh), J_antisym_bh)
            Phi_BH_Choi = np.eye(dim_p_bh**2) - phi_proj_bh - U_op_bh @ swap_operator(dim_p_bh) @ U_op_bh.conj().T
            if not is_positive_semidefinite(
                partial_channel(state, Phi_BH_Choi, sys=p_sys_idx, dim=dims_list), rtol=tol, atol=tol
            ):
                return False  # ENTANGLED

    # Section 5: Sufficient Separability Conditions (for PPT states that passed prior nec. checks)
    rho_pt_A = pt_func(state, sys=0, dim=dims_list)
    if _check_horodecki_operational(state, rho_pt_A, dims_list, state_rank, tol=tol):
        return True

    eig_vals = np.linalg.eigvalsh(state)  # Used by multiple checks below
    lam = np.sort(eig_vals)[::-1]

    if min_d == 2:  # 2xN specific separability conditions
        max_d_for_2xn = max(dA, dB)
        if len(lam) > (2 * max_d_for_2xn - 1):
            if (lam[0] - lam[2 * max_d_for_2xn - 2]) ** 2 <= 4 * lam[2 * max_d_for_2xn - 3] * lam[
                2 * max_d_for_2xn - 1
            ] + tol**2:
                return True  # SEPARABLE (Johnston 2013 Spectrum Eq. 12)

        state_t, dims_t = state, list(dims_list)
        if dims_list[0] != 2:  # Ensure 2D system is first for block structure
            state_t = swap(state, sys=[0, 1], dim=dims_list)
            dims_t[0], dims_t[1] = dims_t[1], dims_t[0]

        A_block = state_t[:max_d_for_2xn, :max_d_for_2xn]
        B_block = state_t[:max_d_for_2xn, max_d_for_2xn : 2 * max_d_for_2xn]
        C_block = state_t[max_d_for_2xn : 2 * max_d_for_2xn, max_d_for_2xn : 2 * max_d_for_2xn]

        if (
            np.linalg.matrix_rank(B_block - B_block.conj().T, tol=tol) <= 1
        ):  # Hildebrand: if PPT and this, then separable
            return True

        X_2n_hildebrand = np.vstack(
            (
                np.hstack(((5 / 6) * A_block - C_block / 6, B_block)),
                np.hstack((B_block.conj().T, (5 / 6) * C_block - A_block / 6)),
            )
        )
        if is_positive_semidefinite(X_2n_hildebrand, rtol=tol, atol=tol) and is_ppt(
            X_2n_hildebrand, sys=0, dim=[2, max_d_for_2xn], tol=tol
        ):  # Check PPT of this transformed op
            return True

        min_eig_A = np.min(np.linalg.eigvalsh(A_block)) if A_block.size > 0 else 0
        min_eig_C = np.min(np.linalg.eigvalsh(C_block)) if C_block.size > 0 else 0
        if B_block.size > 0 and np.linalg.norm(B_block) ** 2 <= min_eig_A * min_eig_C + tol**2:  # Johnston Lemma 1
            return True

    if len(lam) > 1 and len(lam) == prod_dim:  # Rank-1 perturbation of identity
        if lam[1] - lam[prod_dim - 1] < tol**2:
            return True

    if schmidt_rank(state, dim=dims_list) <= 2:
        return True  # OSR <= 2 and PPT implies separable

    # Section 6: Fallback to Symmetric Extension
    if has_symmetric_extension(state, level=level, dim=dims_list, tol=tol):
        return True

    return False  # Default to entangled if no test proved separability
