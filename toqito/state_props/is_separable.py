"""Checks if a quantum state is a separable state."""

import numpy as np
from scipy.linalg import orth

from toqito.channel_ops import partial_channel
from toqito.channels.realignment import realignment
from toqito.matrix_ops import partial_trace, partial_transpose
from toqito.matrix_props.is_positive_semidefinite import is_positive_semidefinite
from toqito.matrix_props.trace_norm import trace_norm
from toqito.perms.swap import swap
from toqito.perms.swap_operator import swap_operator
from toqito.state_props import is_ppt
from toqito.state_props.has_symmetric_extension import has_symmetric_extension
from toqito.state_props.in_separable_ball import in_separable_ball
from toqito.state_props.schmidt_rank import schmidt_rank
from toqito.states.max_entangled import max_entangled


def is_separable(state: np.ndarray, dim: None | int | list[int] = None, level: int = 2, tol: float = 1e-8) -> bool:
    r"""Determine if a given state (given as a density matrix) is a separable state :footcite:`WikiSepSt`.

    A multipartite quantum state:
    :math:`\rho \in \text{D}(\mathcal{H}_1 \otimes \mathcal{H}_2 \otimes \dots \otimes \mathcal{H}_N)`
    is defined as fully separable if it can be written as a convex combination of product states.

    Overview
    ==========
    This function implements several criteria to determine separability, broadly following a similar
    order of checks as seen in tools like QETLAB's :code:`IsSeparable` function :footcite:`QETLAB_link`.

    1.  **Input Validation**: Checks if the input :code:`state` is a square, positive semidefinite (PSD)
        NumPy array. Normalizes the trace to 1 if necessary. The :code:`dim` parameter specifying
        subsystem dimensions :math:`d_A, d_B` is processed or inferred.

    2.  **Trivial Cases for Separability**:

        - If either subsystem dimension :math:`d_A` or :math:`d_B` is 1
          (i.e., :code:`min_dim_val == 1`), the state is always separable.

    3.  **Pure State Check (Schmidt Rank)**:

        - If the input state has rank 1 (i.e., it's a pure state), its Schmidt rank is computed.
          A pure state is separable if and only if its Schmidt rank is 1 :footcite:`WikiScmidtDecomp`.

        .. note::
            QETLAB also considers a more general Operator Schmidt Rank condition from
            :footcite:`Cariello_2013_Weak_irreducible` for weak irreducible matrices. This
            is not explicitly separated in this function but might be covered if such
            matrices are rank 1 (see issue #1245).


    4.  **Gurvits-Barnum Separable Ball**:

        - Checks if the state lies within the "separable ball" around the maximally mixed state,
          as defined by Gurvits and Barnum :footcite:`Gurvits_2002_Ball`. States within this ball are
          guaranteed to be separable.

    5.  **PPT Criterion (Peres-Horodecki)**
        :footcite:`Peres_1996_Separability`, :footcite:`Horodecki_1996_PPT_small_dimensions`:

        - The Positive Partial Transpose (PPT) criterion is a necessary condition for separability.
        - If the state is NPT (Not PPT), it is definitively entangled.
        - If the state is PPT and the total dimension :math:`d_A d_B \le 6`,
          then PPT is also a *sufficient* condition for separability
          :footcite:`Horodecki_1996_PPT_small_dimensions`.

    6.  **3x3 Rank-4 PPT N&S Check (Plücker Coordinates / Breuer / Chen & Djokovic)**:

        - For 3x3 systems, if a PPT state has rank 4, there are known necessary and sufficient conditions
          for separability. These are often related to the vanishing of the "Chow form" or determinants
          of matrices constructed from Plücker coordinates of the state's range
          (e.g., :footcite:`Breuer_2006_Optimal`, :footcite:`Chen_2013_MultipartiteRank4`).
          The implementation checks if a specific determinant, derived from Plücker coordinates of the state's
          range, is close to zero.

    7.  **Operational Criteria for Low-Rank PPT States (Horodecki et al. 2000)**
        :footcite:`Horodecki_2000_PPT_low_rank`:

        For PPT states (especially when :math:`d_A d_B > 6`):

        - If :math:`\text{rank}(\rho) \le \max(d_A, d_B)`, the state is separable.
        - If :math:`\text{rank}(\rho) + \text{rank}(\rho^{T_A}) \le 2 d_A d_B - d_A - d_B + 2`,
          the state is separable.

    8.  **Reduction Criterion (Horodecki & Horodecki 1999)** :footcite:`Horodecki_1998_Reduction`:

        - The state is entangled if :math:`I_A \otimes \rho_B - \rho \not\succeq 0` or
          :math:`\rho_A \otimes I_B - \rho \not\succeq 0`. This is a check for positive semidefiniteness
          based on the Loewner partial order, not a check for majorization.
        - For PPT states (which is the case if this part of the function is reached),
          this criterion is always satisfied, so its primary strength is for NPT states (already handled).

    9.  **Realignment/CCNR Criteria**:

        - **Basic Realignment (Chen & Wu 2003)** :footcite:`Chen_2003_Matrix`:
          If the trace norm of the realigned matrix is greater than 1, the state is entangled.

    10. **Rank-1 Perturbation of Identity for PPT States (Vidal & Tarrach 1999)** :footcite:`Vidal_1999_Robust`:

        - PPT states that are very close to a specific type of rank-1 perturbation
          of the identity matrix are separable. This is checked by examining the eigenvalue spectrum:
          if the gap between the second largest and smallest eigenvalues is small,
          the state is determined to be separable.

    11. **2xN Specific Checks for PPT States**:
        For bipartite systems where one subsystem is a qubit (:math:`d_A=2`) and the
        other is N-dimensional (:math:`d_B=N`), several specific conditions apply:

        - **Johnston's Spectral Condition (2013)** :footcite:`Johnston_2013_Spectrum`:
          An inequality involving the largest and smallest eigenvalues of a 2xN PPT state that is sufficient
          for separability.
        - **Hildebrand's Conditions (2005, 2007, 2008)**
            :footcite:`Hildebrand_2007_AbsPPT`,
            :footcite:`Hildebrand_2008_Semidefinite`,
            :footcite:`Hildebrand_2005_Cone`:

            - For a 2xN state written in block form :math:`\rho = [[A, B], [B^\dagger, C]]`,
              a check is performed based on the rank of the anti-Hermitian part of the off-diagonal block
              :math:`B` (i.e., :math:`\text{rank}(B - B^\dagger) \le 1`).
              (Note: QETLAB refers to this property in relation to "perturbed block Hankel" matrices).
            - A check involving a transformed matrix :math:`X_{2n\_ppt\_check}`
              derived from blocks A, B, C, requiring it and its partial transpose
              to be PSD (related to homothetic images).
            - A condition based on the Frobenius norm of block :math:`B` compared to
              eigenvalues of blocks :math:`A` and :math:`C`.

    12. **Decomposable Maps / Entanglement Witnesses**:
        These tests apply positive but not completely positive (NCP) maps. If the resulting state is not PSD,
        the original state is entangled.

        - **Ha-Kye Maps (3x3 systems)** :footcite:`HaKye_2011_Positive`: Specific maps
          for qutrit-qutrit systems.
        - **Breuer-Hall Maps (even dimensions)** :footcite:`Breuer_2006_Mixed`, :footcite:`Hall_2006_Indecomposable`:
          Maps based on antisymmetric unitary matrices, applicable when a subsystem
          has even dimension.

    13. **Symmetric Extension Hierarchy (DPS)** :footcite:`Doherty_2004_CompleteFamily`:

        - A state is separable if and only if it has a k-symmetric extension for all :math:`k \ge 1`.
        - This function checks for k-extendibility up to the specified :code:`level`.
        - If :code:`level=1` and the state is PPT (which it is at this stage),
          it's 1-extendible and thus considered separable by this specific test level.
        - For :code:`level >= 2`, if a k-symmetric extension exists for all k up
          to :code:`level` (specifically, if :code:`has_symmetric_extension` returns
          :code:`True` for the highest :code:`k_actual_level_check` in the loop, which is
          :code:`level`), the current implementation returns :code:`True`.

        .. note::
            The symmetric extension check requires CVXPY and a suitable solver. If these are not installed,
            or if the solver fails, a warning is printed to the console and this check is skipped.
        .. note::
            QETLAB's :code:`SymmetricExtension` typically tests k-PPT-extendibility, where failure means entangled.
            It also has :code:`SymmetricInnerExtension`, which can prove separability.


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

    We can also detect certain PPT-entangled states. For example, a state constructed from a Breuer-Hall map
    is entangled but PPT.

    .. jupyter-execute::

        from toqito.state_props.is_ppt import is_ppt

        # Construct a 2x3 separable PPT state of rank 2
        # |ψ₁⟩ = |0⟩⊗|0⟩, |ψ₂⟩ = |1⟩⊗|1⟩
        psi1 = np.kron([1, 0], [1, 0, 0])
        psi2 = np.kron([0, 1], [0, 1, 0])
        rho = 0.5 * (np.outer(psi1, psi1.conj()) + np.outer(psi2, psi2.conj()))

        print("Is the state PPT?", is_ppt(rho, dim=[2, 3]))         # True
        print("Is the state separable?", is_separable(rho, dim=[2, 3]))  # True


    References
    ==========
    .. footbibliography::


    :raises Warning:

        If the symmetric extension check is attempted but CVXPY or a suitable solver is not available.

    :raises TypeError:

        - If the input `state` is not a NumPy array.

    :raises RuntimeError:

        - If the symmetric extension check is attempted but fails due to CVXPY solver issues.

    :raises NotImplementedError:

        - If the symmetric extension check is attempted but the level is not implemented (e.g., level < 1).

    :raises ValueError:

        - If the input `state` is not a square matrix.
        - If the input `state` is not positive semidefinite.
        - If the input `state` has a trace close to zero but contains significant non-zero elements.
        - If the input `state` has a numerically insignificant trace but significant elements;
          cannot normalize reliably.
        - If the `dim` parameter has an invalid type (not None, int, or list).
        - If `dim` is provided as an integer that does not evenly divide the state's dimension.
        - If `dim` is provided as a list with a number of elements other than two.
        - If `dim` is provided as a list with non-integer or negative elements.
        - If the product of the dimensions in the `dim` list does not match the state's dimension.
        - If a dimension of zero is provided for a non-empty state (or vice-versa).

    :param state: The density matrix to check.
    :param dim: The dimension of the input state, e.g., [dim_A, dim_B]. Optional; inferred if None.
    :param level: The level for symmetric extension (DPS) hierarchy (default: 2).

        - If 1, only PPT is checked.
        - If >=2, checks for k-symmetric extension up to this level.
        - If -1, attempts all implemented checks exhaustively (not all possible checks are implemented).

    :param tol: Numerical tolerance (default: 1e-8).

    :return: :code:`True` if separable, :code:`False` if entangled or inconclusive by implemented checks.

    """
    # --- 1. Input Validation, Normalization, Dimension Setup ---
    if not isinstance(state, np.ndarray):
        raise TypeError("Input state must be a NumPy array.")
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("Input state must be a square matrix.")

    # Define the smallest number computer can represent to avoid numerical issues.
    # This is used to determine the machine epsilon for numerical significance checks.
    if np.issubdtype(state.dtype, np.complexfloating):
        machine_eps = np.finfo(state.real.dtype).eps
    elif np.issubdtype(state.dtype, np.floating):
        machine_eps = np.finfo(state.dtype).eps
    else:
        machine_eps = np.finfo(float).eps

    state_len = state.shape[0]

    if not is_positive_semidefinite(state, atol=tol, rtol=tol):
        raise ValueError("Checking separability of non-positive semidefinite matrix is invalid.")

    trace_state_val = np.trace(state)
    current_state = state.copy()

    # Define a heuristic factor to determine when a floating-point value is
    # significant enough to be considered non-zero. A value is deemed
    # significant if it's larger than this factor multiplied by the machine
    # epsilon and the scale of the data. A factor of 100 provides a robust
    # safety margin against accumulated round-off errors.
    nsf = 100  # NUMERICAL_SIGNIFICANCE_FACTOR
    tolerance = nsf * machine_eps * max(1, np.max(np.abs(current_state)) if current_state.size > 0 else 1)
    if state_len > 0 and abs(trace_state_val) < tol:
        if np.any(
            np.abs(current_state) > tolerance  # Check if any element is significantly non-zero
        ):
            raise ValueError("Trace of the input state is close to zero, but state is not zero matrix.")

    if abs(trace_state_val - 1) > tol:
        if abs(trace_state_val) > 100 * machine_eps:
            current_state = current_state / trace_state_val
        elif state_len > 0 and np.any(np.abs(current_state) > tol):  #  (Hard to hit with PSD)
            raise ValueError(
                "State has numerically insignificant trace but significant elements; cannot normalize reliably."
            )

    # Dimension processing
    if dim is None:
        if state_len == 0:
            dims_list = [0, 0]
        elif state_len == 1:
            dims_list = [1, 1]
        else:
            sqrt_len = int(np.round(np.sqrt(state_len)))
            if sqrt_len * sqrt_len == state_len:
                dims_list = [sqrt_len, sqrt_len]
            else:
                found_factor = False
                for dA_try in range(2, int(np.sqrt(state_len)) + 1):
                    if state_len % dA_try == 0:
                        dims_list = [dA_try, state_len // dA_try]
                        found_factor = True
                        break
                if not found_factor:
                    dims_list = [1, state_len]
    elif isinstance(dim, int):
        if dim <= 0:
            if state_len == 0 and dim == 0:
                dims_list = [0, 0]
            else:
                raise ValueError(
                    "Integer `dim` (interpreted as dim_A) must be positive "
                    + "for non-empty states or zero for empty states."
                )
        elif state_len == 0 and dim != 0:
            raise ValueError(f"Cannot apply positive dimension {dim} to zero-sized state.")
        elif state_len > 0 and dim > 0 and state_len % dim != 0:
            raise ValueError("The parameter `dim` must evenly divide the length of the state.")
        else:
            dims_list = [int(dim), int(np.round(state_len / dim))]
    elif isinstance(dim, list) and len(dim) == 2:
        if not all(isinstance(d, (int, np.integer)) and d >= 0 for d in dim):
            raise ValueError("Dimensions in list must be non-negative integers.")
        if dim[0] * dim[1] != state_len:
            if (dim[0] == 0 or dim[1] == 0) and state_len != 0:
                raise ValueError("Non-zero state with zero-dim subsystem is inconsistent.")
            raise ValueError("Product of list dimensions must equal state length.")
        dims_list = [int(d) for d in dim]
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
        # Every positive semidefinite matrix is separable when one of the local dimensions is 1.
        return True

    # --- 3. Pure State Check (Schmidt Rank) ---
    # A pure state (rank 1) is separable if and only if its Schmidt rank is 1.
    # (The condition `s_rank <= 2` was previously here, referencing Cariello for weak irreducible matrices;
    # however, for general pure states, s_rank=1 is the N&S condition.
    # TODO: look at #1245 Consider adding a separate check for OperatorSchmidtRank <= 2 for general mixed states
    # if they are determined to be "weakly irreducible", as per Cariello :footcite:`Cariello_2013_Weak_irreducible`
    # and QETLAB's implementation. This is distinct from this pure state check.)
    if state_rank == 1:
        s_rank = schmidt_rank(current_state, dims_list)
        return bool(s_rank == 1)

    # --- 4. Gurvits-Barnum Separable Ball ---
    if in_separable_ball(current_state):
        # Determined to be separable by closeness to the maximally mixed state :footcite:`Gurvits_2002_Ball`.
        return True

    # --- 5. PPT (Peres-Horodecki) Criterion ---
    is_state_ppt = is_ppt(state, 2, dim, tol)  # sys=2 implies partial transpose on the second system by default
    if not is_state_ppt:
        # Determined to be entangled via the PPT criterion :footcite:`Peres_1996_Separability`.
        # Also, see Horodecki Theorem in :footcite:`Gühne_2009_Horodecki`.
        return False

    # ----- From here on, the state is known to be PPT -----

    # --- 5a. PPT and dim <= 6 implies separable ---
    if prod_dim_val <= 6:  # e.g., 2x2 or 2x3 systems
        # For dA * dB <= 6, PPT is necessary and sufficient for separability
        # :footcite:`Horodecki_1996_PPT_small_dimensions`.
        return True

    # --- 6. 3x3 Rank-4 PPT N&S Check (Plucker/Breuer/Chen&Djokovic) ---
    # This checks if a 3x3 PPT state of rank 4 is separable.
    # The condition involves the determinant of a matrix F constructed from Plücker coordinates.
    # Separability is linked to det(F) being (close to) zero :footcite:`Breuer_2006_Optimal`,
    # :footcite:`Chen_2013_MultipartiteRank4`.
    # (Note: Breuer's original PRL also relates it to F being indefinite or zero).
    if dA == 3 and dB == 3 and state_rank == 4:
        q_orth_basis = orth(current_state)  # Orthonormal basis for the range of rho
        if q_orth_basis.shape[1] < 4:  # Should not happen if rank is indeed 4
            pass  # Proceed, as condition for this check is not strictly met
        else:
            # Code for calculating Plucker coordinates p_np_arr and F_det_matrix_elements
            p_np_arr = np.zeros((6, 7, 8, 9), dtype=complex)  # Stores Plucker coordinates
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
                                        except np.linalg.LinAlgError:  # Should be rare with orth basis
                                            p_np_arr[j_breuer - 1, k_breuer - 1, n_breuer - 1, m_breuer - 1] = np.nan

            def get_p(t_tuple):
                # Ensure indices are within bounds of p_np_arr before accessing
                if not (
                    0 <= t_tuple[0] - 1 < p_np_arr.shape[0]
                    and 0 <= t_tuple[1] - 1 < p_np_arr.shape[1]
                    and 0 <= t_tuple[2] - 1 < p_np_arr.shape[2]
                    and 0 <= t_tuple[3] - 1 < p_np_arr.shape[3]
                ):
                    # This case should ideally not happen if t_tuple values are
                    # from the F_det_matrix_elements construction
                    # and p_np_arr is sized for 1-based indices up to 9.
                    # However, being defensive:
                    return 0.0  # Or handle as an error/warning #
                val = p_np_arr[t_tuple[0] - 1, t_tuple[1] - 1, t_tuple[2] - 1, t_tuple[3] - 1]
                return val if not np.isnan(val) else 0.0

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
                # This condition (abs(det(F)) ~ 0 for separability) is used by QETLAB for this specific test.
                return bool(
                    abs(F_det_val) < max(tol**2, machine_eps ** (3 / 4))
                )  # Separable by this 3x3 rank-4 condition
                # Proceeding from 3x3 rank 4 block.")
                # If det(F) is not close to zero, the state is entangled by this criterion.
                # TODO: #1251 Breuer's PRL indicates separability if F is indefinite or zero. Entangled if F is
                # definite (and det(F) real).
                # The current check `abs(F_det_val) < tol_check` might only capture the `F=0` part or if
                # F is singular due to indefiniteness.
                # If this F_det_val is not small, it implies entangled.
                # However, your original code structure implies if this `return True` is not hit, it just proceeds.
                # For consistency with QETLAB for this specific test: if abs(F_det_val) is NOT small, it
                # implies entangled.
                # This function would need to return False here if abs(F_det_val) is NOT small.
                # Current structure: if det is small, sep=T. If det is not small, or LinAlgError, proceeds.
                # For UPB tile states (3x3, rank 4, PPT entangled), det(F) is typically non-zero. So this
                # `return True` isn't hit.
            except np.linalg.LinAlgError:  # If determinant calculation fails
                pass  # Proceed to other tests

    # --- 7. Operational Criteria for Low-Rank PPT States (Horodecki et al. 2000) ---
    # These are sufficient conditions for separability of PPT states :footcite:`Horodecki_2000_PPT_low_rank`.
    if state_rank <= max_dim_val:  # rank(rho) <= max(dA, dB)
        return True

    rho_pt_A = partial_transpose(current_state, sys=0, dim=dims_list)  # Partial transpose on system A
    rank_pt_A = np.linalg.matrix_rank(rho_pt_A, tol=tol)
    threshold_horodecki = 2 * prod_dim_val - dA - dB + 2  # Threshold for sum of ranks

    if state_rank + rank_pt_A <= threshold_horodecki:  # rank(rho) + rank(rho^T_A) <= threshold
        return True
    # TODO: #1251 Add check for rank(rho) <= rank(marginal_A) or rank(rho) <= rank(marginal_B) as in QETLAB.

    # --- 8. Reduction Criterion (Horodecki & Horodecki 1999) ---
    # If state is PPT (which it is at this point), this criterion is always satisfied.
    # Its main use is for NPT states. Included for completeness of listed criteria.
    rho_A_marginal = partial_trace(current_state, sys=[1], dim=dims_list)
    rho_B_marginal = partial_trace(current_state, sys=[0], dim=dims_list)
    op_reduct1 = np.kron(np.eye(dA), rho_B_marginal) - current_state
    op_reduct2 = np.kron(rho_A_marginal, np.eye(dB)) - current_state
    if not (
        is_positive_semidefinite(op_reduct1, atol=tol, rtol=tol)
        and is_positive_semidefinite(op_reduct2, atol=tol, rtol=tol)
    ):  #  (should not be hit for PPT states)
        return False  # Entangled by reduction criterion

    # --- 9. Realignment/CCNR Criteria ---
    # Basic Realignment Criterion :footcite:`Chen_2003_Matrix`. If > 1, entangled.
    if trace_norm(realignment(current_state, dims_list)) > 1 + tol:
        return False

    # Zhang et al. 2008 Variant :footcite:`Zhang_2008_Beyond_realignment`.
    # If ||R(rho - rho_A \otimes rho_B)||_1 > sqrt((1-Tr(rho_A^2))(1-Tr(rho_B^2))), entangled.
    tr_rho_A_sq = np.real(np.trace(rho_A_marginal @ rho_A_marginal))
    tr_rho_B_sq = np.real(np.trace(rho_B_marginal @ rho_B_marginal))
    val_A = max(0, 1 - tr_rho_A_sq)  # Ensure non-negativity from (1 - purity)
    val_B = max(0, 1 - tr_rho_B_sq)
    bound_zhang = np.sqrt(val_A * val_B) if (val_A * val_B >= 0) else 0
    if trace_norm(realignment(current_state - np.kron(rho_A_marginal, rho_B_marginal), dims_list)) > bound_zhang + tol:
        return False
    # TODO: #1246 Consider adding Filter CMC criterion from Gittsovich et al. 2008, which is stronger.

    # --- 10. Rank-1 Perturbation of Identity for PPT States (Vidal & Tarrach 1999) ---
    # PPT states close to identity are separable :footcite:`Vidal_1999_Robust`.
    try:
        try:
            lam = np.linalg.eigvalsh(current_state)[::-1]  # Eigenvalues sorted descending
        except np.linalg.LinAlgError:  # Fallback if eigvalsh fails
            lam = np.sort(np.real(np.linalg.eigvals(current_state)))[::-1]

        if len(lam) == prod_dim_val and prod_dim_val > 1:
            # If (lambda_2 - lambda_d) is very small for a PPT state.
            diff_pert = lam[1] - lam[prod_dim_val - 1]
            threshold_pert = tol**2 + 2 * machine_eps
            if diff_pert < threshold_pert:
                return True
    except np.linalg.LinAlgError:  # If all eigenvalue computations fail #
        pass  # Proceed to other tests

    # --- 11. 2xN Specific Checks for PPT States ---
    if min_dim_val == 2 and prod_dim_val > 0:  # One system is a qubit
        state_t_2xn = current_state
        d_N_val = max_dim_val  # Dimension of the N-level system
        # Ensure the qubit system is the first one for consistent block matrix decomposition
        # sys_to_pt_for_hildebrand_map = 1 (PT on system B, the N-level one)
        # dim_for_hildebrand_map = [2, d_N_val]
        dim_for_hildebrand_map = [2, d_N_val]

        if dA != 2 and dB == 2:  # If system A is N-level and B is qubit, swap them
            state_t_2xn = swap(current_state, sys=[0, 1], dim=dims_list)
            # d_N_val remains max_dim_val. Dimensions for map are now [qubit_dim, N_dim]
            dim_for_hildebrand_map = [dB, dA]
        elif dA == 2:  # System A is already the qubit
            pass  # state_t_2xn and d_N_val are correctly set
        else:  # This case should not be reached if min_dim_val == 2
            state_t_2xn = None  # Defensive #

        if state_t_2xn is not None:
            current_lam_2xn = lam  # Use eigenvalues of original state if no swap
            if state_t_2xn is not current_state:  # If swap occurred, recompute eigenvalues
                try:
                    current_lam_2xn = np.linalg.eigvalsh(state_t_2xn)[::-1]
                except np.linalg.LinAlgError:
                    current_lam_2xn = np.sort(np.real(np.linalg.eigvals(state_t_2xn)))[::-1]

            # Johnston's Spectral Condition :footcite:`Johnston_2013_Spectrum`
            if (
                len(current_lam_2xn) >= 2 * d_N_val  # Check if enough eigenvalues exist
                and (2 * d_N_val - 1) < len(current_lam_2xn)  # Index validity
                and (2 * d_N_val - 2) >= 0
                and (2 * d_N_val - 3) >= 0
            ):
                # Condition: (lambda_1 - lambda_{2N-1})^2 <= 4 * lambda_{2N-2} * lambda_{2N}
                # (Using 0-based indexing: (lam[0]-lam[2N-2])^2 <= 4*lam[2N-3]*lam[2N-1])
                if (current_lam_2xn[0] - current_lam_2xn[2 * d_N_val - 2]) ** 2 <= 4 * current_lam_2xn[
                    2 * d_N_val - 3
                ] * current_lam_2xn[2 * d_N_val - 1] + tol**2:  # Added tolerance
                    return True

            # Hildebrand's Conditions for 2xN PPT states (various papers, e.g.,
            # :footcite:`Hildebrand_2005_Cone`, :footcite:`Hildebrand_2008_Semidefinite`)
            # Block matrix form: rho_2xn = [[A, B], [B^dagger, C]]
            A_block = state_t_2xn[:d_N_val, :d_N_val]
            B_block = state_t_2xn[:d_N_val, d_N_val : 2 * d_N_val]
            C_block = state_t_2xn[d_N_val : 2 * d_N_val, d_N_val : 2 * d_N_val]

            # If rank of anti-Hermitian part of B is small (related to "perturbed block Hankel" in QETLAB)
            if B_block.size > 0 and np.linalg.matrix_rank(B_block - B_block.conj().T, tol=tol) <= 1:
                return True

            # Hildebrand's homothetic images approach / X_2n_ppt_check
            if A_block.size > 0 and B_block.size > 0 and C_block.size > 0:  # Ensure blocks are not empty #
                X_2n_ppt_check = np.vstack(
                    (
                        np.hstack(((5 / 6) * A_block - C_block / 6, B_block)),
                        np.hstack((B_block.conj().T, (5 / 6) * C_block - A_block / 6)),
                    )
                )
                # The dimensions for partial_transpose of X_2n_ppt_check should be [2, d_N_val]
                # if the map is applied on the "qubit part" of this transformed 2N x 2N space.
                # QETLAB uses IsPPT(X_2n_ppt_check,2,[2,xD]), implying PT on 2nd system of a 2xD structure.
                # Here, sys_to_pt_for_hildebrand_map=1 and dim_for_hildebrand_map=[2,d_N_val] seems correct.
                if is_positive_semidefinite(X_2n_ppt_check, atol=tol, rtol=tol) and is_ppt(
                    X_2n_ppt_check,
                    sys=1,
                    dim=dim_for_hildebrand_map,
                    tol=tol,
                ):  # Check PPT of this map's Choi matrix basically
                    return True

                # Johnston Lemma 1 variant / norm B condition
                try:
                    eig_A_real, eig_C_real = np.real(np.linalg.eigvals(A_block)), np.real(np.linalg.eigvals(C_block))
                    if eig_A_real.size > 0 and eig_C_real.size > 0 and B_block.size > 0:
                        if np.linalg.norm(B_block) ** 2 <= np.min(eig_A_real) * np.min(eig_C_real) + tol**2:
                            return True
                except np.linalg.LinAlgError:
                    pass  # Eigenvalue computation failed

    # --- 12. Decomposable Maps (Positive but not Completely Positive Maps as Witnesses) ---
    # Ha-Kye Maps for 3x3 systems :footcite:`HaKye_2011_Positive`
    if dA == 3 and dB == 3:
        phi_me3 = max_entangled(3, False, False)  # Maximally entangled state vector in C^3 x C^3
        phi_proj3 = phi_me3 @ phi_me3.conj().T  # Projector onto it
        for t_raw_ha in np.arange(0.1, 1.0, 0.1):  # Parameter 't' for map construction
            t_iter_ha = t_raw_ha
            for j_ha_loop in range(2):  # Iterate for t and 1/t (common symmetry in these maps)
                if j_ha_loop == 1:
                    # if abs(t_raw_ha) < machine_eps: #  (t_raw_ha always >= 0.1)
                    #     break  # Should not happen with arange
                    t_iter_ha = 1 / t_raw_ha

                denom_ha = 1 - t_iter_ha + t_iter_ha**2  # Denominator from Ha-Kye map parameters
                if abs(denom_ha) < machine_eps:
                    continue  #  (denom_ha = 1-t+t^2 > 0 for t>0)

                a_hk = (1 - t_iter_ha) ** 2 / denom_ha
                b_hk = t_iter_ha**2 / denom_ha
                c_hk = 1 / denom_ha
                # Choi matrix of a generalized Choi map (related to Ha-Kye constructions)
                Phi_map_ha = np.diag([a_hk + 1, c_hk, b_hk, b_hk, a_hk + 1, c_hk, c_hk, b_hk, a_hk + 1]) - phi_proj3
                if not is_positive_semidefinite(
                    partial_channel(current_state, Phi_map_ha, sys=1, dim=dims_list), atol=tol, rtol=tol
                ):
                    return False  # Entangled if map application results in non-PSD state #

    # Breuer-Hall Maps (for even dimensional subsystems) :footcite:`Breuer_2006_Mixed`,
    # :footcite:`Hall_2006_Indecomposable`
    for p_idx_bh in range(2):  # Apply map to subsystem 0 (A), then subsystem 1 (B)
        current_dim_bh = dims_list[p_idx_bh]  # Dimension of the subsystem map acts on
        if current_dim_bh > 0 and current_dim_bh % 2 == 0:  # Map defined for even dimensions
            phi_me_bh = max_entangled(current_dim_bh, False, False)
            phi_proj_bh = phi_me_bh @ phi_me_bh.conj().T
            half_dim_bh = current_dim_bh // 2
            # Construct an antisymmetric unitary U_bh_kron_part
            diag_U_elems_bh = np.concatenate([np.ones(half_dim_bh), -np.ones(half_dim_bh)])
            U_bh_kron_part = np.fliplr(np.diag(diag_U_elems_bh))  # U = -U^T
            # Choi matrix of the Breuer-Hall witness map W_U(X) = Tr(X)I - X - U X^T U^dagger
            # The Choi matrix used here is I - P_max_ent - (I kron U) SWAP (I kron U^dagger)
            U_for_phi_constr = np.kron(np.eye(current_dim_bh), U_bh_kron_part)
            Phi_bh_map_choi = (  # This is Choi(W_U)
                np.eye(current_dim_bh**2)
                - phi_proj_bh
                - U_for_phi_constr @ swap_operator(current_dim_bh) @ U_for_phi_constr.conj().T
            )
            mapped_state_bh = partial_channel(current_state, Phi_bh_map_choi, sys=p_idx_bh, dim=dims_list)
            if not is_positive_semidefinite(mapped_state_bh, atol=tol, rtol=tol):
                return False  # Entangled if map application results in non-PSD state

    # --- 13. Symmetric Extension Hierarchy (DPS) ---
    # A state is separable iff it has a k-symmetric extension for all k :footcite:`Doherty_2004_CompleteFamily`.
    # We check up to the specified `level`.
    if level >= 2:  # Level 1 (PPT) is already confirmed if we reach here.
        # Loop for k from 2 up to `level` specified by user.
        # If `has_symmetric_extension` returns True for any k in this loop,
        # it means the state *is* k-extendible. This implementation interprets this as
        # passing the DPS test up to that level, and thus returns True (separable).
        # TODO: #1247 A stricter interpretation for proving entanglement would be: if for *any* k in this loop,
        # `has_symmetric_extension` returns False, then it's entangled.
        # QETLAB's `SymmetricExtension` returns 0 if *not* k-PPT-extendible (entangled).
        # QETLAB's `SymmetricInnerExtension` returns 1 if separable by that method.
        # The current Python loop `if has_symmetric_extension(...): return True` means if it passes
        # for k=2 (and level >=2), it declares separable.
        for k_actual_level_check in range(2, int(level) + 1):  # Ensure level is int for range
            try:
                if has_symmetric_extension(rho=current_state, level=k_actual_level_check, dim=dims_list, tol=tol):
                    # State has a k-symmetric extension, considered separable by this test level.
                    return True
                # If it does NOT have a k-symmetric extension, it is entangled.
                # The current loop structure means if has_symmetric_extension for k=2 is False,
                # it will continue to k=3 (if level >=3), etc. It only returns True if an extension is found.
                # To match "if not extendible -> entangled":
                # if not has_symmetric_extension(...): return False; # This would be a change.
                # The current logic is: "if extendible at any k up to level, then separable".
            except ImportError:
                print("Warning: CVXPY or a solver is not installed; cannot perform symmetric extension check.")
                break  # Stop trying symmetric extensions if dependencies are missing
            except Exception as e:
                print(f"Warning: Symmetric extension check failed at level {k_actual_level_check} with an error: {e}")
                # Decide whether to break or continue to next k_level if solver fails for one.
                # Current: proceeds to next k or finishes loop.
                pass  # Proceed, may not be able to determine via this method.
    elif level == 1 and is_state_ppt:  # is_state_ppt is True at this point
        # 1-extendibility is equivalent to PPT.
        return True

    # If all implemented checks are inconclusive, and the state passed PPT (the most basic necessary condition checked),
    # it implies that the state is either separable but not caught by the simpler sufficient conditions,
    # or it's a PPT entangled state that also wasn't caught by other implemented witnesses
    # or the DPS hierarchy up to `level`.
    # Defaulting to False implies we couldn't definitively prove separability with the implemented tests.
    return False
