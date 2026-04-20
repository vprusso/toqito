"""Checks if a quantum state is a separable state."""

import numpy as np
from scipy.linalg import orth

from toqito.channel_ops import partial_channel
from toqito.channels.realignment import realignment
from toqito.matrix_ops import partial_trace
from toqito.matrix_props.is_positive_semidefinite import is_positive_semidefinite
from toqito.matrix_props.trace_norm import trace_norm
from toqito.perms.swap import swap
from toqito.perms.swap_operator import swap_operator
from toqito.state_props import is_ppt
from toqito.state_props.has_symmetric_extension import has_symmetric_extension
from toqito.state_props.has_symmetric_inner_extension import has_symmetric_inner_extension
from toqito.state_props.in_separable_ball import in_separable_ball
from toqito.state_props.schmidt_rank import schmidt_rank
from toqito.states.max_entangled import max_entangled
from toqito.states.tile import tile


def _choi_1975_choi_matrix() -> np.ndarray:
    r"""Return the Choi matrix of Choi's 1975 indecomposable positive map on :math:`M_3`.

    The map :math:`\Phi: M_3 \to M_3` is defined by

    .. math::

        \Phi(A)_{00} = A_{00} + A_{22}, \quad
        \Phi(A)_{11} = A_{00} + A_{11}, \quad
        \Phi(A)_{22} = A_{11} + A_{22}, \quad
        \Phi(A)_{ij} = -A_{ij} \text{ for } i \neq j.

    It is a standard example from [@choi1975] of a positive but not
    completely positive map — its Choi matrix has a negative eigenvalue — and
    is used in :func:`is_separable` as an entanglement witness.
    """
    # Diagonal blocks carry Phi(E_ii) along the block diagonal.
    diag_action = [
        np.diag([1.0, 1.0, 0.0]),
        np.diag([0.0, 1.0, 1.0]),
        np.diag([1.0, 0.0, 1.0]),
    ]
    choi = np.zeros((9, 9), dtype=complex)
    for i in range(3):
        choi[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = diag_action[i]
    # Off-diagonal blocks carry Phi(E_ij) = -E_ij for i != j,
    # i.e. a single -1 at position (i, j) inside the (i, j) block.
    for i in range(3):
        for j in range(3):
            if i != j:
                choi[3 * i + i, 3 * j + j] = -1.0
    return choi


def _terhal_2000_tile_witness() -> np.ndarray:
    r"""Terhal 2000 indecomposable positive-map witness built on the 3x3 tile UPB.

    Implements Theorem 3 of [@terhal2000family]: given the five tile UPB
    product vectors :math:`\{|\alpha_i\rangle \otimes |\beta_i\rangle\}_{i=0}^4`
    spanning a 5-dimensional subspace of :math:`\mathbb{C}^3 \otimes
    \mathbb{C}^3`, the Hermitian operator

    .. math::

        H = \sum_{i=0}^{4} |\alpha_i\rangle\langle\alpha_i| \otimes
            |\beta_i\rangle\langle\beta_i|
            - d \varepsilon |\Psi\rangle\langle\Psi|

    is an indecomposable entanglement witness, where :math:`d = 3`,

    .. math::

        \varepsilon = \min_{|\phi_A\rangle, |\phi_B\rangle}
            \sum_{i=0}^{4} |\langle\phi_A|\alpha_i\rangle|^2
                           |\langle\phi_B|\beta_i\rangle|^2,

    and :math:`|\Psi\rangle` is a maximally entangled state with
    :math:`\langle\Psi|\rho_{\text{tile}}|\Psi\rangle > 0` (the ``standard''
    max-ent state :math:`(|00\rangle+|11\rangle+|22\rangle)/\sqrt{3}` fails
    this condition for the tile UPB because :math:`|\Psi\rangle` happens to
    lie in the UPB span; we use :math:`(|10\rangle+|21\rangle+|02\rangle)/\sqrt{3}`
    instead, which satisfies it).

    A state :math:`\rho` is detected as entangled iff
    :math:`\mathrm{tr}(H\rho) < 0`. Positivity on product states follows from
    :math:`|\langle\Psi|\phi_A\otimes\phi_B\rangle|^2 \le 1/d` for maximally
    entangled :math:`|\Psi\rangle` (Lemma 1 of the paper) combined with the
    definition of :math:`\varepsilon`.
    """
    # Factor each tile(i) vector into its A and B factors via SVD.
    alpha_vecs = []
    beta_vecs = []
    for i in range(5):
        psi = np.asarray(tile(i)).flatten()
        u_mat, s_vals, vh_mat = np.linalg.svd(psi.reshape(3, 3))
        # Each tile(i) is a genuine product state, so one singular value dominates.
        alpha = u_mat[:, 0]
        beta = np.conjugate(vh_mat[0])
        alpha_vecs.append(alpha / np.linalg.norm(alpha))
        beta_vecs.append(beta / np.linalg.norm(beta))

    h_sum = np.zeros((9, 9), dtype=complex)
    for a, b in zip(alpha_vecs, beta_vecs):
        h_sum += np.kron(np.outer(a, a.conj()), np.outer(b, b.conj()))

    # Alternating minimization for epsilon. For fixed phi_B the function is
    # <phi_A|M_A|phi_A> with M_A = sum_i |<phi_B|beta_i>|^2 |alpha_i><alpha_i|,
    # minimized by the eigenvector of M_A's smallest eigenvalue. Flip subsystems
    # and repeat. Several random restarts handle non-convexity.
    rng = np.random.default_rng(seed=20260415)
    best_eps = np.inf
    for _ in range(20):
        p_a = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        p_a /= np.linalg.norm(p_a)
        p_b = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        p_b /= np.linalg.norm(p_b)
        prev_val = np.inf
        for _ in range(100):
            m_a = sum(
                abs(p_b.conj() @ b) ** 2 * np.outer(a, a.conj())
                for a, b in zip(alpha_vecs, beta_vecs)
            )
            _, vecs = np.linalg.eigh((m_a + m_a.conj().T) / 2)
            p_a = vecs[:, 0]
            m_b = sum(
                abs(p_a.conj() @ a) ** 2 * np.outer(b, b.conj())
                for a, b in zip(alpha_vecs, beta_vecs)
            )
            _, vecs = np.linalg.eigh((m_b + m_b.conj().T) / 2)
            p_b = vecs[:, 0]
            val = sum(
                abs(p_a.conj() @ a) ** 2 * abs(p_b.conj() @ b) ** 2
                for a, b in zip(alpha_vecs, beta_vecs)
            )
            if abs(val - prev_val) < 1e-14:
                break
            prev_val = val
        best_eps = min(best_eps, float(np.real(val)))
    epsilon = best_eps

    # |Psi> = (|10> + |21> + |02>)/sqrt(3); indices 3, 7, 2 in the flat basis.
    psi_vec = np.zeros(9, dtype=complex)
    psi_vec[3] = 1.0 / np.sqrt(3)
    psi_vec[7] = 1.0 / np.sqrt(3)
    psi_vec[2] = 1.0 / np.sqrt(3)
    psi_proj = np.outer(psi_vec, psi_vec.conj())

    return h_sum - 3 * epsilon * psi_proj


# Cached module-level witness matrix: construction is deterministic (the ε
# optimization uses a fixed seed) and only depends on constants, so we pay
# for the alternating minimization exactly once per process.
_TERHAL_2000_TILE_WITNESS_3X3: np.ndarray = _terhal_2000_tile_witness()


def _range_projector_product_overlap_3x3_rank4(
    state: np.ndarray,
    tol: float,
    n_restarts: int = 64,
    max_iter: int = 100,
) -> float | None:
    r"""Estimate the best product overlap with the range projector of a 3x3 rank-4 state.

    For the projector :math:`P` onto :math:`\operatorname{range}(\rho)`, compute

    .. math::

        \max_{\|a\|=\|b\|=1} \langle a \otimes b | P | a \otimes b \rangle

    by alternating maximization with deterministic random restarts.

    When the optimum is numerically 1, the range contains a product vector.
    Chen and Djokovic show that for 3x3 PPT states of rank 4 this is
    equivalent to separability.
    """
    range_basis = orth(state)
    if range_basis.shape[1] < 4:
        return None

    projector = range_basis @ range_basis.conj().T
    tensor = projector.reshape(3, 3, 3, 3)

    rng = np.random.default_rng(seed=20260420)
    best_overlap = 0.0
    for _ in range(n_restarts):
        vec_a = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        vec_a /= np.linalg.norm(vec_a)
        vec_b = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        vec_b /= np.linalg.norm(vec_b)

        prev_overlap = -np.inf
        overlap = 0.0
        for _ in range(max_iter):
            mat_a = np.einsum("ibjb,b,j->ij", tensor, vec_b.conj(), vec_b)
            _, eigvecs_a = np.linalg.eigh((mat_a + mat_a.conj().T) / 2)
            vec_a = eigvecs_a[:, -1]

            mat_b = np.einsum("iajb,a,i->jb", tensor, vec_a.conj(), vec_a)
            _, eigvecs_b = np.linalg.eigh((mat_b + mat_b.conj().T) / 2)
            vec_b = eigvecs_b[:, -1]

            prod_vec = np.kron(vec_a, vec_b)
            overlap = float(np.real(np.vdot(prod_vec, projector @ prod_vec)))
            if abs(overlap - prev_overlap) < tol:
                break
            prev_overlap = overlap

        best_overlap = max(best_overlap, overlap)
        if 1.0 - best_overlap < 10 * tol:
            return best_overlap

    return best_overlap


def _hermitian_inverse_sqrt(herm: np.ndarray, eig_floor: float) -> np.ndarray | None:
    """Return the inverse square root of a Hermitian PSD matrix, or None if rank-deficient.

    `eig_floor` is the smallest eigenvalue we're willing to invert; below that we
    consider the matrix effectively rank-deficient and return None so the caller
    can skip the filtering step.
    """
    eigvals, eigvecs = np.linalg.eigh((herm + herm.conj().T) / 2)
    if np.min(eigvals) < eig_floor:
        return None
    return eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.conj().T


def _filter_normal_form(
    rho: np.ndarray,
    dims: list[int],
    tol: float,
    max_iter: int = 50,
) -> np.ndarray | None:
    r"""Bring rho to its filter normal form via local SLOCC filtering.

    Implements the iterative algorithm of Verstraete, Dehaene, and De Moor
    (PRA 64, 010101 (2001)) used in Gittsovich et al. [@gittsovich2008unifying]
    Section IV.D: alternate applying :math:`T_A = (d_A \rho_A)^{-1/2}` on
    subsystem A and :math:`T_B = (d_B \rho_B)^{-1/2}` on subsystem B until both
    marginals become proportional to the identity. In normal form
    :math:`\tilde\rho_A = I/d_A` and :math:`\tilde\rho_B = I/d_B`, so the
    reduced states carry no separability information and the correlations
    live entirely in the off-diagonal block of the covariance matrix.

    Each filter step preserves the trace exactly:

    .. math::

        \mathrm{tr}\big[(T_A \otimes I) \rho (T_A^\dagger \otimes I)\big]
        = \mathrm{tr}\big[(d_A \rho_A)^{-1} \rho_A\big] = 1/d_A \cdot d_A = 1,

    so no renormalization is needed.

    Returns the filter normal form, or `None` if the iteration encounters a
    rank-deficient marginal (filtering requires invertible marginals) or fails
    to converge within `max_iter` passes.
    """
    dA, dB = dims[0], dims[1]
    id_a = np.eye(dA, dtype=complex)
    id_b = np.eye(dB, dtype=complex)
    target_a = id_a / dA
    target_b = id_b / dB

    rho_tilde = rho.astype(complex, copy=True)
    eig_floor = tol

    for _ in range(max_iter):
        rho_a = partial_trace(rho_tilde, sys=[1], dim=dims)
        rho_b = partial_trace(rho_tilde, sys=[0], dim=dims)

        err = max(
            np.linalg.norm(rho_a - target_a, ord="fro"),
            np.linalg.norm(rho_b - target_b, ord="fro"),
        )
        if err < tol:
            return rho_tilde

        t_a = _hermitian_inverse_sqrt(dA * rho_a, eig_floor)
        if t_a is None:
            return None
        rho_tilde = np.kron(t_a, id_b) @ rho_tilde @ np.kron(t_a.conj().T, id_b)

        rho_b_new = partial_trace(rho_tilde, sys=[0], dim=dims)
        t_b = _hermitian_inverse_sqrt(dB * rho_b_new, eig_floor)
        if t_b is None:
            return None
        rho_tilde = np.kron(id_a, t_b) @ rho_tilde @ np.kron(id_a, t_b.conj().T)

    # Iteration budget exhausted. Accept whatever we have if the marginals
    # are sufficiently close to the targets; otherwise report failure.
    rho_a = partial_trace(rho_tilde, sys=[1], dim=dims)
    rho_b = partial_trace(rho_tilde, sys=[0], dim=dims)
    err = max(
        np.linalg.norm(rho_a - target_a, ord="fro"),
        np.linalg.norm(rho_b - target_b, ord="fro"),
    )
    if err < 10 * tol:
        return rho_tilde
    return None


def _filter_cmc_xi_sum(rho_normal: np.ndarray, dims: list[int]) -> float:
    r"""Compute :math:`\sum_i \xi_i` for a state already in filter normal form.

    The :math:`\xi_i` are the coefficients in the operator-basis expansion

    .. math::

        \tilde\rho = \frac{1}{d_A d_B} \left(I + \sum_k \xi_k G^A_k \otimes G^B_k\right),

    equal to :math:`d_A d_B` times the operator Schmidt coefficients of
    :math:`\tilde\rho - I/(d_A d_B)` (which in turn are the singular values
    of the realignment of that trace-zero operator). See Gittsovich et al.
    2008, Eq. (66).
    """
    dA, dB = dims[0], dims[1]
    sigma = rho_normal - np.eye(dA * dB, dtype=complex) / (dA * dB)
    r_sigma = realignment(sigma, dim=dims)
    singular_values = np.linalg.svd(r_sigma, compute_uv=False)
    return float(dA * dB * np.sum(np.real(singular_values)))


def _filter_cmc_bound(dA: int, dB: int) -> float:
    r"""Return the Filter CMC separability bound for a :math:`d_A \times d_B` system.

    From Gittsovich et al. 2008, Proposition IV.15, Eq. (72):

    .. math::

        \sum_k \xi_k \le \sqrt{d_A d_B (d_A - 1)(d_B - 1)}.

    This generalizes Proposition IV.13 (:math:`d^2 - d` for the symmetric
    :math:`d_A = d_B = d` case) and, in practice, is tighter than Eq. (71)
    across the dimensions of interest here.
    """
    return float(np.sqrt(dA * dB * (dA - 1) * (dB - 1)))


def _iterative_product_state_subtraction(
    state: np.ndarray,
    dims: list[int],
    tol: float,
    max_outer_iter: int = 50,
    n_restarts: int = 5,
    max_inner_iter: int = 25,
    rng: np.random.Generator | None = None,
) -> bool:
    r"""Try to prove separability by iteratively subtracting product states.

    Loosely modelled on QETLAB's ``sk_iterate`` (with ``s = 1``) and the
    Gühne method [@guhne2009entanglement]. At each outer iteration, run
    alternating rank-1 maximization (with a handful of random restarts) to
    find a product state :math:`|\psi\rangle|\phi\rangle` with high overlap
    on the residual state, then subtract as much of
    :math:`|\psi\rangle\langle\psi| \otimes |\phi\rangle\langle\phi|` as
    positivity allows (via a geometric backoff search). Returns ``True`` if
    the residual eventually enters the Gurvits-Barnum separable ball or
    shrinks to numerical zero. Returns ``False`` if the loop stalls
    (no product state with non-trivial overlap, or subtraction driven to
    zero) or the iteration budget is exhausted — callers should treat
    ``False`` as inconclusive rather than as an entanglement verdict.

    The `rng` parameter accepts a `numpy.random.Generator` for reproducible
    restarts in tests. Callers that don't supply one get a fresh
    non-deterministic generator, so production calls genuinely randomize
    across restarts.

    This is a *one-sided* witness: the algorithm can prove separability
    constructively, but cannot disprove it.
    """
    dA, dB = dims[0], dims[1]
    residual = state.astype(complex, copy=True)
    if rng is None:
        rng = np.random.default_rng()

    for _outer in range(max_outer_iter):
        residual_trace = float(np.real(np.trace(residual)))
        if residual_trace < tol:
            return True  # Fully decomposed into product states.
        if in_separable_ball(residual / residual_trace):
            return True  # Residual fell into the Gurvits-Barnum ball.

        tensor = residual.reshape(dA, dB, dA, dB)
        best_overlap = 0.0
        best_prod_vec: np.ndarray | None = None
        # Both the inner-loop convergence threshold and the outer "stuck"
        # threshold scale with the current residual trace, so a small
        # residual (or a high-dimension diffuse residual) doesn't get
        # spuriously declared stuck when the best achievable overlap is
        # naturally below an absolute `tol`.
        scaled_tol = tol * max(1.0, residual_trace)
        for _ in range(n_restarts):
            psi = rng.standard_normal(dA) + 1j * rng.standard_normal(dA)
            psi /= np.linalg.norm(psi)
            phi = rng.standard_normal(dB) + 1j * rng.standard_normal(dB)
            phi /= np.linalg.norm(phi)

            prev_overlap = -np.inf
            overlap = 0.0
            for _inner in range(max_inner_iter):
                # Fix phi, maximize over psi: top eigenvector of
                # M_psi[i, k] = sum_{j, l} conj(phi_j) T[i, j, k, l] phi_l.
                m_psi = np.einsum("j,ijkl,l->ik", np.conj(phi), tensor, phi)
                m_psi = (m_psi + m_psi.conj().T) / 2.0
                _, vecs = np.linalg.eigh(m_psi)
                psi = vecs[:, -1]

                # Fix psi, maximize over phi: top eigenvector of
                # M_phi[j, l] = sum_{i, k} conj(psi_i) T[i, j, k, l] psi_k.
                m_phi = np.einsum("i,ijkl,k->jl", np.conj(psi), tensor, psi)
                m_phi = (m_phi + m_phi.conj().T) / 2.0
                _, vecs = np.linalg.eigh(m_phi)
                phi = vecs[:, -1]

                prod_vec = np.kron(psi, phi)
                overlap = float(np.real(prod_vec.conj() @ residual @ prod_vec))
                if abs(overlap - prev_overlap) < scaled_tol:
                    break
                prev_overlap = overlap

            if overlap > best_overlap:
                best_overlap = overlap
                best_prod_vec = prod_vec

        if best_prod_vec is None or best_overlap < scaled_tol:
            return False  # Stuck: no product state with non-trivial overlap.

        # Subtract as much of |prod><prod| as keeps the residual PSD.
        # Start with the full overlap and back off geometrically on PSD failure.
        # If the product state has a component in the null space of the residual,
        # no positive subtraction is possible and the backoff drives s below tol.
        prod_proj = np.outer(best_prod_vec, best_prod_vec.conj())
        s_subtract = best_overlap
        accepted = False
        for _backoff in range(40):
            candidate = residual - s_subtract * prod_proj
            candidate = (candidate + candidate.conj().T) / 2.0
            min_eig = float(np.linalg.eigvalsh(candidate)[0])
            if min_eig >= -tol:
                residual = candidate
                accepted = True
                break
            s_subtract *= 0.5
            if s_subtract < tol:
                break
        if not accepted:
            return False  # Backoff exhausted; algorithm is stuck.

    return False  # Iteration budget exhausted.


def is_separable(
    state: np.ndarray,
    dim: None | int | list[int] = None,
    level: int = 2,
    tol: float = 1e-8,
    strength: int = 1,
) -> tuple[bool, str]:
    r"""Determine if a given state (given as a density matrix) is a separable state [@wikipediaseparable].

    A multipartite quantum state:
    \(\rho \in \text{D}(\mathcal{H}_1 \otimes \mathcal{H}_2 \otimes \dots \otimes \mathcal{H}_N)\)
    is defined as fully separable if it can be written as a convex combination of product states.

    This function implements several criteria to determine separability, broadly following a similar
    order of checks as seen in tools like QETLAB's `IsSeparable` function [@qetlablink].

    1.  **Input Validation**: Checks if the input `state` is a square, positive semidefinite (PSD)
        NumPy array. Normalizes the trace to 1 if necessary. The `dim` parameter specifying
        subsystem dimensions \(d_A, d_B\) is processed or inferred.

    2.  **Trivial Cases for Separability**:

        - If either subsystem dimension \(d_A\) or \(d_B\) is 1
          (i.e., `min_dim_val == 1`), the state is always separable.

    3.  **Pure State Check (Schmidt Rank)**:

        - If the input state has rank 1 (i.e., it's a pure state), its Schmidt rank is computed.
          A pure state is separable if and only if its Schmidt rank is 1 [@wikipediaschmidt].

        !!! Note
            The more general Operator Schmidt Rank \(\le 2\) condition from
            [@cariello2013separability] is applied after PPT in section 5b
            (below), matching QETLAB's `IsSeparable` behavior.


    4.  **Gurvits-Barnum Separable Ball**:

        - Checks if the state lies within the "separable ball" around the maximally mixed state,
          as defined by Gurvits and Barnum [@gurvits2002largest]. States within this ball are
          guaranteed to be separable.

    5.  **PPT Criterion (Peres-Horodecki)**
        [@peres1996separability], [@horodecki1996separability]:

        - The Positive Partial Transpose (PPT) criterion is a necessary condition for separability.
        - If the state is NPT (Not PPT), it is definitively entangled.
        - If the state is PPT and the total dimension \(d_A d_B \le 6\),
          then PPT is also a *sufficient* condition for separability
          [@horodecki1996separability].

    5b. **Operator Schmidt Rank \(\le 2\)** [@cariello2013separability]:

        - For a PPT state, if the operator Schmidt rank of the density matrix is
          \(\le 2\), the state is separable. This generalizes the pure-state
          Schmidt rank check from section 3 to mixed states and matches QETLAB's
          `OperatorSchmidtRank(X, dim) <= 2` check in `IsSeparable`.

    6.  **3x3 Rank-4 PPT N&S Check (Plücker Coordinates / Breuer / Chen & Djokovic)**:

        - For 3x3 systems, if a PPT state has rank 4, there are known necessary and sufficient conditions
          for separability. These are often related to the vanishing of the "Chow form" or determinants
          of matrices constructed from Plücker coordinates of the state's range
          (e.g., [@breuer2006optimal], [@chen2013separability]).
          The implementation checks if a specific determinant, derived from Plücker coordinates of the state's
          range, is close to zero.

    7.  **Operational Criteria for Low-Rank PPT States (Horodecki et al. 2000)**
        [@horodecki2000constructive]:

        For PPT states (especially when \(d_A d_B > 6\)):

        - If \(\text{rank}(\rho) \le \max(d_A, d_B)\), the state is separable
          (Theorem 1 of the paper).
        - If \(\text{rank}(\rho) \le \text{rank}(\rho_A)\) or
          \(\text{rank}(\rho) \le \text{rank}(\rho_B)\), the state is separable.
          This is the "rank-marginal" corollary of Theorem 1 obtained by
          viewing \(\rho\) as a state on its reduced support
          \(\text{supp}(\rho_A) \otimes \text{supp}(\rho_B)\); matches QETLAB.

        !!! Note
            The rank-sum bound \(\text{rank}(\rho) + \text{rank}(\rho^{T_A}) \le
            2 d_A d_B - d_A - d_B + 2\) from Section IV of the same paper is
            *not* by itself a sufficient condition for separability (see issue
            #1506). It is the regime in which the range of \(\rho\) has a
            finite number of product-vector candidates, enumerable via a
            system of polynomial equations, after which an algorithmic check
            (Theorem 2 of the paper) decides separability. The earlier
            versions of this function short-circuited to "separable" on just
            the bound, giving false positives on e.g. UPB-tile states. The
            full algorithmic check is not implemented here; the bound is no
            longer used.

    8.  **Reduction Criterion (Horodecki & Horodecki 1999)** [@horodecki1998reduction]:

        - The state is entangled if \(I_A \otimes \rho_B - \rho \not\succeq 0\) or
          \(\rho_A \otimes I_B - \rho \not\succeq 0\). This is a check for positive semidefiniteness
          based on the Loewner partial order, not a check for majorization.
        - For PPT states (which is the case if this part of the function is reached),
          this criterion is always satisfied, so its primary strength is for NPT states (already handled).

    9.  **Realignment/CCNR Criteria**:

        - **Basic Realignment (Chen & Wu 2003)** [@chen2003matrix]:
          If the trace norm of the realigned matrix is greater than 1, the state is entangled.
        - **Zhang variant (Zhang et al. 2008)** [@zhang2008entanglement]:
          Uses the purity-based bound on \(\|R(\rho - \rho_A \otimes \rho_B)\|_1\).
        - **Filter Covariance Matrix Criterion (Gittsovich et al. 2008)**
          [@gittsovich2008unifying]: strictly stronger than the basic CCNR.
          First brings \(\rho\) to its filter normal form with maximally mixed
          marginals via iterative local SLOCC filtering (Verstraete-Dehaene-
          De Moor algorithm), then checks Eq. (72) of Proposition IV.15:
          \(\sum_k \xi_k \le \sqrt{d_A d_B (d_A - 1)(d_B - 1)}\). Violation
          implies \(\rho\) is entangled. For two qubits the filter CMC is a
          necessary and sufficient separability test (Remark IV.14).

    10. **Rank-1 Perturbation of Identity for PPT States (Vidal & Tarrach 1999)** [@vidal1999robustness]:

        - PPT states that are very close to a specific type of rank-1 perturbation
          of the identity matrix are separable. This is checked by examining the eigenvalue spectrum:
          if the gap between the second largest and smallest eigenvalues is small,
          the state is determined to be separable.

    11. **2xN Specific Checks for PPT States**:
        For bipartite systems where one subsystem is a qubit (\(d_A=2\)) and the
        other is N-dimensional (\(d_B=N\)), several specific conditions apply:

        - **Johnston's Spectral Condition (2013)** [@johnston2013separability]:
          An inequality involving the largest and smallest eigenvalues of a 2xN PPT state that is sufficient
          for separability.
        - **Hildebrand's Conditions (2005, 2007, 2008)**
            [@hildebrand2007positive],
            [@hildebrand2008semidefinite],
            [@hildebrand2005comparison]:

            - For a 2xN state written in block form \(\rho = [[A, B], [B^\dagger, C]]\),
              a check is performed based on the rank of the anti-Hermitian part of the off-diagonal block
              \(B\) (i.e., \(\text{rank}(B - B^\dagger) \le 1\)).
              (Note: QETLAB refers to this property in relation to "perturbed block Hankel" matrices).
            - A check involving a transformed matrix \(X_{2n\_ppt\_check}\)
              derived from blocks A, B, C, requiring it and its partial transpose
              to be PSD (related to homothetic images).
            - A condition based on the Frobenius norm of block \(B\) compared to
              eigenvalues of blocks \(A\) and \(C\).

    12. **Positive (but not completely positive) Maps / Entanglement Witnesses**:
        These tests apply positive but not completely positive maps. If the resulting state is not PSD,
        the original state is entangled. Both decomposable and indecomposable maps appear in this section
        (e.g. Choi's 1975 map and Breuer-Hall are indecomposable; the Ha-Kye family contains both
        decomposable and indecomposable members depending on its parameter).

        - **Choi's 1975 Map (3x3 systems)** [@choi1975]: A canonical
          indecomposable positive map on \(M_3\), applied to both subsystems
          in turn. Distinct from (and complementary to) the Ha-Kye parametric
          family below.
        - **UPB-based Witness (Terhal 2000, tile UPB)** [@terhal2000family]:
          Hermitian entanglement witness constructed directly from the tile
          UPB product vectors in \(\mathbb{C}^3 \otimes \mathbb{C}^3\), of
          the form
          \(H = \sum_i |\alpha_i\rangle\langle\alpha_i|\otimes|\beta_i\rangle\langle\beta_i|
          - d\varepsilon|\Psi\rangle\langle\Psi|\)
          where \(\varepsilon\) is the minimum of
          \(\sum_i |\langle\phi_A|\alpha_i\rangle|^2 |\langle\phi_B|\beta_i\rangle|^2\)
          over unit product states. Evaluated as \(\mathrm{tr}(H\rho) < 0\),
          not via `partial_channel`.
        - **Ha-Kye Maps (3x3 systems)** [@ha2011positive]: Specific maps
          for qutrit-qutrit systems.
        - **Breuer-Hall Maps (even dimensions)** [@breuer2006optimal], [@hall2006indecomposable]:
          Maps based on antisymmetric unitary matrices, applicable when a subsystem
          has even dimension.

    12b. **Iterative Product-State Subtraction (Gühne / sk_iterate)**:

        A constructive sufficient test for separability. Uses alternating
        rank-1 maximization (with a handful of random restarts) to find a
        product state \(|\psi\rangle|\phi\rangle\) with high overlap on
        the residual density matrix, then subtracts as much of
        \(|\psi\rangle\langle\psi| \otimes |\phi\rangle\langle\phi|\) as
        positivity allows and repeats. If the residual shrinks into the
        Gurvits-Barnum separable ball or to numerical zero, the state is
        declared separable. Otherwise the algorithm falls through silently
        to the DPS hierarchy below — this is a *one-sided* witness and
        never returns an entanglement verdict.

    13. **Symmetric Extension Hierarchy (DPS)** [@doherty2004complete]:

        - A state is separable if and only if it has a k-symmetric extension for all \(k \ge 1\).
        - Failing to find a k-symmetric extension at any tested level proves
          entanglement.
        - Passing a finite number of extension levels does **not** by itself
          prove separability; finite k-extendibility is only a relaxation.
        - To prove separability at finite `level`, this function additionally
          checks the Navascués-Owari-Plenio inner cone via
          `has_symmetric_inner_extension`, which is a sufficient test for
          separability based on bosonic symmetric extensions.

        !!! Note
            The symmetric extension check requires CVXPY and a suitable solver. If these are not installed,
            or if the solver fails, a warning is printed to the console and this check is skipped.

        !!! Note
            This matches QETLAB's split: `SymmetricExtension` is used as a
            one-sided entanglement test, while `SymmetricInnerExtension`
            supplies the one-sided separability proof.


    Args:
        state: The density matrix to check.
        dim: The dimension of the input state, e.g., [dim_A, dim_B]. Optional; inferred if None.
        level:
            - Controls only the depth of the DPS symmetric-extension hierarchy
              (default: 2). All other post-PPT checks run regardless of
              `level` (provided `strength` does not cut them off early).
            - If `level == 1`, no DPS SDP is run; finite 1-extendibility adds
              no information beyond the earlier PPT checks.
            - If `level >= 2`, the function checks for a k-symmetric extension
              and the corresponding inner-extension relaxation for every k from
              2 up to `level`.
            - `strength == 0` triggers an early inconclusive return before the
              DPS block is reached, so `level` is effectively ignored in that
              mode (see `strength` below).
        tol: Numerical tolerance (default: 1e-8).
        strength:
            Controls how thoroughly the function checks for separability. `strength`
            picks *which* families of checks run; `level` continues to pick *how
            deep* the DPS hierarchy goes once DPS is running.

            - `strength = 0` — quick-check mode. Runs only the fast
              pre-checks (trivial cases, pure-state Schmidt rank,
              Gurvits-Barnum separable ball, PPT, and the PPT <= 6 dimension
              sufficiency), then returns an inconclusive verdict. All later
              checks (3x3 rank-4 Plucker, Horodecki rank bounds, reduction,
              realignment/CCNR, Vidal-Tarrach, 2xN Johnston/Hildebrand,
              Ha-Kye and Breuer-Hall witnesses, DPS hierarchy) are skipped.
              Useful when you want a cheap answer or are batch-processing
              many states and only care about the easy cases.
            - `strength = 1` (default) — runs every check currently
              implemented in the function, including the iterative
              product-state subtraction constructive witness (section 12b)
              that was added alongside this parameter's expansion.
            - `strength >= 2` — reserved for future expensive criteria
              (additional positive maps from the UPB/Terhal family, refined
              Breuer rank-4 check); currently equivalent to `strength = 1`.
            - `strength = -1` — alias for "run every implemented check".
              Currently equivalent to `strength = 1`, will grow with future
              additions.

    Returns:
        A 2-tuple `(separable, reason)` where `separable` is `True` if a sufficient
        separability criterion fired and `False` if an entanglement witness fired
        or no criterion proved separability, and `reason` is a short human-readable
        string naming the criterion that produced the verdict. Every return path
        provides a non-empty reason, including trivial and inconclusive cases.

    Raises:
        Warning: If the symmetric extension check is attempted but CVXPY or a suitable solver is not available.
        TypeError: If the input `state` is not a NumPy array.
        RuntimeError: If the symmetric extension check is attempted but fails due to CVXPY solver issues.
        NotImplementedError: If the symmetric extension check is attempted but the level is not implemented
            (e.g., level < 1).
        ValueError:
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


    Examples:
        Consider the following separable (by construction) state:

        \[
            \rho = \rho_1 \otimes \rho_2,
        \]

        \[
        \begin{aligned}
            \rho_1 &= \frac{1}{2} \left(|0 \rangle \langle 0| + |0 \rangle \langle 1|
                    + |1 \rangle \langle 0| + |1 \rangle \langle 1| \right), \\
            \rho_2 &= \frac{1}{2} \left( |0 \rangle \langle 0| + |1 \rangle \langle 1| \right).
        \end{aligned}
        \]

        The resulting density matrix will be:

        \[
            \rho =  \frac{1}{4} \begin{pmatrix}
                    1 & 0 & 1 & 0 \\
                    0 & 1 & 0 & 1 \\
                    1 & 0 & 1 & 0 \\
                    0 & 1 & 0 & 1
                    \end{pmatrix} \in \text{D}(\mathcal{X}).
        \]

        We provide the input as a density matrix \(\rho\).

        On the other hand, a random density matrix will be an entangled state (a separable state).

        ```python exec="1" source="above" result="text" session="is_separable_example"
        import numpy as np
        from toqito.rand.random_density_matrix import random_density_matrix
        from toqito.state_props.is_separable import is_separable
        rho_separable = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        sep, reason = is_separable(rho_separable)
        print(f"separable={sep}, reason={reason}")
        ```

        ```python exec="1" source="above" result="text" session="is_separable_example"
        rho_not_separable = np.array([[ 0.13407875+0.j        , -0.08263926-0.17760437j,
            -0.0135111 -0.12352182j,  0.0368423 -0.05563985j],
        [-0.08263926+0.17760437j,  0.53338542+0.j        ,
            0.19782968-0.04549732j,  0.11287093+0.17024249j],
        [-0.0135111 +0.12352182j,  0.19782968+0.04549732j,
            0.21254612+0.j        , -0.00875865+0.11144344j],
        [ 0.0368423 +0.05563985j,  0.11287093-0.17024249j,
            -0.00875865-0.11144344j,  0.11998971+0.j        ]])
        sep, reason = is_separable(rho_not_separable)
        print(f"separable={sep}, reason={reason}")
        ```

        We can also detect certain PPT-entangled states. For example, a state constructed from a Breuer-Hall map
        is entangled but PPT.

        ```python exec="1" source="above" result="text" session="is_separable_example"
        from toqito.state_props.is_ppt import is_ppt

        # Construct a 2x3 separable PPT state of rank 2
        # |ψ₁⟩ = |0⟩⊗|0⟩, |ψ₂⟩ = |1⟩⊗|1⟩
        psi1 = np.kron([1, 0], [1, 0, 0])
        psi2 = np.kron([0, 1], [0, 1, 0])
        rho = 0.5 * (np.outer(psi1, psi1.conj()) + np.outer(psi2, psi2.conj()))

        print("Is the state PPT?", is_ppt(rho, dim=[2, 3]))         # True
        sep, reason = is_separable(rho, dim=[2, 3])
        print(f"Is the state separable? {sep} (reason: {reason})")
        ```

    """
    # --- 1. Input Validation, Normalization, Dimension Setup ---
    if not isinstance(state, np.ndarray):
        raise TypeError("Input state must be a NumPy array.")
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("Input state must be a square matrix.")

    # Validate and normalize `strength`. Documented values are -1, 0, and any
    # integer >= 1; anything else (including bools, floats, and other negatives)
    # is rejected so typos don't silently behave like the full run.
    if isinstance(strength, bool) or not isinstance(strength, (int, np.integer)):
        raise ValueError(f"`strength` must be an int; got {type(strength).__name__}.")
    if strength < -1:
        raise ValueError(f"`strength` must be -1, 0, or a positive integer; got {strength}.")
    # Normalize: -1 and any value >= 1 all mean "run every implemented check",
    # which today is identical behavior. Collapse them to 1 so downstream logic
    # only has to distinguish 0 vs non-0.
    if strength != 0:
        strength = 1

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
        return True, "trivial: empty state"

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
        return True, "trivial: one subsystem has dimension 1"

    # --- 3. Pure State Check (Schmidt Rank) ---
    # A pure state (rank 1) is separable if and only if its Schmidt rank is 1.
    # (The condition `s_rank <= 2` was previously here, referencing Cariello for weak irreducible matrices;
    # however, for general pure states, s_rank=1 is the N&S condition.
    # TODO: look at #1245 Consider adding a separate check for OperatorSchmidtRank <= 2 for general mixed states
    # if they are determined to be "weakly irreducible", as per Cariello [@cariello2013separability]
    # and QETLAB's implementation. This is distinct from this pure state check.)
    if state_rank == 1:
        s_rank = schmidt_rank(current_state, dims_list)
        if s_rank == 1:
            return True, "pure state with Schmidt rank 1"
        return False, f"pure state with Schmidt rank {int(s_rank)} > 1"

    # --- 4. Gurvits-Barnum Separable Ball ---
    if in_separable_ball(current_state):
        # Determined to be separable by closeness to the maximally mixed state [@gurvits2002largest].
        return True, "lies within the Gurvits-Barnum separable ball"

    # --- 5. PPT (Peres-Horodecki) Criterion ---
    is_state_ppt = is_ppt(state, 2, dim, tol)  # sys=2 implies partial transpose on the second system by default
    if not is_state_ppt:
        # Determined to be entangled via the PPT criterion [@peres1996separability].
        # Also, see Horodecki Theorem in [@guhne2009entanglement].
        return False, "NPT (Peres-Horodecki PPT criterion)"

    # ----- From here on, the state is known to be PPT -----

    # --- 5a. PPT and dim <= 6 implies separable ---
    if prod_dim_val <= 6:  # e.g., 2x2 or 2x3 systems
        # For dA * dB <= 6, PPT is necessary and sufficient for separability
        # [@horodecki1996separability].
        return True, "PPT with d_A * d_B <= 6 (Horodecki 1996)"

    # ----- Strength cutoff -----
    # At `strength == 0`, only the fast pre-checks above (trivial, pure state,
    # separable ball, PPT, PPT <= 6) run. Everything below this point — the
    # operator Schmidt rank check, 3x3 rank-4 Plucker determinant, Horodecki
    # rank bounds, reduction, realignment/CCNR, Vidal-Tarrach, 2xN conditions,
    # Ha-Kye/Breuer-Hall witnesses, and the DPS hierarchy — is skipped, and
    # the function returns an inconclusive verdict. This is the "quick check"
    # mode.
    if strength == 0:
        return False, "inconclusive: strength=0 capped after PPT pre-checks"

    # --- 5b. Operator Schmidt Rank <= 2 (Cariello 2013) ---
    # For PPT states, if the operator Schmidt rank of the density matrix is
    # <= 2, the state is separable [@cariello2013separability]. This generalizes
    # the pure-state Schmidt rank check in section 3 to mixed states, and
    # matches QETLAB's `IsSeparable` use of `OperatorSchmidtRank(X, dim) <= 2`.
    #
    # The operator Schmidt coefficients of `rho` are exactly the singular
    # values of its realignment, so the operator Schmidt rank equals
    # `matrix_rank(R(rho))`. We compute it directly with `matrix_rank(..., tol=tol)`
    # instead of calling `schmidt_rank`, which ignores the caller's tolerance.
    op_schmidt_rank = np.linalg.matrix_rank(realignment(current_state, dim=dims_list), tol=tol)
    if op_schmidt_rank <= 2:
        return True, f"operator Schmidt rank = {int(op_schmidt_rank)} <= 2 (Cariello 2013)"

    # --- 6. 3x3 Rank-4 PPT N&S Check (Chen-Djokovic 2013) ---
    # For 3x3 PPT states of rank 4, separability is equivalent to the range
    # containing a product vector [@chen2013separability]. Use a direct
    # range-projector optimization rather than the brittle determinant-only
    # shortcut that previously lived here.
    if dA == 3 and dB == 3 and state_rank == 4:
        best_overlap = _range_projector_product_overlap_3x3_rank4(current_state, tol)
        if best_overlap is not None:
            if 1.0 - best_overlap < 10 * tol:
                return True, "3x3 rank-4 PPT: range contains a product vector (Chen-Djokovic 2013)"
            if 1.0 - best_overlap > 1e-3:
                return (
                    False,
                    f"3x3 rank-4 PPT: range-optimization found no product vector (best overlap={best_overlap:.6f})",
                )

    # --- 7. Operational Criteria for Low-Rank PPT States (Horodecki et al. 2000) ---
    # These are sufficient conditions for separability of PPT states [@horodecki2000constructive].
    if state_rank <= max_dim_val:  # rank(rho) <= max(dA, dB)
        return True, "rank(rho) <= max(d_A, d_B) (Horodecki et al. 2000)"

    # Note on the rank-sum bound `rank(rho) + rank(rho^T_A) <= 2 d_A d_B - d_A - d_B + 2`
    # from Horodecki et al. 2000 (Section IV): this bound is NOT by itself a
    # sufficient condition for separability, contrary to how earlier versions of
    # this function treated it (see issue #1506). The paper establishes it as
    # the regime in which the number of product vectors in the range of rho can
    # be enumerated in a *finite* number via a system of polynomial equations.
    # Separability then requires running the algorithmic check (Theorem 2,
    # Section IV.C) on those candidates. Treating the bound alone as sufficient
    # gives false positives — most starkly, UPB-tile-like 3x3 rank-4 PPT states
    # satisfy the bound but are bound-entangled (their range is a completely
    # entangled subspace, so the paper's check would find zero candidates and
    # correctly return False).
    #
    # We drop the unconditional `return True` here. The algorithmic check is a
    # substantial separate implementation and is not attempted in this pass;
    # states that would previously have been (wrongly) returned separable by
    # this branch now fall through to the reduction / realignment / 2xN /
    # witness / DPS / iterative-subtraction stages below.

    # Rank-marginal condition [@horodecki2000constructive]: for a PPT state,
    # if rank(rho) <= rank(rho_A) or rank(rho) <= rank(rho_B), then rho is
    # separable. Matches QETLAB's corresponding check in `IsSeparable`.
    rho_A_marginal = partial_trace(current_state, sys=[1], dim=dims_list)
    rho_B_marginal = partial_trace(current_state, sys=[0], dim=dims_list)
    rank_marg_A = np.linalg.matrix_rank(rho_A_marginal, tol=tol)
    rank_marg_B = np.linalg.matrix_rank(rho_B_marginal, tol=tol)
    if state_rank <= rank_marg_A:
        return True, f"rank(rho)={state_rank} <= rank(rho_A)={rank_marg_A} (Horodecki et al. 2000)"
    if state_rank <= rank_marg_B:
        return True, f"rank(rho)={state_rank} <= rank(rho_B)={rank_marg_B} (Horodecki et al. 2000)"

    # --- 8. Reduction Criterion (Horodecki & Horodecki 1999) ---
    # If state is PPT (which it is at this point), this criterion is always satisfied.
    # Its main use is for NPT states. Included for completeness of listed criteria.
    # rho_A_marginal and rho_B_marginal were already computed in section 7 above.
    op_reduct1 = np.kron(np.eye(dA), rho_B_marginal) - current_state
    op_reduct2 = np.kron(rho_A_marginal, np.eye(dB)) - current_state
    if not (
        is_positive_semidefinite(op_reduct1, atol=tol, rtol=tol)
        and is_positive_semidefinite(op_reduct2, atol=tol, rtol=tol)
    ):  #  (should not be hit for PPT states)
        return False, "reduction criterion violated (Horodecki 1999)"

    # --- 9. Realignment/CCNR Criteria ---
    # Basic Realignment Criterion [@chen2003matrix]. If > 1, entangled.
    if trace_norm(realignment(current_state, dims_list)) > 1 + tol:
        return False, "realignment/CCNR: ||R(rho)||_1 > 1 (Chen-Wu 2003)"

    # Zhang et al. 2008 Variant [@zhang2008entanglement].
    # If ||R(rho - rho_A \otimes rho_B)||_1 > sqrt((1-Tr(rho_A^2))(1-Tr(rho_B^2))), entangled.
    tr_rho_A_sq = np.real(np.trace(rho_A_marginal @ rho_A_marginal))
    tr_rho_B_sq = np.real(np.trace(rho_B_marginal @ rho_B_marginal))
    val_A = max(0, 1 - tr_rho_A_sq)  # Ensure non-negativity from (1 - purity)
    val_B = max(0, 1 - tr_rho_B_sq)
    bound_zhang = np.sqrt(val_A * val_B) if (val_A * val_B >= 0) else 0
    if trace_norm(realignment(current_state - np.kron(rho_A_marginal, rho_B_marginal), dims_list)) > bound_zhang + tol:
        return False, "Zhang realignment variant: ||R(rho - rho_A (x) rho_B)||_1 exceeds purity bound (Zhang 2008)"

    # --- 9b. Filter Covariance Matrix Criterion (Gittsovich et al. 2008) ---
    # Strictly stronger than the basic CCNR/realignment criterion; for two qubits
    # (Remark IV.14 of the paper) it is a necessary and sufficient separability
    # test. We first transport rho to its filter normal form with maximally
    # mixed marginals via local SLOCC filtering, then check the paper's
    # Proposition IV.15 Eq. (72) bound on the sum of the filter-CMC coefficients.
    # Skipped silently (no verdict) when the filtering iteration can't converge
    # — e.g., when a marginal is rank-deficient and no invertible filter exists.
    rho_normal = _filter_normal_form(current_state, dims_list, tol)
    if rho_normal is not None:
        xi_sum = _filter_cmc_xi_sum(rho_normal, dims_list)
        xi_bound = _filter_cmc_bound(dA, dB)
        if xi_sum > xi_bound + tol:
            return False, (
                f"Filter CMC: sum xi = {xi_sum:.4g} > bound {xi_bound:.4g} "
                f"(Gittsovich et al. 2008, Prop. IV.15)"
            )

    # --- 10. Rank-1 Perturbation of Identity for PPT States (Vidal & Tarrach 1999) ---
    # PPT states close to identity are separable [@vidal1999robustness].
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
                return True, "PPT state close to rank-1 identity perturbation (Vidal-Tarrach 1999)"
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

            # Johnston's Spectral Condition [@johnston2013separability]
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
                    return True, "Johnston spectral condition for 2xN PPT states (2013)"

            # Hildebrand's Conditions for 2xN PPT states (various papers, e.g.,
            # [@hildebrand2005comparison], [@hildebrand2008semidefinite])
            # Block matrix form: rho_2xn = [[A, B], [B^dagger, C]]
            A_block = state_t_2xn[:d_N_val, :d_N_val]
            B_block = state_t_2xn[:d_N_val, d_N_val : 2 * d_N_val]
            C_block = state_t_2xn[d_N_val : 2 * d_N_val, d_N_val : 2 * d_N_val]

            # If rank of anti-Hermitian part of B is small (related to "perturbed block Hankel" in QETLAB)
            if B_block.size > 0 and np.linalg.matrix_rank(B_block - B_block.conj().T, tol=tol) <= 1:
                return True, "Hildebrand 2xN condition: rank(B - B^dagger) <= 1"

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
                    return True, "Hildebrand 2xN homothetic-image condition (PSD and PPT)"

                # Johnston Lemma 1 variant / norm B condition
                try:
                    eig_A_real, eig_C_real = np.real(np.linalg.eigvals(A_block)), np.real(np.linalg.eigvals(C_block))
                    if eig_A_real.size > 0 and eig_C_real.size > 0 and B_block.size > 0:
                        if np.linalg.norm(B_block) ** 2 <= np.min(eig_A_real) * np.min(eig_C_real) + tol**2:
                            return True, "Johnston Lemma 1 for 2xN PPT states: ||B||^2 <= lambda_min(A) * lambda_min(C)"
                except np.linalg.LinAlgError:
                    pass  # Eigenvalue computation failed

    # --- 12. Positive (but not Completely Positive) Map Witnesses ---
    # This section mixes decomposable and indecomposable positive maps — Choi's
    # 1975 map and Breuer-Hall are indecomposable, while the Ha-Kye family
    # contains both decomposable and indecomposable members depending on its
    # parameter. They are all used as entanglement witnesses via partial_channel.
    #
    # Choi's 1975 indecomposable positive map on M_3 [@choi1975].
    # The map Phi_Choi: M_3 -> M_3 acts as
    #     Phi(A)_{00} = A_{00} + A_{22},
    #     Phi(A)_{11} = A_{00} + A_{11},
    #     Phi(A)_{22} = A_{11} + A_{22},
    #     Phi(A)_{ij} = -A_{ij}  for i != j,
    # and is the canonical example of a positive but not completely positive
    # map that is *not* a scalar of the generalized Choi maps from the Ha-Kye
    # family below — the Ha-Kye loop sweeps a different parametric family,
    # so this check is complementary rather than redundant.
    if dA == 3 and dB == 3:
        phi_choi_1975 = _choi_1975_choi_matrix()
        # `partial_channel` uses 1-indexed `sys` (sys=1 is the first subsystem,
        # sys=2 is the second). Iterate 1..2, not 0..1.
        for p_idx_choi in (1, 2):
            if not is_positive_semidefinite(
                partial_channel(current_state, phi_choi_1975, sys=p_idx_choi, dim=dims_list),
                atol=tol,
                rtol=tol,
            ):
                return False, f"Choi 1975 positive-map witness (on subsystem {p_idx_choi}, 3x3)"

        # UPB-based witness from Terhal 2000 [@terhal2000family], built on the
        # tile UPB. Evaluated as a trace: rho is entangled iff tr(H * rho) < 0.
        # The witness matrix is cached at module level; see
        # `_terhal_2000_tile_witness` for the construction (UPB projector minus
        # d * epsilon * |Psi><Psi| with epsilon found by non-convex alternating
        # minimization over product states).
        tr_h_rho = float(np.real(np.trace(_TERHAL_2000_TILE_WITNESS_3X3 @ current_state)))
        if tr_h_rho < -tol:
            return False, (
                f"UPB-based witness on tile UPB (Terhal 2000): tr(H*rho)={tr_h_rho:.4g} < 0"
            )

    # Ha-Kye Maps for 3x3 systems [@ha2011positive]
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
                    return False, f"Ha-Kye positive-map witness (3x3, t={t_iter_ha:.4g})"

    # Breuer-Hall Maps (for even dimensional subsystems) [@breuer2006optimal],
    # [@hall2006indecomposable]
    # `partial_channel` uses 1-indexed `sys` (sys=1 is the first subsystem,
    # sys=2 is the second). Iterate 1..2 and index `dims_list` with `sys-1`.
    for p_idx_bh in (1, 2):  # Apply map to subsystem 1 (A), then subsystem 2 (B)
        current_dim_bh = dims_list[p_idx_bh - 1]  # Dimension of the subsystem map acts on
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
                return False, f"Breuer-Hall positive-map witness (on subsystem {p_idx_bh}, dim={current_dim_bh})"

    # --- 12b. Iterative Product-State Subtraction (Guhne / sk_iterate) ---
    # Try to constructively prove separability by iteratively finding and
    # subtracting product states from the residual. This is a one-sided
    # witness: if it succeeds, the state is separable; if it stalls, the
    # caller falls through to DPS. We run it before DPS because a successful
    # constructive proof is cheaper than a full SDP solve, and because DPS
    # gives only a hierarchy-level verdict rather than a decomposition.
    if _iterative_product_state_subtraction(current_state, dims_list, tol):
        return True, "iterative product-state subtraction decomposed the state (Guhne / sk_iterate)"

    # --- 13. Symmetric Extension Hierarchy (DPS) ---
    # A state is separable iff it has a k-symmetric extension for all k [@doherty2004complete],
    # but finite k-extendibility is only a relaxation. Thus:
    # - if the state is NOT k-extendible at any tested level, it is entangled;
    # - if it lies in the corresponding inner cone, it is separable;
    # - otherwise, passing the tested k-extendibility levels is only inconclusive.
    if level >= 2:
        for k_actual_level_check in range(2, int(level) + 1):
            try:
                if not has_symmetric_extension(rho=current_state, level=k_actual_level_check, dim=dims_list, tol=tol):
                    return False, f"no {k_actual_level_check}-symmetric extension (DPS hierarchy)"
                if has_symmetric_inner_extension(
                    rho=current_state, level=k_actual_level_check, dim=dims_list, ppt=True, tol=tol
                ):
                    return True, f"passed inner DPS symmetric extension cone at level={k_actual_level_check}"
            except ImportError:
                print("Warning: CVXPY or a solver is not installed; cannot perform symmetric extension check.")
                break
            except Exception as e:
                print(f"Warning: Symmetric extension check failed at level {k_actual_level_check} with an error: {e}")
                break

    # If all implemented checks are inconclusive, and the state passed PPT (the most basic necessary condition checked),
    # it implies that the state is either separable but not caught by the simpler sufficient conditions,
    # or it's a PPT entangled state that also wasn't caught by other implemented witnesses
    # or the DPS hierarchy up to `level`.
    # Defaulting to False implies we couldn't definitively prove separability with the implemented tests.
    return False, "inconclusive: PPT but no implemented sufficient condition proved separability"
