"""Checks separability using iterative product state subtraction (IPSS)."""

from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from toqito.matrix_props import is_positive_semidefinite


def _kron_vec(psi: NDArray[np.complex128], phi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return kronecker product vector kron(psi, phi)."""
    return np.kron(psi, phi)


def _partial_trace_B_op_on_phi(
    residual: NDArray[np.complex128],
    phi: NDArray[np.complex128],
    dim_a: int,
    dim_b: int,
) -> NDArray[np.complex128]:
    """
    Compute M = Tr_B(residual · (I_A ⊗ |phi><phi|))
    Returns an (dim_a x dim_a) Hermitian matrix.
    """
    rho_rs = residual.reshape(dim_a, dim_b, dim_a, dim_b)
    # Correct contraction over the B indices (b and b')
    M = np.einsum("abcd,b,d->ac", rho_rs, phi.conj(), phi)
    return (M + M.conj().T) / 2


def _partial_trace_A_op_on_psi(
    residual: NDArray[np.complex128],
    psi: NDArray[np.complex128],
    dim_a: int,
    dim_b: int,
) -> NDArray[np.complex128]:
    """
    Compute N = Tr_A(residual · (|psi><psi| ⊗ I_B))
    Returns an (dim_b x dim_b) Hermitian matrix.
    """
    rho_rs = residual.reshape(dim_a, dim_b, dim_a, dim_b)
    # Correct contraction over the A indices (a and a')
    N = np.einsum("abcd,a,c->bd", rho_rs, psi.conj(), psi)
    return (N + N.conj().T) / 2


def _find_max_overlap_product_state(
    residual: NDArray[np.complex128],
    dim: List[int],
    max_iter: int = 200,
    tol: float = 1e-12,
    n_restarts: int = 5,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    """
    Alternating optimization (with random restarts) to find product vectors
    |psi> |phi> that approximately maximize p^* residual p, where p = kron(psi,phi).
    Returns (psi, phi, overlap).
    """
    if len(dim) != 2:
        raise ValueError("Only bipartite systems supported (dim list of length 2).")

    dim_a, dim_b = dim
    total_dim = dim_a * dim_b
    if residual.shape != (total_dim, total_dim):
        raise ValueError("Residual dimension incompatible with dim")

    best_overlap = -np.inf
    best_psi: Optional[NDArray[np.complex128]] = None
    best_phi: Optional[NDArray[np.complex128]] = None

    for _ in range(max(1, int(n_restarts))):
        # initialize random normalized complex vectors (no global reseed)
        psi = np.random.randn(dim_a) + 1j * np.random.randn(dim_a)
        nrm = np.linalg.norm(psi)
        psi = psi / nrm if nrm > 0 else np.ones(dim_a, dtype=complex) / np.sqrt(dim_a)

        phi = np.random.randn(dim_b) + 1j * np.random.randn(dim_b)
        nrm = np.linalg.norm(phi)
        phi = phi / nrm if nrm > 0 else np.ones(dim_b, dtype=complex) / np.sqrt(dim_b)

        overlap_old = -np.inf

        for _ in range(int(max_iter)):
            # Fix phi, compute M and leading eigenvector
            M = _partial_trace_B_op_on_phi(residual, phi, dim_a, dim_b)
            _, eigvecs_M = np.linalg.eigh(M)
            psi_candidate = eigvecs_M[:, -1]
            nrm = np.linalg.norm(psi_candidate)
            if nrm > 0:
                psi = psi_candidate / nrm

            # Fix psi, compute N and leading eigenvector
            N = _partial_trace_A_op_on_psi(residual, psi, dim_a, dim_b)
            _, eigvecs_N = np.linalg.eigh(N)
            phi_candidate = eigvecs_N[:, -1]
            nrm = np.linalg.norm(phi_candidate)
            if nrm > 0:
                phi = phi_candidate / nrm

            # compute overlap via kron-vector for numerical stability
            p = _kron_vec(psi, phi)
            overlap = float(np.real(np.vdot(p, residual @ p)))

            if abs(overlap - overlap_old) < tol:
                break
            overlap_old = overlap

        if overlap > best_overlap:
            best_overlap = overlap
            best_psi = psi.copy()
            best_phi = phi.copy()

    # numeric floor
    if best_overlap < 0 and best_overlap > -1e-15:
        best_overlap = 0.0

    return best_psi, best_phi, float(best_overlap)


def iterative_product_state_subtraction(
    rho: NDArray[np.complex128],
    dim: List[int],
    max_iterations: int = 200,
    overlap_tol: float = 1e-10,
    residual_tol: float = 1e-8,
    psd_tol: float = -1e-10,
    verbose: bool = False,
    n_restarts_find: int = 7,
    renormalize: bool = False,
    strength: int = 1,
) -> Tuple[bool, List[Tuple[float, NDArray[np.complex128]]], NDArray[np.complex128]]:
    r"""Iterative Product State Subtraction (IPSS) for separability testing.

    Implements a constructive method for proving separability by explicitly
    decomposing a bipartite quantum state into product states.

    At each iteration the algorithm:
        1. Finds product vectors maximizing overlap with the residual state
        2. Subtracts the largest possible multiple preserving positivity
        3. Repeats until convergence or failure

    Features
    ==========
    - Uses alternating eigenvector optimization to identify product states
    - Supports multiple random initializations to mitigate local optima
    - Optional residual renormalization between subtraction steps
    - Search effort controlled via a strength parameter
    - Numerical tolerances to maintain positive semidefiniteness

    References
    ==========
    Gühne, O., et al. "Entanglement detection." Physics Reports 474.1-6 (2009).  
    QETLAB `sk_iterate` routine.

    Parameters
    ==========
    rho : ndarray
        Density matrix acting on a bipartite system.
    dim : list[int]
        Subsystem dimensions [dim_A, dim_B].
    strength : int
        Controls computational effort.

    Returns
    ==========
    bool
        True if the state is certified separable, False otherwise.
    decomposition : list
        Constructive separable decomposition found (if any).
    residual : ndarray
        Remaining residual state.
    """
    if len(dim) != 2:
        raise ValueError("Only bipartite systems supported")

    dim_a, dim_b = dim
    total_dim = int(dim_a * dim_b)
    if rho.shape != (total_dim, total_dim):
        raise ValueError("rho dimension mismatch with provided dim")

    # Hermitize input
    rho = (rho + rho.conj().T) / 2

    # Basic validation
    eigs_rho = np.linalg.eigvalsh(rho)
    if np.min(eigs_rho) < psd_tol - 1e-12:
        raise ValueError("Input state has significantly negative eigenvalue; not a valid density matrix")

    trace_rho = float(np.real(np.trace(rho)))
    if abs(trace_rho - 1.0) > 1e-6:
        raise ValueError(f"Input state trace = {trace_rho}; expected 1")

    # We'll keep a single absolute residual (residual_abs) and optionally form a
    # normalized working residual for searches when renormalize=True.
    residual_abs = rho.copy()
    decomposition: List[Tuple[float, NDArray[np.complex128]]] = []
    total_subtracted = 0.0

    # scale restarts by strength (simple linear scaling)
    effective_restarts = max(1, int(n_restarts_find * max(1, strength)))

    for iteration in range(int(max_iterations)):
        # current absolute residual norm/traces
        residual_norm = np.linalg.norm(residual_abs, ord="fro")
        trace_abs = float(np.real(np.trace(residual_abs)))

        if verbose:
            print(
                f"[IPSS] iter {iteration:3d}: residual norm {residual_norm:.3e}, trace {trace_abs:.3e}"
            )

        # check convergence in absolute residual
        if residual_norm < residual_tol and is_positive_semidefinite(residual_abs, psd_tol):
            if verbose:
                print("[IPSS] Converged: residual small and PSD.")
            return True, decomposition, residual_abs

        # Prepare working residual for the inner search
        if renormalize and trace_abs > 0:
            working_residual = residual_abs / trace_abs
        else:
            working_residual = residual_abs

        # Find product state maximizing overlap on working_residual
        psi, phi, overlap_work = _find_max_overlap_product_state(
            working_residual, dim, n_restarts=effective_restarts
        )
        if verbose:
            print(f"  Found overlap (working frame) = {overlap_work:.6e}")

        # Convert overlap to absolute weight upper bound
        if renormalize:
            weight_hi_abs = trace_abs * float(overlap_work)
        else:
            weight_hi_abs = float(overlap_work)

        # If overlap too small, can't progress
        if weight_hi_abs <= overlap_tol:
            if verbose:
                print("  Overlap/weight too small — cannot make further progress.")
            return False, decomposition, residual_abs

        p_vec = _kron_vec(psi, phi)
        P = np.outer(p_vec, p_vec.conj())  # rank-1 projector

        # If subtracting full weight preserves PSD, accept it; otherwise binary search.
        test_resid_abs = residual_abs - weight_hi_abs * P
        min_eig_test = np.min(np.linalg.eigvalsh((test_resid_abs + test_resid_abs.conj().T) / 2))
        if min_eig_test >= psd_tol:
            weight_abs = weight_hi_abs
        else:
            # binary search over absolute weights in [0, weight_hi_abs]
            weight_lo = 0.0
            weight_hi = weight_hi_abs
            for _ in range(80):
                mid = 0.5 * (weight_lo + weight_hi)
                test_resid_abs = residual_abs - mid * P
                min_eig = np.min(np.linalg.eigvalsh((test_resid_abs + test_resid_abs.conj().T) / 2))
                if min_eig >= psd_tol:
                    weight_lo = mid
                else:
                    weight_hi = mid
                if weight_hi - weight_lo < 1e-14:
                    break
            weight_abs = float(weight_lo)

        # If chosen weight negligible, stop
        if weight_abs < overlap_tol:
            if verbose:
                print("  Best subtractable weight is too small — stopping.")
            return False, decomposition, residual_abs

        # Subtract in absolute frame and record
        residual_abs = residual_abs - weight_abs * P
        decomposition.append((weight_abs, P))
        total_subtracted += weight_abs

        # Numerical cleanup: enforce Hermiticity and clamp tiny negative eigenvalues
        residual_abs = (residual_abs + residual_abs.conj().T) / 2
        evals, evecs = np.linalg.eigh(residual_abs)
        # clamp negative eigenvalues to at least psd_tol (do not push to zero aggressively)
        evals_clamped = np.where(evals < 0, np.maximum(evals, psd_tol), evals)
        residual_abs = (evecs * evals_clamped) @ evecs.conj().T

        # If renormalize is True, the next search will use a normalized working_residual
        # computed at the beginning of the next loop iteration (we keep residual_abs as the
        # absolute matrix so recorded weights are absolute amounts subtracted from rho).

    # reached max iterations without full convergence
    if verbose:
        print("[IPSS] Reached maximum iterations without converging.")
    return False, decomposition, residual_abs


def verify_separable_decomposition(
    rho: NDArray[np.complex128],
    decomposition: List[Tuple[float, NDArray[np.complex128]]],
    atol: float = 1e-7,
) -> bool:
    """
    Verify that sum_i w_i * P_i approximates rho within atol (Frobenius norm).
    decomposition elements must be (weight, product_state_matrix).
    """
    if not decomposition:
        return False
    reconstructed = sum(w * P for w, P in decomposition)
    return np.linalg.norm(rho - reconstructed, ord="fro") < atol


