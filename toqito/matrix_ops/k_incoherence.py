import numpy as np
import cvxpy as cp


def is_k_incoherence(rho: np.ndarray, k: int) -> bool:
    r"""
    Determine whether a density matrix is $k$-incoherent.

    A density matrix :math:`\rho` is said to be *$k$-incoherent* if it can be expressed
    as a convex combination of pure states, each of which has support on at most
    :math:`k` basis vectors. In other words, each pure state in the mixture has at most
    :math:`k` non-zero entries when written in the reference basis.

    Equivalently, this is the same as requiring that the matrix :math:`\rho` has 
    factor width at most :math:`k`. This function checks whether such a decomposition 
    exists, and returns ``True`` if so, and ``False`` otherwise.

    Examples
    ========
    .. jupyter-execute::

                       import numpy as np
                       from toqito.state_props.is_k_incoherent import is_k_incoherent

                       # A diagonal matrix is 1-incoherent (fully incoherent)
                       rho_diag = np.diag([0.5, 0.5])
                       print(is_k_incoherent(rho_diag, 1))
                       # True

                       # A state with off-diagonal coherence is not 1-incoherent
                       rho_coh = np.array([[0.5, 0.4],[0.4, 0.5]])
                       print(is_k_incoherent(rho_coh, 1))
                       # False
                       print(is_k_incoherent(rho_coh, 2))
                       # True

    The definition of k-incoherence can be found in :cite:`Nathaniel_2023_Tight`.

    References
    ==========
   .. bibliography::
        :filter: docname in docnames


    :param rho : A square Hermitian matrix representing a quantum state (density matrix).
                 Must be positive semidefinite and trace 1.
    :param k : The incoherence parameter. Must be a positive integer no larger than the
               dimension of the Hilbert space.
    :return : ``True`` if :math:`\rho` is $k$-incoherent; ``False`` otherwise.

    """
    rho = np.array(rho, dtype=complex)
    n, m = rho.shape if rho.ndim == 2 else (None, None)
    if n is None or n != m:
        raise ValueError("Input matrix must be square.")
    if k is None or not isinstance(k, (int, np.integer)):
        raise ValueError("Parameter k must be an integer.")
    if k < 1:
        raise ValueError("Parameter k must be a positive integer (k >= 1).")
    if k >= n:
        # If k is greater or equal to the dimension, any state is k-incoherent (no restriction).
        return True

    # Verify rho is Hermitian PSD (within numerical tolerance).
    if not np.allclose(rho, rho.conj().T, atol=1e-8):
        # If not Hermitian, it cannot be a valid density matrix.
        raise ValueError("Input matrix must be Hermitian (equal to its conjugate transpose).")
    # Optional: We could check for positive semidefiniteness (e.g., np.linalg.eigvals >= -tol).
    eigvals = np.linalg.eigvalsh(rho)
    if np.min(eigvals) < -1e-8:
        raise ValueError("Input matrix must be positive semidefinite (non-negative eigenvalues).")

    # If rho is effectively rank-1 (a pure state), directly check its support size.
    # (Count of non-zero components in some computational basis representation.)
    # We use the basis in which rho is diagonal (its eigenbasis) for this check.
    if np.count_nonzero(eigvals > 1e-8) == 1:
        # Extract the (normalized) pure state vector.
        # If rho has one non-zero eigenvalue, the corresponding eigenvector is the pure state.
        vec = np.linalg.eigh(rho)[1][:, -1]
        support_size = np.count_nonzero(np.abs(vec) > 1e-8)
        return support_size <= k

    # If k == 1, require rho to be diagonal in the computational basis.
    if k == 1:
        off_diag = rho - np.diag(np.diag(rho))
        return np.allclose(off_diag, 0, atol=1e-8)

    # General case: Set up an SDP to find a k-incoherent decomposition.
    # We attempt to express rho = sum_{S in Omega} F_S, where each F_S is PSD and supported on a subset S of indices (|S| <= k).
    # Omega = collection of all index subsets of {0,...,n-1} of size 1 <= |S| <= k.
    indices = range(n)
    subsets = []
    from itertools import combinations
    for r in range(1, k+1):
        for S in combinations(indices, r):
            subsets.append(S)

    # Create PSD variable for each subset support.
    F_vars = []
    constraints = []
    for S in subsets:
        size = len(S)
        # Represent F_S as an n×n matrix variable, but constrain it to have support only on S.
        F = cp.Variable((n, n), hermitian=True)  # Hermitian PSD variable
        constraints.append(F >> 0)  # PSD constraint
        # Constrain entries outside SxS to zero:
        for i in range(n):
            for j in range(n):
                if not (i in S and j in S):
                    constraints.append(F[i, j] == 0)
        F_vars.append(F)
    constraints.append(sum(F_vars) - rho == 0)

    # Solve the SDP (feasibility problem). Use a CVXPy solver suitable for semidefinite programs.
    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        result = prob.solve(solver=cp.SCS, verbose=False)
    except Exception as e:
        # If solver fails (e.g., no solver available or numerical issues), we handle accordingly.
        raise RuntimeError("Solver failed to solve the k-incoherence SDP: " + str(e))

    # If the problem is solved successfully, check feasibility.
    return prob.status in ["optimal", "optimal_inaccurate"]
