"""Computes the S(k)-norm of a matrix."""

import warnings

import cvxpy
import numpy as np
import scipy

from toqito._sdp import solve_sdp
from toqito._symmetric_extension import SymmetricExtension
from toqito.channels.realignment import realignment
from toqito.matrix_ops._vec_maps import (
    _apply_vec_map,
    _vec_kron_identity,
    _vec_partial_trace,
    _vec_partial_transpose_perm,
)
from toqito.matrix_props.is_hermitian import is_hermitian
from toqito.matrix_props.kp_norm import kp_norm
from toqito.state_props.schmidt_rank import schmidt_rank
from toqito.state_props.sk_vec_norm import sk_vector_norm


def sk_operator_norm(
    mat: np.ndarray,
    k: int = 1,
    dim: int | list[int] | None = None,
    target: float | None = None,
    effort: int = 2,
) -> tuple[float, float]:
    r"""Compute the S(k)-norm of a matrix [@johnston2010family].

    The \(S(k)\)-norm of a matrix \(X\) is defined as:

    \[
        \big|\big| X \big|\big|_{S(k)} := sup_{|v\rangle, |w\rangle}
        \Big\{
            |\langle w | X |v \rangle| :
            \text{Schmidt - rank}(|v\rangle) \leq k,
            \text{Schmidt - rank}(|w\rangle) \leq k
        \Big\}
    \]

    Since computing the exact value of S(k)-norm [@johnston2012norms] is in the general case an intractable
    problem, this function tries to find some good lower and upper bounds. You can control the amount of computation you
    want to devote to computing the bounds by `effort` input argument. Note that if the input matrix is not positive
    semidefinite the output bounds might be quite poor.

    This function was adapted from QETLAB [@qetlablink].

    Args:
        mat: A matrix.
        k: The "index" of the norm--that is, it is the Schmidt rank of the vectors that are multiplying X on the left
            and right in the definition of the norm.
        dim: The dimension of the two sub-systems. By default it's assumed to be equal.
        target: A target value that you wish to prove that the norm is above or below.
        effort: An integer value indicating the amount of computation you want to devote to computing the bounds.

    Returns:
        A lower and an upper bound on S(k)-norm of `mat`.

    Raises:
        ValueError: If dimension of the input matrix is not specified.

    Examples:
        The \(S(1)\)-norm of a Werner state \(\rho_a \in M_n \otimes M_n\) is

        \[
            \big|\big| \rho_a \big|\big|_{S(1)} = \frac{1 + |min\{a, 0\}|}{n (n - a)}
        \]

        ```python exec="1" source="above" result="text"
        from toqito.states.werner import werner
        from toqito.matrix_props.sk_norm import sk_operator_norm

        rho = werner(4, 0.)

        print(sk_operator_norm(rho))
        ```

    """
    eps = np.finfo(float).eps
    tol = eps ** (3 / 8)

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square.")

    dim_xy = mat.shape[0]
    # Set default dimension if none was provided.
    if dim is None:
        dim = int(np.round(np.sqrt(dim_xy)))

    # Allow the user to enter in a single integer for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, dim_xy / dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * dim_xy * np.finfo(float).eps:
            raise ValueError("If `dim` is a scalar, it must evenly divide the length of the matrix.")
        dim[1] = int(np.round(dim[1]))

    dim = np.array(dim, dtype=int)

    # Some useful, repeatedly-used, values.
    prod_dim = np.prod(dim)
    op_norm = np.linalg.norm(mat, ord=2)
    if np.allclose(op_norm, 0.0):
        return 0.0, 0.0

    rank = np.linalg.matrix_rank(mat)
    # Rescale X to have unit norm.
    mat = mat / op_norm

    # The S(k)-norm is just the operator norm if k is large enough.
    if k >= min(dim):
        lower_bound = op_norm
        upper_bound = op_norm
        return lower_bound, upper_bound

    # If X is rank 1 then the S(k)-norm is easy to compute via Proposition 10 of [1].
    if rank == 1:
        u_mat, _, v_mat = np.linalg.svd(mat, full_matrices=False)
        lower_bound = op_norm * sk_vector_norm(u_mat[:, 0], k, dim) * sk_vector_norm(v_mat[0, :], k, dim)
        upper_bound = lower_bound
        return lower_bound, upper_bound

    # Compute some more simple bounds. We will multiply these by op_norm before we leave this function.
    # This comes from Theorem 4.13 in [1].
    lower_bound = k / min(dim)
    # our most basic upper bound
    upper_bound = 1

    # Break out of the function if the target value has already been met.
    if __target_is_proved(lower_bound, upper_bound, op_norm, tol, target):
        return op_norm * lower_bound, op_norm * upper_bound

    # If input is not Hermitian, we don't have better bounds, so we abort.
    if not is_hermitian(mat):
        return op_norm * lower_bound, op_norm * upper_bound

    # Compute eigendecomposition and sort the eigenvalues.
    eig_val, eig_vec = np.linalg.eigh(mat)
    ind = np.argsort(eig_val)
    eig_val = eig_val[ind]
    eig_vec = eig_vec[:, ind]

    atol = 1e-8
    is_positive = all(x >= -abs(atol) for x in eig_val)
    is_projection = False
    if is_positive:
        is_projection = np.allclose(np.linalg.matrix_power(mat, 2), mat)

    is_trans_exact = min(dim) == 2 and max(dim) <= 3

    # If the exact answer won't be found by SDP, compute bounds via other methods first.
    if not (is_positive and is_trans_exact and k == 1 and effort >= 1):
        # Use the lower bound of Proposition 4.14 of [1].
        for r in range(k, min(dim) + 1):
            t_ind = np.prod(dim) - np.prod(dim - r) - 1
            lower_bound = max(lower_bound, (k / r) * eig_val[t_ind])

        # Use the lower bound of Theorem 4.2.15 of [3].
        if k == 1:
            lower_bound = max(
                lower_bound,
                (np.trace(mat) + np.sqrt((prod_dim * np.trace(mat @ mat) - np.trace(mat) ** 2) / (prod_dim - 1)))
                / prod_dim,
            )

        if is_positive:
            # Use the upper bound of Proposition 15 of [1].
            upper_bound = min(
                upper_bound,
                sum(abs(eig_val[i]) * sk_vector_norm(eig_vec[:, i], k, dim) ** 2 for i in range(prod_dim)),
            )

            # Use the upper bound of Proposition 4.2.11 of [3].
            upper_bound = min(upper_bound, kp_norm(realignment(mat, dim), k**2, 2))

        # Use the lower bound of Theorem 4.2.17 of [3].
        if is_projection:
            lower_bound = max(
                lower_bound,
                min(
                    1,
                    k / np.ceil((dim[0] + dim[1] - np.sqrt((dim[0] - dim[1]) ** 2 + 4 * rank - 4)) / 2),
                ),
            )

            lower_bound = max(
                lower_bound,
                (min(dim) - k)
                * (rank + np.sqrt((prod_dim * rank - rank**2) / (prod_dim - 1)))
                / (prod_dim * (min(dim) - 1))
                + (k - 1) / (min(dim) - 1),
            )

        # Break out of the function if the target value has already been met.
        if __target_is_proved(lower_bound, upper_bound, op_norm, tol, target):
            return op_norm * lower_bound, op_norm * upper_bound

        # Use a randomized iterative method to try to improve the lower bound.
        if is_positive:
            for _ in range(5**effort):
                lower_bound = max(
                    lower_bound,
                    __lower_bound_sk_norm_randomized(mat, k, dim, tol**2),
                )

                # Break out of the function if the target value has already been met.
                if __target_is_proved(lower_bound, upper_bound, op_norm, tol, target):
                    return op_norm * lower_bound, op_norm * upper_bound

    # Start the semidefinite programming approach for getting upper bounds.
    bool_cond = [
        (effort >= 1 and lower_bound + tol < upper_bound and is_positive),
        (effort >= 1 and is_positive and is_trans_exact and k == 1),
    ]
    if any(bool_cond):
        rho = cvxpy.Variable((prod_dim, prod_dim), hermitian=True, name="rho")
        objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(mat @ rho)))

        constraints = [rho >> 0, cvxpy.real(cvxpy.trace(rho)) <= 1]
        if k == 1:
            perm = _vec_partial_transpose_perm(dim.tolist(), 1)
            constraints.append(cvxpy.reshape(cvxpy.vec(rho, order="C")[perm], rho.shape, order="C") >> 0)
        else:
            # k Tr_Y(rho) (x) 1_Y >= rho, with the left-hand side as one sparse map of vec(rho).
            reduction = _vec_kron_identity(dim[0], dim[1]) @ _vec_partial_trace(dim.tolist(), [1])
            constraints.append(k * _apply_vec_map(reduction, rho, prod_dim) >> rho)

        cvx_optval = solve_sdp(cvxpy.Problem(objective, constraints))

        upper_bound = min(upper_bound, cvx_optval)

        # In small dimensions, the transpose map gets the result exactly.
        if is_trans_exact and k == 1:
            lower_bound = upper_bound
        elif k == 1:
            # We can also get decent lower bounds from the SDP results when k=1
            # See Theorem 5.2.8 of [2].
            roots, _ = scipy.special.roots_jacobi(1, dim[1] - 2, 1)
            gs = min(1 - roots)
            xmineig = min(eig_val)
            lower_bound = max(
                lower_bound,
                np.real(cvx_optval) * (1 - dim[1] * gs / (2 * dim[1] - 1)) + xmineig * gs / (2 * dim[1] - 2),
            )

            # Done the effort = 1 SDP, now get better upper bounds via symmetric
            # extensions if effort >= 2.
            for j in range(2, effort + 1):
                # Break out of the function if the target value has already been met.
                if __target_is_proved(lower_bound, upper_bound, op_norm, tol, target):
                    return op_norm * lower_bound, op_norm * upper_bound

                # The extension lives on X (x) Sym^j(Y), which enforces its symmetry by construction and keeps the
                # variable small; its trace equals that of the reduced variable.
                extension = SymmetricExtension(int(dim[0]), int(dim[1]), j)
                marginal = extension.partial_trace(list(range(2, j + 1)))
                objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(mat @ marginal)))
                constraints = [
                    extension.variable >> 0,
                    cvxpy.real(cvxpy.trace(extension.variable)) <= 1,
                    extension.partial_transpose(list(range(int(np.ceil(j / 2)) + 1))) >> 0,
                ]
                cvx_optval = solve_sdp(cvxpy.Problem(objective, constraints))

                upper_bound = min(upper_bound, cvx_optval)

                roots, _ = scipy.special.roots_jacobi(np.floor(j / 2) + 1, dim[1] - 2, j % 2)
                gs = min(1 - roots)
                lower_bound = max(
                    lower_bound,
                    np.real(cvx_optval) * (1 - dim[1] * gs / (2 * dim[1] - 1)) + xmineig * gs / (2 * dim[1] - 2),
                )

    lower_bound = op_norm * lower_bound
    upper_bound = op_norm * upper_bound

    return lower_bound, upper_bound


def __target_is_proved(lower_bound: float, upper_bound: float, op_norm: float, tol: float, target: float) -> bool:
    """Check if the lower/upper bound meets the target value."""
    return op_norm * (lower_bound + tol) >= op_norm * upper_bound or (
        target is not None and (op_norm * (lower_bound - tol) >= target or op_norm * (upper_bound + tol) <= target)
    )


def __lower_bound_sk_norm_randomized(
    mat: np.ndarray,
    k: int = 1,
    dim: int | list[int] | None = None,
    tol: float = 1e-5,
    start_vec: np.ndarray | None = None,
) -> float:
    """Compute a lower bound of the S(k)-norm via a randomized method.

    A vector of Schmidt rank at most ``k`` is stored as a ``dim[0] x dim[1]`` matrix of rank at most ``k``.
    Each half-step holds one Schmidt factor fixed, as an orthonormal frame read off the singular value
    decomposition of that matrix, and maximizes the quadratic form over the other factor, which is an
    ordinary Hermitian eigenproblem of size ``k * dim[i]``.
    """
    if mat.shape[0] != mat.shape[1]:  # pragma: no cover - defensive; caller validates shape
        raise ValueError("Input matrix must be square.")

    dim_a, dim_b = (int(d) for d in dim)
    mat4 = np.asarray(mat, dtype=complex).reshape(dim_a, dim_b, dim_a, dim_b)

    v_mat = None
    if start_vec is not None:
        if (s_rank := schmidt_rank(start_vec, dim)) > k:
            warnings.warn(
                f"The Schmidt rank of the initial vector is {s_rank}, which is larger than k={k}. "
                "Using a randomly-generated initial vector instead."
            )
        else:
            v_mat = np.asarray(start_vec, dtype=complex).reshape(dim_a, dim_b)

    if v_mat is None:
        a_mat = np.random.randn(k, dim_a) + 1j * np.random.randn(k, dim_a)
        b_mat = np.random.randn(k, dim_b) + 1j * np.random.randn(k, dim_b)
        v_mat = a_mat.T @ b_mat
    v_mat = v_mat / np.linalg.norm(v_mat)

    v_vec = v_mat.ravel()
    sk_lower_bound = float(np.real(np.vdot(v_vec, mat @ v_vec)))
    it_lower_bound_improved = True

    while it_lower_bound_improved:
        it_lower_bound_improved = False

        # Loop through the 2 parties.
        for p in range(2):
            u_mat, _, vh_mat = np.linalg.svd(v_mat)

            # Fix one of the parties and optimize over the other party.
            if p == 0:
                frame = u_mat[:, :k]
                size = k * dim_b
                reduced = np.einsum("ai,abcd,cj->ibjd", frame.conj(), mat4, frame).reshape(size, size)
            else:
                frame = vh_mat[:k, :]
                size = k * dim_a
                reduced = np.einsum("ib,abcd,jd->iajc", frame.conj(), mat4, frame).reshape(size, size)

            eigvals, eigvecs = np.linalg.eigh(reduced)

            if (new_sk_lower_bound := float(eigvals[-1])) >= sk_lower_bound + tol:
                it_lower_bound_improved = True
                sk_lower_bound = new_sk_lower_bound

                if p == 0:
                    v_mat = frame @ eigvecs[:, -1].reshape(k, dim_b)
                else:
                    v_mat = eigvecs[:, -1].reshape(k, dim_a).T @ frame

    return sk_lower_bound
