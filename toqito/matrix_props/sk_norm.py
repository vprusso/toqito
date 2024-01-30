"""Compute the S(k)-norm of a matrix."""


import warnings

import cvxpy
import numpy as np
import scipy

from toqito.channels import partial_trace, partial_transpose, realignment
from toqito.matrix_props import is_hermitian, kp_norm
from toqito.perms import swap, symmetric_projection
from toqito.state_ops.schmidt_decomposition import schmidt_decomposition
from toqito.state_props.schmidt_rank import schmidt_rank
from toqito.state_props.sk_vec_norm import sk_vector_norm
from toqito.states import max_entangled


def sk_operator_norm(  # pylint: disable=too-many-locals
    mat: np.ndarray,
    k: int = 1,
    dim: int | list[int] = None,
    target: float = None,
    effort: int = 2,
) -> float:
    r"""Compute the S(k)-norm of a matrix :cite:`Johnston_2010_AFamily`.

    The :math:`S(k)`-norm of of a matrix :math:`X` is defined as:

    .. math::
        \big|\big| X \big|\big|_{S(k)} := sup_{|v\rangle, |w\rangle}
        \Big\{
            |\langle w | X |v \rangle| :
            \text{Schmidt - rank}(|v\rangle) \leq k,
            \text{Schmidt - rank}(|w\rangle) \leq k
        \Big\}

    Since computing the exact value of S(k)-norm :cite:`Johnston_2012_Norms` is in the general case
    an intractable problem, this function tries to find some good lower and
    upper bounds. You can control the amount of computation you want to
    devote to computing the bounds by `effort` input argument. Note that if
    the input matrix is not positive semidefinite the output bounds might be
    quite pooor.

    This function was adapted from QETLAB.

    Examples
    ========

    The :math:`S(1)`-norm of a Werner state :math:`\rho_a \in M_n \otimes M_n` is

    .. math::
        \big|\big| \rho_a \big|\big|_{S(1)} = \frac{1 + |min\{a, 0\}|}{n (n - a)}

    >>> from toqito.states import werner
    >>> from toqito.matrix_props import sk_operator_norm
    >>>
    >>> # Werner state.
    >>> n = 4; a = 0
    >>> rho = werner(4, 0.)
    >>> sk_operator_norm(rho)
    (0.0625, 0.0625)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If dimension of the input matrix is not specified.
    :param mat: A matrix.
    :param k: The "index" of the norm -- that is, it is the Schmidt rank of the
              vectors that are multiplying X on the left and right in the definition
              of the norm.
    :param dim: The dimension of the two sub-systems. By default it's
                assumed to be equal.
    :param target: A target value that you wish to prove that the norm is above or below.
    :param effort: An integer value indicating the amount of computation you want to
                   devote to computing the bounds.
    :return: A lower and an upper bound on S(k)-norm of :code:`mat`.

    """
    eps = np.finfo(float).eps
    tol = eps ** (3 / 8)

    dim_xy = mat.shape[0]
    # Set default dimension if none was provided.
    if dim is None:
        dim = int(np.round(np.sqrt(dim_xy)))

    # Allow the user to enter in a single integer for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, dim_xy / dim])  # pylint: disable=redefined-variable-type
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * dim_xy * np.finfo(float).eps:
            raise ValueError(
                "If `dim` is a scalar, it must evenly divide the length of the matrix."
            )
        dim[1] = int(np.round(dim[1]))

    dim = np.array(dim, dtype=int)

    # some useful, repeatedly-used, values
    prod_dim = np.prod(dim)
    op_norm = np.linalg.norm(mat, ord=2)
    if np.allclose(op_norm, 0.0):
        return 0.0, 0.0

    rank = np.linalg.matrix_rank(mat)
    # rescale X to have unit norm
    mat = mat / op_norm

    # The S(k)-norm is just the operator norm if k is large enough.
    if k >= min(dim):
        lower_bound = op_norm
        upper_bound = op_norm
        return lower_bound, upper_bound

    # If X is rank 1 then the S(k)-norm is easy to compute via Proposition 10 of [1].
    if rank == 1:
        u_mat, _, v_mat = np.linalg.svd(mat, full_matrices=False)
        lower_bound = (
            op_norm * sk_vector_norm(u_mat[:, 0], k, dim) * sk_vector_norm(v_mat[0, :], k, dim)
        )
        upper_bound = lower_bound
        return lower_bound, upper_bound

    # Compute some more simple bounds. We will multiply these by op_norm before
    # we leave this function.
    # comes from Theorem 4.13 in [1]
    lower_bound = k / min(dim)
    # our most basic upper bound
    upper_bound = 1  # pylint: disable=redefined-variable-type

    # break out of the function if the target value has already been met
    if __target_is_proved(lower_bound, upper_bound, op_norm, tol, target):
        return op_norm * lower_bound, op_norm * upper_bound

    # If input is not hermitian, we don't have better bounds, so we aboort.
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

    # if the exact answer won't be found by SDP, compute bounds via other methods first
    if not (is_positive and is_trans_exact and k == 1 and effort >= 1):
        # use the lower bound of Proposition 4.14 of [1]
        for r in range(k, min(dim) + 1):
            t_ind = np.prod(dim) - np.prod(dim - r) - 1
            lower_bound = max(lower_bound, (k / r) * eig_val[t_ind])

        # use the lower bound of Theorem 4.2.15 of [3]
        if k == 1:
            lower_bound = max(
                lower_bound,
                (
                    np.trace(mat)
                    + np.sqrt(
                        (prod_dim * np.trace(mat @ mat) - np.trace(mat) ** 2) / (prod_dim - 1)
                    )
                )
                / prod_dim,
            )

        if is_positive:
            # Use the upper bound of Proposition 15 of [1].
            upper_bound = min(
                upper_bound,
                sum(
                    abs(eig_val[i]) * sk_vector_norm(eig_vec[:, i], k, dim) ** 2
                    for i in range(prod_dim)
                ),
            )

            # Use the upper bound of Proposition 4.2.11 of [3].
            upper_bound = min(upper_bound, kp_norm(realignment(mat, dim), k ** 2, 2))

        # Use the lower bound of Theorem 4.2.17 of [3].
        if is_projection:
            lower_bound = max(
                lower_bound,
                min(
                    1,
                    k
                    / np.ceil(
                        (dim[0] + dim[1] - np.sqrt((dim[0] - dim[1]) ** 2 + 4 * rank - 4)) / 2
                    ),
                ),
            )

            lower_bound = max(
                lower_bound,
                (min(dim) - k)
                * (rank + np.sqrt((prod_dim * rank - rank ** 2) / (prod_dim - 1)))
                / (prod_dim * (min(dim) - 1))
                + (k - 1) / (min(dim) - 1),
            )

        # break out of the function if the target value has already been met
        if __target_is_proved(lower_bound, upper_bound, op_norm, tol, target):
            return op_norm * lower_bound, op_norm * upper_bound

        # Use a randomized iterative method to try to improve the lower bound.
        if is_positive:
            for _ in range(5 ** effort):
                lower_bound = max(
                    lower_bound,
                    __lower_bound_sk_norm_randomized(mat, k, dim, tol ** 2),
                )

                # break out of the function if the target value has already been met
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
            constraints.append(partial_transpose(rho, [1], dim) >> 0)
        else:
            constraints.append(k * cvxpy.kron(partial_trace(rho, [1], dim), np.eye(dim[1])) >> rho)

        problem = cvxpy.Problem(objective, constraints)
        cvx_optval = problem.solve()
        if problem.status != "optimal":
            raise ValueError("Numerical problems encountered.")

        upper_bound = min(upper_bound, np.real(cvx_optval))

        # In small dimensions, the transpose map gets the result exactly.
        if is_trans_exact and k == 1:
            lower_bound = upper_bound
        elif k == 1:
            # we can also get decent lower bounds from the SDP results when k=1
            # See Theorem 5.2.8 of [2]
            roots, _ = scipy.special.roots_jacobi(  # pylint: disable=unbalanced-tuple-unpacking
                1, dim[1] - 2, 1
            )
            gs = min(1 - roots)
            xmineig = min(eig_val)
            lower_bound = max(
                lower_bound,
                np.real(cvx_optval) * (1 - dim[1] * gs / (2 * dim[1] - 1))
                + xmineig * gs / (2 * dim[1] - 2),
            )

            # Done the effort = 1 SDP, now get better upper bounds via symmetric
            # extensions if effort >= 2.
            for j in range(2, effort + 1):
                # break out of the function if the target value has already been met
                if __target_is_proved(lower_bound, upper_bound, op_norm, tol, target):
                    return op_norm * lower_bound, op_norm * upper_bound

                sym_dim = [dim[0]] + [dim[1]] * j
                prod_sym_dim = dim[0] * (dim[1] ** j)
                sym_proj = np.kron(np.eye(dim[0]), symmetric_projection(dim[1], j))

                rho = cvxpy.Variable((prod_sym_dim, prod_sym_dim), hermitian=True, name="rho")
                objective = cvxpy.Maximize(
                    cvxpy.real(
                        cvxpy.trace(mat @ partial_trace(rho, list(range(2, j + 1)), sym_dim))
                    )
                )

                constraints = [
                    rho >> 0,
                    cvxpy.real(cvxpy.trace(rho)) <= 1,
                    sym_proj @ rho @ sym_proj == rho,
                    partial_transpose(rho, list(range(0, int(np.ceil(j / 2)) + 1)), sym_dim) >> 0,
                ]

                problem = cvxpy.Problem(objective, constraints)
                cvx_optval = problem.solve()
                if problem.status != "optimal":
                    raise ValueError("Numerical problems encountered.")

                upper_bound = min(upper_bound, np.real(cvx_optval))

                roots, _ = scipy.special.roots_jacobi(  # pylint: disable=unbalanced-tuple-unpacking
                    np.floor(j / 2) + 1, dim[1] - 2, j % 2
                )
                gs = min(1 - roots)
                lower_bound = max(
                    lower_bound,
                    np.real(cvx_optval) * (1 - dim[1] * gs / (2 * dim[1] - 1))
                    + xmineig * gs / (2 * dim[1] - 2),
                )

    lower_bound = op_norm * lower_bound
    upper_bound = op_norm * upper_bound

    return lower_bound, upper_bound


# This function checks whether or not the lower bound or upper bound
# already computed meets the desired target value (within numerical error)
# and thus we can abort early.
def __target_is_proved(
    lower_bound: float, upper_bound: float, op_norm: float, tol: float, target: float
) -> bool:
    return op_norm * (lower_bound + tol) >= op_norm * upper_bound or (
        target is not None
        and (op_norm * (lower_bound - tol) >= target or op_norm * (upper_bound + tol) <= target)
    )


# This function computes a lower bound of the S(k)-norm of the input matrix
# via a randomized method that searches for local maxima.
#
# In more detail, starting from a random vector of Schmidt-rank less than k,
# we alternately fix the Schmidt vectors of one sub-system and optimize the
# Schmidt vectors of the other sub-system. This optimization is equivalent to
# a generalized eigenvalue problem. The algorithm terminates when an iteration
# cannot improve the lower bound by more than tol.
def __lower_bound_sk_norm_randomized(  # pylint: disable=too-many-locals
    mat: np.ndarray,
    k: int = 1,
    dim: int | list[int] = None,
    tol: float = 1e-5,
    start_vec: np.ndarray = None,
) -> float:
    dim_a, dim_b = dim

    psi = max_entangled(k, is_normalized=False)
    left_swap_entagled_kron_id = swap(
        np.kron(psi, np.eye(dim_a * dim_b)), [2, 3], [k, k, dim_a, dim_b], row_only=True
    )

    swap_entagled_kron_id = left_swap_entagled_kron_id @ left_swap_entagled_kron_id.conj().T
    swap_entagled_kron_mat = swap(
        np.kron(psi @ psi.conj().T, mat), [2, 3], [k, k, dim_a, dim_b], row_only=False
    )

    opt_vec = None
    if start_vec is not None:
        singular_vals, vt_mat, u_mat = schmidt_decomposition(start_vec, dim)
        if (s_rank := len(singular_vals)) > k:
            warnings.warn(
                f"The Schmidt rank of the initial vector is {s_rank}, which is larger than k={k}. \
                Using a randomly-generated intial vector instead."
            )
        else:
            opt_vec = start_vec
            opt_schmidt = np.zeros((max(dim) * k, max(dim) * k))
            opt_schmidt[: k * dim_a, 0] = np.ravel(vt_mat @ np.diag(singular_vals), order="F")
            opt_schmidt[: k * dim_b, 1] = np.ravel(u_mat, order="F")

    if opt_vec is None:
        opt_schmidt = np.random.randn(max(dim) * k, 2) + 1j * np.random.randn(max(dim) * k, 2)
        opt_schmidt[k * dim_a :, 0] = 0
        opt_schmidt[k * dim_b :, 1] = 0
        opt_vec = left_swap_entagled_kron_id.conj().T @ np.kron(
            opt_schmidt[: k * dim_a, 0], opt_schmidt[: k * dim_b, 1]
        )
        opt_vec /= np.linalg.norm(opt_vec, ord=2)

    opt_schmidt /= np.linalg.norm(opt_schmidt, ord=2, axis=0)

    sk_lower_bound = np.real(opt_vec.conj().T @ mat @ opt_vec)
    it_lower_bound_improved = True

    while it_lower_bound_improved:
        it_lower_bound_improved = False

        # Loop through the 2 parties.
        for p in range(2):
            # If Schmidt rank is not full, we will have numerical problems; go to
            # lower Schmidt rank iteration.
            if (s_rank := schmidt_rank(opt_vec, dim)) < k:
                return __lower_bound_sk_norm_randomized(mat, s_rank, dim, tol, opt_vec)

            # Fix one of the parties and optimize over the other party.
            if p == 0:
                v0_mat = np.kron(opt_schmidt[: dim_a * k, 0].reshape((-1, 1)), np.eye(dim_b * k))
            else:
                v0_mat = np.kron(np.eye(dim_a * k), opt_schmidt[: dim_b * k, 1].reshape((-1, 1)))

            a_mat = v0_mat.conj().T @ swap_entagled_kron_mat @ v0_mat
            b_mat = v0_mat.conj().T @ swap_entagled_kron_id @ v0_mat

            largest_eigval, largest_eigvec = scipy.linalg.eigh(
                a_mat, b=b_mat, subset_by_index=[a_mat.shape[0] - 1, a_mat.shape[0] - 1]
            )

            if (new_sk_lower_bound := np.real(largest_eigval[0])) >= sk_lower_bound + tol:
                it_lower_bound_improved = True
                sk_lower_bound = new_sk_lower_bound

                opt_schmidt[: v0_mat.shape[1], (p + 1) % 2] = largest_eigvec.ravel()
                opt_vec = left_swap_entagled_kron_id.conj().T @ np.kron(
                    opt_schmidt[: k * dim_a, 0], opt_schmidt[: k * dim_b, 1]
                )

                opt_schmidt[:, (p + 1) % 2] /= np.linalg.norm(opt_schmidt[:, (p + 1) % 2], ord=2)
                opt_vec /= np.linalg.norm(opt_vec, ord=2)

    return sk_lower_bound
