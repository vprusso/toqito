"""Sinkhorn operator."""

import warnings

import numpy as np
import scipy as sp

from toqito.channels.partial_trace import partial_trace
from toqito.matrix_props.is_square import is_square


def operator_sinkhorn(
    rho: np.ndarray, dim: list[int] = None, tol: float = np.sqrt(np.finfo(float).eps), max_iterations: int = 10_000
) -> tuple[np.ndarray, list[np.ndarray]]:
    r"""Perform operator Sinkhorn iteration (iterative sinkhorn algorithm) on a quantum system's density matrix.

    This function is adapted from QETLAB. :cite:`QETLAB_link`.

    This function relies on Sinkhorn's theorem :cite:`Sinkhorn_1964_Relationship` which states "for any
    positive-definite square matrix, there exist diagonal matrices :math:`D_1` and :math:`D_2` such that
    :math:`D_1 \, \cdot A \, \cdot D_2` is doubly stochastic.

    The iterative Sinkhorn algorithm alternately rescales the input density matrix of the quantum system along each
    subsystem so that its marginals become uniform (i.e., partial traces or row/column sums become proportional to
    identity), until convergence.

    Upon convergence we end up with a density matrix `sigma` (:math:`\sigma`), which is locally equivalent to the given
    multipartite density matrix `rho` (:math:`\rho`), but having maximally mixed subsystems. The algorithm will also
    return the list of Filtering Operations which can be directly applied on :math:`\rho` to arrive at :math:`\sigma`.
    Such converted forms of density matrices are useful to analyse entangled systems and to study effects of operations
    on each subsystems. (as discussed in :cite:`Gurvits_2004_Classical`)

    Examples
    ==========

    To obtain a locally normalized density matrix of a random 2-qubit bipartite state using operator Sinkhorn:

    ..jupyter-execute::
        import numpy as np
        from toqito.matrices import operator_sinkhorn
        from toqito.rand import random_density_matrix

        rho_random = random_density_matrix(4, seed=42)

        print(rho_random)

    :code:`operator_sinkhorn` returns the result density matrix along with the operations list :code:`local_ops`.
    :code:`sigma` (:math:`\sigma`) has all of its (single-party) reduced density matrices proportional
    to the identity, while satisfying

    .. math::
        \sigma \, = \, F \, \cdot \, \rho \, \cdot \, F^{\dagger}

    In other words, :code:`local_ops` contains invertible local operations that demonstrate that :code:`rho` and
    :code:`sigma` are locally equivalent. This can be checked by first obtaining the :math:`F` matrix using the
    elements of :code:`local_ops`

    for this example, :math:`F_1` = :code:`local_ops[0]`, and :math:`F_2` = :code:`local_ops[1]`
    and :math:`F` matrix is then
    .. math::
        F = \left( F_1 \otimes F_2 \right)

    and in general,
    .. math::
        F = \left( F_1 \otimes F_2 \otimes F_3 \otimes ... F_n \right)

    ..jupyter-execute::
        sigma, local_ops = operator_sinkhorn(rho=rho_random, dim=[2, 2])
        print(sigma)
        print(len(local_ops))

    :code:`local_ops` here will be the list of 2 local filtering operators, first of which is

    ..jupyter-execute::
        print(local_ops[0])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param rho: Input density matrix of a multipartite system
    :param dim: List containing dimensions of each subsystem.
                :code:`None` is passed by default which assumes 2 subsystems with equal dimensions.
    :param tol: `np.sqrt(np.finfo(float).eps)` Convergence tolerance of the iterative Sinkhorn Algorithm.
                Assumes square root of numpy eps as default.
    :param max_iterations: Number of iterations after which the solver terminates with a convergence error.
    :raises: ValueError: if input density matrix is not a square matrix.
    :raises: ValueError: if the product of dimensions provided/assumed does not match the dimension of density matrix.
    :raises: ValueError: if the density matrix provided is singular (or is not of full rank).
    :raises: RuntimeError: if the Sinkhorn algorithm does not converge before :code:`max_iterations` iterations.
    :return: A tuple of 2 items :code:`(sigma, local_ops)` where,
        - :code:`sigma` is the locally normalized form of the input density matrix :code:`rho`.
          (:code:`numpy.ndarray` of shape equal to :code:`rho`)
        - :code:`local_ops` is the list of invertible local operators which can obtain :code:`sigma` from :code:`rho`.

    """
    # Run checks on the input density matrix.
    rho = np.asarray(rho)
    if not is_square(rho):
        raise ValueError("Input 'rho' must be a square matrix.")

    rho = rho.astype(np.complex128)

    dX = rho.shape[0]
    sdX = int(round(np.sqrt(dX)))

    # Set optional argument defaults: dim=sqrt(len(rho)), tol=sqrt(eps).
    if dim is None:
        dim = [sdX, sdX]

    # Allow the user to enter a single number for dim.
    if len(dim) == 1:
        dim.append(dX / dim[0])

        if abs(dim[1] - round(dim[1])) >= (2 * dX * np.finfo(float).eps):
            raise ValueError(
                "If dim is of size 1, rho must be square and dim[0] must evenly divide rho.shape[0]; "
                "please provide the dim array containing the dimensions of the subsystems."
            )

        dim[1] = round(dim[1])

    num_sys = len(dim)
    # Dimensions check before starting iterations.
    if np.prod(dim) != dX:
        raise ValueError(f"Product of dimensions {dim} does not match rho dimension {dX}.")

    # Precompute trace for all iterations.
    tr_rho_p = np.power(np.trace(rho), 1.0 / (2.0 * num_sys)) if np.trace(rho) != 0 else 1.0

    # Prepare the iteration.
    Prho = [np.eye(d, dtype=np.complex128) / d for d in dim]
    local_ops = [np.eye(d, dtype=np.complex128) * tr_rho_p for d in dim]
    ldim = [np.prod(dim[:j]) for j in range(num_sys)]
    rdim = [np.prod(dim[j + 1 :]) for j in range(num_sys)]

    # Perform the operator Sinkhorn iteration.
    it_err = 1.0
    max_cond = 0
    iterations = 0
    rho2 = rho.copy().astype(np.complex128)

    while it_err > tol:
        if iterations > max_iterations:
            raise RuntimeError(f"operator_sinkhorn did not converge within {max_iterations} iterations.")

        it_err, max_cond, error_flag_in_iteration = 0.0, 0.0, False

        try:
            for j in range(num_sys):
                # Compute the reduced density matrix on the j-th system.
                Prho_tmp = partial_trace(rho2, list(set(range(num_sys)) - {j}), dim)
                # For numerical stability.
                Prho_tmp = (Prho_tmp + Prho_tmp.conj().T) / 2.0
                it_err += np.linalg.norm(Prho[j] - Prho_tmp)
                Prho[j] = Prho_tmp.astype(np.complex128)

                # Apply filter with explicit complex128.
                T_inv = np.linalg.inv(Prho[j]).astype(np.complex128)

                T = sp.linalg.sqrtm(T_inv) / np.sqrt(dim[j])
                T = T.astype(np.complex128)

                # Enforce hermiticity for numerical stability.
                T = (T + T.conj().T) / 2.0

                eye_ldim = np.eye(int(ldim[j]), dtype=np.complex128)
                eye_rdim = np.eye(int(rdim[j]), dtype=np.complex128)
                Tk = np.kron(eye_ldim, np.kron(T, eye_rdim))

                rho2 = Tk @ rho2 @ Tk.conj().T
                local_ops[j] = T @ local_ops[j]

                max_cond = max(max_cond, np.linalg.cond(local_ops[j]))

        except Exception:
            error_flag_in_iteration = True

        # Check the condition number to ensure invertibility.
        if (error_flag_in_iteration) or (max_cond >= 1 / tol) or (np.isinf(max_cond)):
            raise ValueError(
                "The operator Sinkhorn iteration does not converge for rho. "
                "This is often the case if rho is not of full rank."
            )

        iterations += 1

    # Stabilize the output for numerical stability.
    sigma = (rho2 + rho2.conj().T) / 2
    # Handle cases where trace(sigma) can be near zero:
    # This can happen
    # 1. if input trace is near zero (singular matrix) - Algorithm will not converge.
    # 2. if there was some numerical instability and result drifted off - Warn users about
    # about a potentially wrong answer.
    sigma = np.trace(rho) * sigma / np.trace(sigma)
    if np.trace(sigma) < 10 * np.finfo(float).eps:
        warnings.warn("Final trace is near zero, but initial trace was not. Result may be unreliable.", RuntimeWarning)

    return sigma, local_ops
