"""Sinkhorn operator."""

import numpy as np
import scipy as sp

from toqito.channels.partial_trace import partial_trace
from toqito.matrix_props.is_square import is_square


def operator_sinkhorn(
    rho: np.ndarray,
    dim: list[int] = None,
    tol: float = np.sqrt(np.finfo(float).eps),
    max_iterations: int = 10000) -> tuple[np.ndarray, list[np.ndarray]]:
    r"""Perform the operator Sinkhorn iteration.

    This function implements the iterative Sinkhorn algorithm to find a density matrix
    that is locally equivalent to a given bipartite or multipartite density matrix
    `rho`, but has both of its partial traces proportional to the identity.

    Examples
    ==========

    To Perform Operator Sinkhorn on a random 2-Qubit Bi-partite state:

    >>> import numpy as np
    >>> from toqito.matrices import operator_sinkhorn
    >>> from toqito.rand import random_density_matrix

    >>> rho_random = random_density_matrix(4, seed=42)

    returns a random 4x4 complex matrix like this

    >>> print("rho_random", rho_random, sep='\n')
    rho_random
    [[0.23434155+0.j         0.20535572-0.04701708j 0.11523158-0.02017518j
      0.18524981-0.13636277j]
     [0.20535572+0.04701708j 0.2544268 +0.j         0.14478708-0.02004061j
      0.22496295-0.11837381j]
     [0.11523158+0.02017518j 0.14478708+0.02004061j 0.11824612+0.j
      0.09088248-0.05551508j]
     [0.18524981+0.13636277j 0.22496295+0.11837381j 0.09088248+0.05551508j
      0.39298553+0.j        ]]

    >>> sigma, F = operator_sinkhorn(rho=rho_random, dim=[2, 2])

    This returns the result density matrix along with the operations list F.
    SIGMA has all of its (single-party) reduced density matrices
    proportional to the identity, and SIGMA = Tensor(F)*RHO*Tensor(F)'. In
    other words, F contains invertible local operations that demonstrate
    that RHO and SIGMA are locally equivalent.

    >>> print("sigma", sigma, sep='\n')
    sigma
    [[ 0.34784186+0.j          0.09278034+0.00551919j -0.02275152+0.05104798j
       0.20443565-0.12511156j]
     [ 0.09278034-0.00551919j  0.15215814+0.j          0.1039554 -0.0350973j
       0.02275152-0.05104798j]
     [-0.02275152-0.05104798j  0.1039554 +0.0350973j   0.15215814+0.j
      -0.09278034-0.00551919j]
     [ 0.20443565+0.12511156j  0.02275152+0.05104798j -0.09278034+0.00551919j
       0.34784186+0.j        ]]

    >>> print("F", F, sep='\n')
    F
    [array([[ 1.33132101+0.00313432j, -0.38927801+0.14985847j],
           [-0.37199021-0.14296461j,  1.24635929-0.00300407j]]), array([[ 1.16317427+0.00765735j, -0.16925696+0.09054309j],
           [-0.18974418-0.09028502j,  0.93359381-0.00777829j]])]

    Similarly to perform sinkhorn operation on multipartite state where the
    subsystems are of dimensions 2, 3 and 4.

    >>> # rho_random = random_density_matrix(24)
    >>> # sigma, F = operator_sinkhorn(rho=rho_random, dim=[2, 3, 4])
    >>> # print(sigma, F)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param rho: input density matrix of a multipartite system
    :param dim: `None` list containing dimensions of each subsystem.
                Assumes 2 subsystems with equal dimensions as default.
    :param tol: `np.sqrt(np.finfo(float).eps)` Convergence tolerance of the iterative Sinkhorn Algorithm.
                Assumes square root of numpy eps as default.
    :param max_iterations: Number of iterations after which the solver terminates with a convergence error (Refer RuntimeError).
    :raises: ValueError: if input density matrix is not a square matrix.
    :raises: ValueError: if the product of dimensions provided/assumed does not match the dimension of density matrix.
    :raises: RuntimeError: if the sinkhorn algorithm doesnot converge before `max_iterations` iterations.
    :raises: ValueError: if the density matrix provided is singular (or is not of full rank).

    """ # noqa: E501
    # Run checks on the input density matrix
    rho = np.asarray(rho)
    if not is_square(rho):
        raise ValueError("Input 'rho' must be a square matrix.")

    rho = rho.astype(np.complex128)

    dX = rho.shape[0]
    sdX = int(round(np.sqrt(dX)))

    # set optional argument defaults: dim=sqrt(length(rho)), tol=sqrt(eps).
    if dim is None:
        dim = [sdX, sdX]

    # allow the user to enter a single number for dim.
    if len(dim) == 1:
        dim.append(dX / dim[0])

        if abs(dim[1] - round(dim[1])) >= (2 * dX * np.finfo(float).eps):
            raise ValueError("If dim is of size 1, rho must be square and dim[0] must evenly divide length(rho); "
                             "please provide the dim array containing the dimensions of the subsystems.")

        dim[1] = round(dim[1])

    num_sys = len(dim)
    # dimensions check before starting iterations
    if np.prod(dim) != dX:
        raise ValueError(f"Product of dimensions {dim} does not match rho dimension {dX}.")

    # precompute trace for all iterations
    tr_rho_p = np.power(np.trace(rho), 1.0 / (2.0 * num_sys)) if np.trace(rho) != 0 else 1.0

    # Prepare the iteration.
    Prho = [np.eye(d, dtype=np.complex128) / d for d in dim]
    F = [np.eye(d, dtype=np.complex128) * tr_rho_p for d in dim]
    ldim = [np.prod(dim[:j]) for j in range(num_sys)]
    rdim = [np.prod(dim[j+1:]) for j in range(num_sys)]

    # Perform the operator Sinkhorn iteration.
    it_err = 1.0
    max_cond = 0
    iterations = 0
    rho2 = rho.copy().astype(np.complex128)

    while it_err > tol:

        if iterations > max_iterations:
            raise RuntimeError(f"operator_sinkhorn did not converge within {max_iterations} iterations.")

        it_err = 0.0
        max_cond = 0.0

        error_flag_in_iteration = False

        try:
            for j in range(num_sys):
                # Compute the reduced density matrix on the j-th system.
                Prho_tmp = partial_trace(rho2, list(set(range(num_sys)) - {j}), dim)
                Prho_tmp = (Prho_tmp + Prho_tmp.conj().T) / 2.0  # for numerical stability
                it_err += np.linalg.norm(Prho[j] - Prho_tmp)
                Prho[j] = Prho_tmp.astype(np.complex128)

                # Apply filter with explicit complex128
                T_inv = np.linalg.inv(Prho[j]).astype(np.complex128)

                # check for NaNs and infinities after inversion
                if np.any(np.isnan(T_inv)) or np.any(np.isinf(T_inv)):
                    error_flag_in_iteration = True

                T = sp.linalg.sqrtm(T_inv) / np.sqrt(dim[j])
                T = T.astype(np.complex128)

                # enforce hermiticity for numerical stability
                T = (T + T.conj().T) / 2.0

                eye_ldim = np.eye(int(ldim[j]), dtype=np.complex128)
                eye_rdim = np.eye(int(rdim[j]), dtype=np.complex128)
                Tk = np.kron(eye_ldim, np.kron(T, eye_rdim))

                rho2 = Tk @ rho2 @ Tk.conj().T
                F[j] = T @ F[j]

                max_cond = max(max_cond, np.linalg.cond(F[j]))

        except Exception:
            error_flag_in_iteration = True

        # Check the condition number to ensure invertibility.
        if (error_flag_in_iteration) or (max_cond >= 1 / tol) or (np.isinf(max_cond)):
            raise ValueError("The operator Sinkhorn iteration does not converge for RHO. "
                             "This is often the case if RHO is not of full rank.")

        iterations += 1

    # Stabilize the output for numerical reasons.
    sigma = (rho2 + rho2.conj().T) / 2
    sigma = np.trace(rho) * sigma / np.trace(sigma)

    # handle cases where trace(sigma) can be near zero: TODO

    return sigma, F
