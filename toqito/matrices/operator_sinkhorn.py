"""Sinkhorn operator."""

import numpy as np
import scipy.linalg

from toqito.channels import partial_trace


def operator_sinkhorn(rho, dim=None, tol=np.sqrt(np.finfo(float).eps)):
    """Perform the operator Sinkhorn iteration.

    This function implements the Sinkhorn algorithm to find a density matrix that
    is locally equivalent to a given bipartite or multipartite density matrix
    `rho`, but has both of its partial traces proportional to the identity.

    Examples
    =========
    Verify the partial traces of a randomly generated density matrix using the Sinkhorn operator iteration.

    >>> from toqito.rand import random_density_matrix
    >>> rho = random_density_matrix(9)
    >>> sigma, F = operator_sinkhorn(rho)
    >>> np.around(partial_trace(sigma, 0), decimals=2)
    array([[0.33, 0.  , 0.  ],
        [0.  , 0.33, 0.  ],
        [0.  , 0.  , 0.33]])

    Perform operator Sinkhorn iteration on a density matrix acting on 3-qubit space.

    >>> rho = random_density_matrix(8)
    >>> sigma, F = operator_sinkhorn(rho, [2, 2, 2])
    >>> np.around(partial_trace(sigma, [1, 2], [2, 2, 2]), decimals=2)
    array([[0.5, 0. ],
        [0. , 0.5]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises: ValueError: If dimensions inconsistent or `rho` is low-rank and the algorithm does not converge.
    :param rho:  The input density matrix, which must be square and of full rank.
    :param dim: A list of dimensions of the subsystems on which `rho` acts.
                Defaults to two subsystems of equal dimension.
    :param tol: The numerical tolerance used for convergence. Defaults to `sqrt(eps)`.
    :return: A tuple containing:
            - `sigma`: The locally equivalent density matrix with partial traces proportional to the identity.
            - `F`: A list of local operators demonstrating local equivalence between `rho` and `sigma`.

    """
    dX = len(rho)
    sdX = round(np.sqrt(dX))
    tr_rho = np.trace(rho)

    # set optional argument defaults: dim=sqrt(length(rho)), tol=sqrt(eps).
    if dim is None:
        dim = [sdX, sdX]
    num_sys = len(dim)

    # allow the user to enter a single number for dim.
    if num_sys == 1:
        dim.append(dX // dim[0])
        if abs(dim[1] - round(dim[1])) >= 2 * dX * np.finfo(float).eps:
            raise ValueError("If DIM is a scalar, X must be square and DIM must evenly divide length(X); "
                             "please provide the DIM array containing the dimensions of the subsystems.")
        dim[1] = round(dim[1])
        num_sys = 2

    tr_rho_p = tr_rho ** (1 / (2 * num_sys))

    # Prepare the iteration.
    Prho = [np.eye(d) / d for d in dim]
    F = [np.eye(d) * tr_rho_p for d in dim]
    ldim = [np.prod(dim[:j]) for j in range(num_sys)]
    rdim = [np.prod(dim[j+1:]) for j in range(num_sys)]

    # Perform the operator Sinkhorn iteration.
    it_err = 1
    max_cond = 0

    while it_err > tol:
        it_err = 0
        max_cond = 0

        try:
            for j in range(num_sys):
                # Compute the reduced density matrix on the j-th system.
                Prho_tmp = partial_trace(rho, list(set(range(num_sys)) - {j}), dim)
                Prho_tmp = (Prho_tmp + Prho_tmp.T) / 2  # for numerical stability
                it_err += np.linalg.norm(Prho[j] - Prho_tmp)
                Prho[j] = Prho_tmp

                # Apply the filter.
                T = scipy.linalg.sqrtm(np.linalg.inv(Prho[j])) / np.sqrt(dim[j])
                Tk = np.kron(np.eye(int(ldim[j])), np.kron(T, np.eye(int(rdim[j]))))

                rho = Tk @ rho @ Tk.T
                F[j] = T @ F[j]

                max_cond = max(max_cond, np.linalg.cond(F[j]))

        except np.linalg.LinAlgError:
            it_err = 1

        # Check the condition number to ensure invertibility.
        if it_err == 1 or max_cond >= 1 / tol:
            raise ValueError("The operator Sinkhorn iteration does not converge for RHO. "
                             "This is often the case if RHO is not of full rank.")

    # Stabilize the output for numerical reasons.
    sigma = (rho + rho.T) / 2
    sigma = tr_rho * sigma / np.trace(sigma)

    return sigma, F
