"""Sinkhorn operator."""

import warnings

import numpy as np
import scipy.linalg

from toqito.channels import partial_trace


def operator_sinkhorn(
    rho: np.ndarray,
    dim: list[int] =None,
    tol: float =np.sqrt(np.finfo(float).eps)):
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

    >>> rho_random = random_density_matrix(4)

    returns a random 4x4 complex matrix like this

    >>> print("rho_random", rho_random, sep='\n')
   
    rho_random
    array([[0.34777941+0.j -0.03231138+0.07992951j 0.08131097+0.03285289j -0.07485986+0.07115859j]
           [-0.03231138-0.07992951j 0.26158857+0.j 0.04867659-0.13939625j 0.05986692+0.00369842j]
           [0.08131097-0.03285289j 0.04867659+0.13939625j 0.16827722+0.j 0.03699826+0.14622j]
           [-0.07485986-0.07115859j 0.05986692-0.00369842j 0.03699826-0.14622j 0.22235479+0.j]])

    >>> sigma, F = operator_sinkhorn(rho=rho_random, dim=[2, 2])

    This returns the result density matrix along with the operations list F.
    SIGMA has all of its (single-party) reduced density matrices
    proportional to the identity, and SIGMA = Tensor(F)*RHO*Tensor(F)'. In
    other words, F contains invertible local operations that demonstrate
    that RHO and SIGMA are locally equivalent.

    >>> print("sigma", sigma, sep='\n')

    sigma
    array([[0.30410752+0.j 0.02421406-0.0181278j -0.11612204-0.03620092j 0.21646849+0.05515334j]
           [0.02421406+0.0181278j 0.19589248+0.j -0.02561571+0.01174259j 0.11612205+0.03620092j]
           [-0.11612204+0.03620092j -0.02561571-0.01174259j  0.19589248+0.j -0.02421406+0.0181278j]
           [ 0.21646849-0.05515334j  0.11612205-0.03620092j -0.02421406-0.0181278j 0.30410752+0.j]])

    >>> print("F", F, sep='\n')

    F
    [array([[ 1.14288548-0.00662635j, -0.07948299-0.24568425j],[-0.07280878+0.20604394j,  1.02019328+0.00723707j]]),
     array([[ 0.92821148-0.00380203j, -0.45803016+0.01202661j],[-0.49498221-0.00491192j,  1.92337107+0.00388873j]])]

    Similarly to perform sinkhorn operation on multipartite state where the
    subsystems are of dimensions 2, 3 and 4.

    >>> rho_random = random_density_matrix(24)
    >>> sigma, F = operator_sinkhorn(rho=rho_random, dim=[2, 3, 4])
    >>> print(sigma, F)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param rho: input density matrix of a multipartite system
    :param dim: `None` list containing dimensions of each subsystem.
                Assumes 2 subsystems with equal dimensions as default.
    :param tol: `np.sqrt(np.finfo(float).eps)` Convergence tolerance of the iterative Sinkhorn Algorithm.
                Assumes square root of numpy eps as default.
    :raises: ValueError: if input density matrix is not a square matrix.
    :raises: ValueError: if the product of dimensions provided/assumed does not match the dimension of density matrix.
    :raises: RuntimeError: if the sinkhorn algorithm doesnot converge before 10000 iterations.
    :raises: ValueError: if the density matrix provided is singular (or is not of full rank).

    """
    # Run checks on the input density matrix
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Input 'rho' must be a square matrix.")

    rho = rho.astype(np.complex128)  # Ensure complex128 type for rho

    dX = rho.shape[0]
    sdX = int(round(np.sqrt(dX)))
    tr_rho = np.trace(rho)

    # set optional argument defaults: dim=sqrt(length(rho)), tol=sqrt(eps).
    if dim is None:
        dim = [sdX, sdX]

    # allow the user to enter a single number for dim.
    if len(dim) == 1:
        dim.append(dX / dim[0])

        if abs(dim[1] - round(dim[1])) >= (2 * dX * np.finfo(float).eps):
            raise ValueError("If DIM is a scalar, X must be square and DIM must evenly divide length(X); "
                             "please provide the DIM array containing the dimensions of the subsystems.")

        dim[1] = round(dim[1])

    num_sys = len(dim)
    # dimensions check before starting iterations
    if np.prod(dim) != dX:
        raise ValueError(f"Product of dimensions {dim} does not match rho dimension {dX}.")

    # precompute trace for all iterations
    tr_rho_p = np.power(tr_rho, 1.0 / (2.0 * num_sys)) if tr_rho != 0 else 1.0

    # Prepare the iteration.
    Prho = [np.eye(d, dtype=np.complex128) / d for d in dim]  # Force complex128 type for Prho
    F = [np.eye(d, dtype=np.complex128) * tr_rho_p for d in dim]  # Force complex128 for F
    ldim = [np.prod(dim[:j]) for j in range(num_sys)]
    rdim = [np.prod(dim[j+1:]) for j in range(num_sys)]

    # Perform the operator Sinkhorn iteration.
    it_err = 1.0
    max_cond = 0
    max_iterations = 10000
    iterations = 0
    rho2 = rho.copy().astype(np.complex128)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        while it_err > tol:

            if iterations > max_iterations:
                raise RuntimeError(f"operator_sinkhorn did not converge within {max_iterations} iterations.")

            it_err = 0.0
            max_cond = 0.0

            error_flag_in_iteration = False
            error_message = ""

            try:
                for j in range(num_sys):
                    # Compute the reduced density matrix on the j-th system.
                    Prho_tmp = partial_trace(rho2, list(set(range(num_sys)) - {j}), dim)
                    Prho_tmp = (Prho_tmp + Prho_tmp.conj().T) / 2.0  # for numerical stability
                    it_err += np.linalg.norm(Prho[j] - Prho_tmp)
                    Prho[j] = Prho_tmp.astype(np.complex128)  # Force complex128 for Prho_tmp

                    # Apply the filter with explicit complex128 conversion
                    try:
                        T_inv = np.linalg.inv(Prho[j]).astype(np.complex128)

                        # check for NaNs and infinities after inversion
                        if np.any(np.isnan(T_inv)) or np.any(np.isinf(T_inv)):
                                raise np.linalg.LinAlgError("Singular matrix encountered during inversion.")

                        T = scipy.linalg.sqrtm(T_inv) / np.sqrt(dim[j])
                        T = T.astype(np.complex128)

                        # enforce hermiticity for numerical stability
                        T = (T + T.conj().T) / 2.0

                    except np.linalg.LinAlgError as la_err:
                        error_flag_in_iteration = True
                        error_message = f"Matrix inversion/sqrt failed for subsystem {j}: {la_err}"
                        break

                    # Construct the Kronecker product
                    eye_ldim = np.eye(int(ldim[j]), dtype=np.complex128)
                    eye_rdim = np.eye(int(rdim[j]), dtype=np.complex128)
                    Tk = np.kron(eye_ldim, np.kron(T, eye_rdim))

                    rho2 = Tk @ rho2 @ Tk.conj().T
                    F[j] = T @ F[j]

                    try:
                        max_cond = max(max_cond, np.linalg.cond(F[j]))
                    except np.linalg.LinAlgError:
                        max_cond = np.inf # possible singular matrix

            except Exception as gen_err:
                error_flag_in_iteration = True

            # Check the condition number to ensure invertibility.
            if (error_flag_in_iteration) or (max_cond >= 1 / tol) or (np.isinf(max_cond)):
                raise ValueError("The operator Sinkhorn iteration does not converge for RHO. "
                             "This is often the case if RHO is not of full rank.")

            iterations += 1

    # Stabilize the output for numerical reasons.
    sigma = (rho2 + rho2.conj().T) / 2
    sigma = tr_rho * sigma / np.trace(sigma)

    # handle cases where trace(sigma) can be near zero: TODO

    return sigma, F
