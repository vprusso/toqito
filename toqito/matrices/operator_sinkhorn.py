"""Sinkhorn operator."""

import numpy as np
import scipy.linalg

from toqito.channels import partial_trace
# from toqito.rand import random_density_matrix

import warnings


def operator_sinkhorn(
    rho: np.ndarray, 
    dim: list[int] =None, 
    tol: float =np.sqrt(np.finfo(float).eps)):
    
    """Perform the operator Sinkhorn iteration.
    This function implements the Sinkhorn algorithm to find a density matrix that
    is locally equivalent to a given bipartite or multipartite density matrix
    `rho`, but has both of its partial traces proportional to the identity.
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
        
    num_sys = len(dim);
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
                error_message = f"Unexpected error during iteration: {gen_err}"

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
