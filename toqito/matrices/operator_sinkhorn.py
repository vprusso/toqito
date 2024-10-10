"""Sinkhorn operator."""

import numpy as np
import scipy.linalg

from toqito import channels


def operator_sinkhorn(rho, dim=None, tol=np.sqrt(np.finfo(float).eps)):
    """Perform the operator Sinkhorn iteration.

    This function implements the Sinkhorn algorithm to find a density matrix that
    is locally equivalent to a given bipartite or multipartite density matrix
    `rho`, but has both of its partial traces proportional to the identity.
    """
    rho = rho.astype(np.complex128)  # Ensure complex128 type for rho

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
    Prho = [np.eye(d, dtype=np.complex128) / d for d in dim]  # Force complex128 type for Prho
    F = [np.eye(d, dtype=np.complex128) * tr_rho_p for d in dim]  # Force complex128 for F
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
                Prho_tmp = channels.partial_trace(rho, list(set(range(num_sys)) - {j}), dim)
                Prho_tmp = (Prho_tmp + Prho_tmp.T) / 2  # for numerical stability
                it_err += np.linalg.norm(Prho[j] - Prho_tmp)
                Prho[j] = Prho_tmp.astype(np.complex128)  # Force complex128 for Prho_tmp

                # Apply the filter with explicit complex128 conversion
                T_inv = np.linalg.inv(Prho[j]).astype(np.complex128)
                T = scipy.linalg.sqrtm(T_inv) / np.sqrt(dim[j])
                T = T.astype(np.complex128)

                # Construct the Kronecker product
                eye_ldim = np.eye(int(ldim[j]), dtype=np.complex128)
                eye_rdim = np.eye(int(rdim[j]), dtype=np.complex128)
                Tk = np.kron(eye_ldim, np.kron(T, eye_rdim))

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
