"""Computes the fidelity of two density matrices."""
import cvxpy
import scipy
import numpy as np


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the fidelity of two density matrices.

    Calculate the fidelity between the two density matrices `rho` and `sigma`,
    defined by:

    :math: `||\sqrt(\rho) * \sqrt(\sigma)||_1`

    where :math: `|| . ||_1` denotes the trace norm. The return is a value
    between 0 and 1, with 0 corresponding to matrices `rho` and `sigma` with
    orthogonal support, and 1 corresponding to the case `rho = sigma`.

    References:
        [1] Wikipedia: Fidelity of quantum states
        https://en.wikipedia.org/wiki/Fidelity_of_quantum_states

    :param rho: Density matrix.
    :param sigma: Density matrix.
    :return: The fidelity between `rho` and `sigma`.
    """
    # Perform some error checking.
    if not np.all(rho.shape == sigma.shape):
        raise ValueError(
            "InvalidDim: `rho` and `sigma` must be matrices of the" " same size."
        )
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("InvalidDim: `rho` and `sigma` must be square.")

    # If `rho` or `sigma` is a cvxpy variable then compute fidelity via
    # semidefinite programming, so that this function can be used in the
    # objective function or constraints of other cvxpy optimization problems.
    if isinstance(rho, cvxpy.atoms.affine.vstack.Vstack) or isinstance(
        sigma, cvxpy.atoms.affine.vstack.Vstack
    ):
        z_var = cvxpy.Variable(rho.shape, complex=True)
        objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(z_var + z_var.H)))
        constraints = [cvxpy.bmat([[rho, z_var], [z_var.H, sigma]]) >> 0]
        problem = cvxpy.Problem(objective, constraints)

        return 1 / 2 * problem.solve()

    # If `rho` or `sigma` are *not* cvxpy variables, compute fidelity normally,
    # since this is much faster.
    sq_rho = scipy.linalg.sqrtm(rho)
    sq_fid = scipy.linalg.sqrtm(np.matmul(np.matmul(sq_rho, sigma), sq_rho))
    return np.real(np.trace(sq_fid))
