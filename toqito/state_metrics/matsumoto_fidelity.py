"""Matsumoto fidelity metric."""
import cvxpy
import scipy
import numpy as np

from toqito.matrix_props import is_density


def matsumoto_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""TODO: docstring"""

    if not np.all(rho.shape == sigma.shape):
        raise ValueError("InvalidDim: `rho` and `sigma` must be matrices of the same size.")

    # If `rho` or `sigma` is a cvxpy variable then compute fidelity via
    # semidefinite programming, so that this function can be used in the
    # objective function or constraints of other cvxpy optimization problems.
    if isinstance(rho, cvxpy.atoms.affine.vstack.Vstack) or isinstance(
        sigma, cvxpy.atoms.affine.vstack.Vstack
    ):
        w_var = cvxpy.Variable(rho.shape, hermitian=True)
        objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(w_var)))
        constraints = [cvxpy.bmat([[rho, w_var], [w_var, sigma]]) >> 0]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()

    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Fidelity is only defined for density operators.")

    # If `rho` or `sigma` are *not* cvxpy variables, compute Matsumoto fidelity directly.
    # For numerical stability, invert the matrix with larger determinant
    if np.abs(scipy.linalg.det(sigma)) > np.abs(scipy.linalg.det(rho)):
        rho, sigma = sigma, rho

    # If rho is singular, add epsilon
    try:
        sq_rho = scipy.linalg.sqrtm(rho)
        sqinv_rho = scipy.linalg.inv(sq_rho)
    except np.linalg.LinAlgError:
        sq_rho = scipy.linalg.sqrtm(rho+1e-8) # if rho is not invertible, add epsilon=1e-8 to it
        sqinv_rho = scipy.linalg.inv(sq_rho)

    sq_mfid = sq_rho @ scipy.linalg.sqrtm(sqinv_rho @ sigma @ sqinv_rho) @ sq_rho
    return np.real(np.trace(sq_mfid))
