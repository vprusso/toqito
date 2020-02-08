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

    :param rho: Density matrix.
    :param sigma: Density matrix.
    :return: The fidelity between `rho` and `sigma`.
    """

    # Perform some error checking.
    if not np.all(rho.shape == sigma.shape):
        msg = """
            InvalidDim: `rho` and `sigma` must be matrices of the same size.
        """
        raise ValueError(msg)
    if rho.shape[0] != rho.shape[1]:
        msg = """
            InvalidDim: `rho` and `sigma` must be square.
        """
        raise ValueError(msg)

    if isinstance(rho, cvxpy.expressions.expression.Expression) or \
            isinstance(sigma, cvxpy.expressions.expression.Expression):
        pass

    # If `rho` or `sigma` are *not* cvxpy variables, compute fidelity normally,
    # since this is much faster.
    sq_rho = scipy.linalg.sqrtm(rho)
    sq_fid = scipy.linalg.sqrtm(np.matmul(np.matmul(sq_rho, sigma), sq_rho))
    return np.real(np.trace(sq_fid))
