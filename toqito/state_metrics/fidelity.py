"""Fidelity metric."""
import cvxpy
import scipy
import numpy as np

from toqito.matrix_props import is_density


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the fidelity of two density matrices [WikFid]_.

    Calculate the fidelity between the two density matrices :code:`rho` and :code:`sigma`, defined
    by:

    .. math::
        ||\sqrt(\rho) * \sqrt(\sigma)||_1,

    where :math:`|| \cdot ||_1` denotes the trace norm. The return is a value between :math:`0` and
    :math:`1`, with :math:`0` corresponding to matrices :code:`rho` and :code:`sigma` with
    orthogonal support, and :math:`1` corresponding to the case :code:`rho = sigma`.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    In the event where we calculate the fidelity between states that are identical, we should obtain
    the value of :math:`1`. This can be observed in :code:`toqito` as follows.

    >>> from toqito.state_metrics import fidelity
    >>> import numpy as np
    >>> rho = 1 / 2 * np.array(
    >>>     [[1, 0, 0, 1],
    >>>      [0, 0, 0, 0],
    >>>      [0, 0, 0, 0],
    >>>      [1, 0, 0, 1]]
    >>> )
    >>> sigma = rho
    >>> fidelity(rho, sigma)
    1.0000000000000002

    References
    ==========
    .. [WikFid] Wikipedia: Fidelity of quantum states
        https://en.wikipedia.org/wiki/Fidelity_of_quantum_states

    :param rho: Density operator.
    :param sigma: Density operator.
    :return: The fidelity between :code:`rho` and :code:`sigma`.
    """
    # Perform some error checking.
    if not np.all(rho.shape == sigma.shape):
        raise ValueError("InvalidDim: `rho` and `sigma` must be matrices of the same size.")

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

    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Fidelity is only defined for density operators.")

    # If `rho` or `sigma` are *not* cvxpy variables, compute fidelity normally,
    # since this is much faster.
    sq_rho = scipy.linalg.sqrtm(rho)
    sq_fid = scipy.linalg.sqrtm(sq_rho @ sigma @ sq_rho)
    return np.real(np.trace(sq_fid))
