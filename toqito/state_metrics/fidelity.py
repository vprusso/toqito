"""Fidelity is a metric that qualifies how close two quantum states are."""

import cvxpy
import numpy as np
import scipy

from toqito.matrix_props import is_density


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""Compute the fidelity of two density matrices [@wikipediafidelity].

    Calculate the fidelity between the two density matrices `rho` and `sigma`, defined by:

    \[
        ||\sqrt(\rho) \sqrt(\sigma)||_1,
    \]

    where \(|| \cdot ||_1\) denotes the trace norm. The return is a value between \(0\) and \(1\), with
    \(0\) corresponding to matrices `rho` and `sigma` with orthogonal support, and \(1\)
    corresponding to the case `rho = sigma`.

    Examples:
        Consider the following Bell state

        \[
            u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.
        \]

        The corresponding density matrix of \(u\) may be calculated by:

        \[
            \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                             1 & 0 & 0 & 1 \\
                             0 & 0 & 0 & 0 \\
                             0 & 0 & 0 & 0 \\
                             1 & 0 & 0 & 1
                           \end{pmatrix} \in \text{D}(\mathcal{X}).
        \]

        In the event where we calculate the fidelity between states that are identical, we should obtain the value of
        \(1\). This can be observed in `|toqito⟩` as follows.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.state_metrics import fidelity

        rho = 1 / 2 * np.array(
            [[1, 0, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]]
        )
        sigma = rho

        print(fidelity(rho, sigma))
        ```

    Raises:
        ValueError: If matrices are not density operators.

    Args:
        rho: Density operator.
        sigma: Density operator.

    Returns:
        The fidelity between `rho` and `sigma`.

    """
    # Perform some error checking.
    if not np.all(rho.shape == sigma.shape):
        raise ValueError("InvalidDim: `rho` and `sigma` must be matrices of the same size.")

    # If `rho` or `sigma` is a cvxpy variable then compute fidelity via semidefinite programming, so that this function
    # can be used in the objective function or constraints of other cvxpy optimization problems.
    if isinstance(rho, cvxpy.atoms.affine.vstack.Vstack) or isinstance(sigma, cvxpy.atoms.affine.vstack.Vstack):
        z_var = cvxpy.Variable(rho.shape, complex=True)
        objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(z_var + z_var.H)))
        constraints = [cvxpy.bmat([[rho, z_var], [z_var.H, sigma]]) >> 0]
        problem = cvxpy.Problem(objective, constraints)

        return 1 / 2 * problem.solve()

    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Fidelity is only defined for density operators.")

    # If `rho` or `sigma` are *not* cvxpy variables, compute fidelity normally, since this is much faster.
    sq_rho = scipy.linalg.sqrtm(rho)
    sq_fid = scipy.linalg.sqrtm(sq_rho @ sigma @ sq_rho)
    return np.real(np.trace(sq_fid))
