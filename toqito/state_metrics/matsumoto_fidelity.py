"""Matsumoto fidelity is the maximum classical fidelity associated with a classical-to-quantum preparation procedure."""

import cvxpy
import numpy as np
import scipy

from toqito.matrix_props import is_density


def matsumoto_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float | np.floating:
    r"""Compute the Matsumoto fidelity of two density matrices [@Matsumoto_2010_Reverse].

    Calculate the Matsumoto fidelity between the two density matrices `rho` and `sigma`, defined by:

    \[
        \mathrm{tr}(\rho\#\sigma),
    \]

    where \(\#\) denotes the matrix geometric mean, which for invertible states is

    \[
        \rho\#\sigma = \rho^{1/2}\sqrt{\rho^{-1/2}\sigma\rho^{-1/2}}\rho^{1/2}.
    \]

    For singular states it is defined by the limit

    \[
        \rho\#\sigma = \lim_{\epsilon\to0}(\rho+\epsilon\mathbb{I})\#(+\epsilon\mathbb{I}).
    \]

    The return is a value between \(0\) and \(1\), with \(0\) corresponding to matrices `rho` and
    `sigma` with orthogonal support, and \(1\) corresponding to the case `rho = sigma`. The Matsumoto
    fidelity is a lower bound for the fidelity.

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

    In the event where we calculate the Matsumoto fidelity between states that are identical, we should obtain the value
    of \(1\). This can be observed in `|toqito⟩` as follows.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.state_metrics import matsumoto_fidelity

    rho = 1 / 2 * np.array(
        [[1, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]
    )
    sigma = rho

    print(np.around(matsumoto_fidelity(rho, sigma), decimals=2))
    ```

    Raises:
        ValueError: If matrices are not of equal dimension.

    Args:
        rho: Density operator.
        sigma: Density operator.

    Returns:
        The Matsumoto fidelity between `rho` and `sigma`.

    """
    if not np.all(rho.shape == sigma.shape):
        raise ValueError("InvalidDim: `rho` and `sigma` must be matrices of the same size.")

    # If `rho` or `sigma` is a cvxpy variable then compute Matsumoto fidelity via
    # semidefinite programming, so that this function can be used in the
    # objective function or constraints of other cvxpy optimization problems.
    if isinstance(rho, cvxpy.atoms.affine.vstack.Vstack) or isinstance(sigma, cvxpy.atoms.affine.vstack.Vstack):
        w_var = cvxpy.Variable(rho.shape, hermitian=True)
        objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(w_var)))
        constraints = [cvxpy.bmat([[rho, w_var], [w_var, sigma]]) >> 0]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()

    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Matsumoto fidelity is only defined for density operators.")

    # If `rho` or `sigma` are *not* cvxpy variables, compute Matsumoto fidelity directly.
    # For numerical stability, invert the matrix with larger determinant
    if np.abs(scipy.linalg.det(sigma)) > np.abs(scipy.linalg.det(rho)):
        rho, sigma = sigma, rho

    # If rho is singular, add epsilon
    try:
        sq_rho = scipy.linalg.sqrtm(rho)
        sqinv_rho = scipy.linalg.inv(sq_rho)
    except np.linalg.LinAlgError:
        sq_rho = scipy.linalg.sqrtm(rho + 1e-7)  # if rho is not invertible, add epsilon=1e-7 to it
        # note if epsilon=1e-8 or smaller, it leads to test failures.
        sqinv_rho = scipy.linalg.inv(sq_rho)

    sq_mfid = sq_rho @ scipy.linalg.sqrtm(sqinv_rho @ sigma @ sqinv_rho) @ sq_rho
    return np.real(np.trace(sq_mfid))
