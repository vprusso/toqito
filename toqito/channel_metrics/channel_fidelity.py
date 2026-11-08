"""Computes the channel fidelity between two quantum channels."""

import warnings

import cvxpy
import numpy as np

from toqito.matrix_ops import partial_trace


def channel_fidelity(choi_1: np.ndarray, choi_2: np.ndarray, eps: float = 1e-7) -> float:
    r"""Compute the channel fidelity between two quantum channels [@katariya2021geometric].

    Let \(\Phi : \text{L}(\mathcal{Y}) \rightarrow \text{L}(\mathcal{X})\) and
    \(\Psi: \text{L}(\mathcal{Y}) \rightarrow \text{L}(\mathcal{X})\) be quantum channels. Then
    the root channel fidelity defined as

    \[
        \sqrt{F}(\Phi, \Psi) := \text{inf}_{\rho} \sqrt{F}(\Phi(\rho), \Psi(\rho))
    \]

    where \(\rho \in \text{D}(\mathcal{Z} \otimes \mathcal{X})\) can be calculated by means of
    the following semidefinite program (Proposition 50) in [@katariya2021geometric],

    \[
        \begin{align*}
            \text{maximize:} \quad & \lambda \\
            \text{subject to:} \quad & \lambda \mathbb{I}_{\mathcal{Z}} \leq
                \text{Re}\left( \text{Tr}_{\mathcal{Y}} \left( Q \right) \right),\\
                & \begin{pmatrix}
                    J(\Phi) & Q^* \\
                    Q & J(\Psi)
                \end{pmatrix} \geq 0
        \end{align*}
    \]

    where \(Q \in \text{L}(\mathcal{Z} \otimes \mathcal{X})\) and \(\text{Re}(X) = (X + X^*)/2\) is the Hermitian
    part of an operator, so the first constraint is an operator inequality.

    The block constraint confines the column space of \(Q\) to the support of \(J(\Psi)\) and its row space to
    the support of \(J(\Phi)\). The program is therefore solved for \(Q\) on those supports, which keeps it strictly
    feasible when either Choi matrix is singular and makes the solve fast and accurate.

    Args:
        choi_1: The Choi matrix of the first quantum channel.
        choi_2: The Choi matrix of the second quantum channel.
        eps: The solver tolerance for convergence to feasibility.

    Returns:
        The channel fidelity between the channels specified by the quantum channels corresponding to the Choi matrices
        `choi_1` and `choi_2`.

    Raises:
        ValueError: If matrices are not of equal dimension.
        ValueError: If matrices are not square.

    Examples:
        For two identical channels, we should expect that the channel fidelity should yield a value of
        \(1\).

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import dephasing
        from toqito.channel_metrics import channel_fidelity
        # The Choi matrices of dimension-4 for the dephasing channel
        choi_1 = dephasing(4)
        choi_2 = dephasing(4)
        print(channel_fidelity(choi_1, choi_2))
        ```

        We can also compute the channel fidelity between two different channels. For example, we can
        compute the channel fidelity between the dephasing and depolarizing channels.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import dephasing, depolarizing
        from toqito.channel_metrics import channel_fidelity
        # The Choi matrices of dimension-4 for the dephasing and depolarizing channels
        choi_1 = dephasing(4)
        choi_2 = depolarizing(4)
        print(channel_fidelity(choi_1, choi_2))
        ```

    """
    if choi_1.shape != choi_2.shape:
        raise ValueError("The Choi matrices provided should be of equal dimension.")

    choi_dim_x, choi_dim_y = choi_1.shape
    if choi_dim_x != choi_dim_y:
        raise ValueError("The Choi matrix provided must be square.")

    choi_dim = choi_dim_x
    # The Choi matrix of a channel with equal input/output dimension d is d^2 by d^2, so the subsystem dimension is
    # sqrt(choi_dim). The previous int(log2(choi_dim)) only coincided with this for d <= 4 and was wrong for larger d.
    dim = int(round(np.sqrt(choi_dim)))
    if dim * dim != choi_dim:
        raise ValueError("The Choi matrix dimension must be a perfect square (equal input and output dimensions).")

    basis_1, spectrum_1 = _support(choi_1)
    basis_2, spectrum_2 = _support(choi_2)

    lam = cvxpy.Variable(nonneg=True)
    q_small = cvxpy.Variable((basis_2.shape[1], basis_1.shape[1]), complex=True)
    q_var = basis_2 @ q_small @ basis_1.conj().T
    traced = partial_trace(q_var, [1], [dim, dim])

    constraints = [
        cvxpy.bmat([[np.diag(spectrum_1), q_small.H], [q_small, np.diag(spectrum_2)]]) >> 0,
        (traced + traced.H) / 2 >> lam * np.identity(dim),
    ]
    problem = cvxpy.Problem(cvxpy.Maximize(lam), constraints)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Solution may be inaccurate")
        return problem.solve(solver=cvxpy.SCS, eps=eps)


def _support(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal basis of the support of a positive semidefinite matrix and the eigenvalues on it."""
    eigenvalues, eigenvectors = np.linalg.eigh((mat + mat.conj().T) / 2)
    keep = eigenvalues > 1e-10 * max(1.0, float(eigenvalues[-1]))
    return eigenvectors[:, keep], eigenvalues[keep]
