"""Measured relative entropy quantifies how well two states can be distinguished by measuring individual copies."""

import cvxpy as cvx
import numpy as np

from toqito._quadrature import _gauss_legendre_01
from toqito.matrix_props import is_density, is_positive_semidefinite


def measured_relative_entropy(rho: np.ndarray, sigma: np.ndarray, eps: float = 1e-5) -> float:
    r"""Compute the measured relative entropy of two quantum states. [@huang2025semidefinite].

    Given a quantum state \(\rho\) and a positive semi-definite operator \(\sigma\),
    the measured relative entropy is defined by optimizing the relative entropy over all
    possible measurements:

    \[
        D^M(\rho \| \sigma) := \sup_{\mathcal{X}, (\Lambda_x)_{x \in \mathcal{X}}}
        \sum_{x \in \mathcal{X}} \operatorname{Tr}[\Lambda_x \rho] \ln \left(
        \frac{\operatorname{Tr}[\Lambda_x \rho]}{\operatorname{Tr}[\Lambda_x \sigma]} \right),
    \]

    where the supremum is over every finite alphabet \(\mathcal{X}\) and every
    positive-operator valued measure (POVM) \((\Lambda_x)_{x \in \mathcal{X}}\)
    (i.e., satisfying \(\Lambda_x \geq 0\) for all \(x \in \mathcal{X}\) and
    \(\sum_{x \in \mathcal{X}}\Lambda_x = I\)).

    When \(\rho\) and \(\sigma\) are \(d \times d\) matrices, the quantity
    \(D^M(\rho \| \sigma)\) can be efficiently calculated by means of a semi-definite
    program up to an additive error \(\varepsilon\), by means of
    \(O(\sqrt{\ln(1/\varepsilon)})\) linear matrix inequalities, each of size
    \(2d \times 2d\). Specifically, there exist \(m, k \in \mathbb{N}\) such that
    \(m+k = O(\sqrt{\ln(1/\varepsilon)})\) and the following inequality holds:

    \[
        |D^M(\rho \| \sigma) - D_{m,k}^M(\rho \| \sigma)| \leq \varepsilon,
    \]

    where

    \[
        D_{m,k}^M(\rho \| \sigma) :=
        \mathop{\sup}\limits_{\substack{
            \omega > 0,\; \theta \in \mathbb{H},\\
            T_1,\dots,T_m \in \mathbb{H},\\
            Z_0,\dots,Z_k \in \mathbb{H}}}
        \left\{
        \begin{array}{c}
        \operatorname{Tr}[\theta \rho] - \operatorname{Tr}[\omega \sigma] + 1 : \\[6pt]
        Z_0 = \omega, \qquad \sum_{j=1}^m w_j T_j = 2^{-k} \theta, \\[6pt]
        \left\{\begin{bmatrix}
        Z_i & Z_{i+1}\\
        Z_{i+1} & I
        \end{bmatrix}
        \ge 0 \right\}_{i=0}^{k-1}, \\[10pt]
        \left\{\begin{bmatrix}
        Z_k - I - T_j & -\sqrt{t_j}T_j \\
        -\sqrt{t_j}T_j & I - t_jT_j
        \end{bmatrix}
        \ge 0
        \right\}_{j=1}^{m} \end{array}
        \right\}
    \]


    and, for all \(j \in \{1, \dots, m\}\), \(w_j\) and \(t_j\)
    are the weights and nodes, respectively, for the \(m\)-point Gauss--Legendre quadrature
    on the interval \([0, 1]\).

    Args:
        rho: Density operator.
        sigma: Positive semi-definite operator.
        eps: Tolerance level.

    Returns:
        The measured relative entropy between `rho` and `sigma`.

    Raises:
        ValueError: If `rho` if not a density operator or if `sigma` is not positive semi-definite.
        ValueError: If the support of `rho` is not contained in the support of `sigma`, in which
            case the measured relative entropy is infinite.

    Examples:
        Consider the following quantum state \(\rho = \frac{1}{2}(I + r \cdot \mathbf{\sigma})\)
        and the PSD operator \(\sigma = \frac{1}{2}(I + s \cdot \mathbf{\sigma})\), where
        \(r = (0.9, 0.05, -0.02)\), \(s = (-0.8, 0.1, 0.1)\), and \(\mathbf{\sigma} =
        (\sigma_x, \sigma_y, \sigma_z)\) are the Pauli operators.

        Calculating the measured relative entropy can be done as follows.

        ```python exec="1" source="above" result="text"
        from toqito.matrices import pauli
        from toqito.state_metrics import measured_relative_entropy
        import numpy as np

        r = np.array([0.9, 0.05, -0.02])
        s = np.array([-0.8, 0.1, 0.1])
        rho = 0.5 * (pauli("I") + r[0] * pauli("X") + r[1] * pauli("Y") + r[2] * pauli("Z"))
        sigma = 0.5 * (pauli("I") + s[0] * pauli("X") + s[1] * pauli("Y") + s[2] * pauli("Z"))
        print(measured_relative_entropy(rho, sigma, 1e-5))
        ```

    """
    if not is_density(rho):
        raise ValueError("Measured relative entropy is only defined if rho is a density operator.")
    if not is_positive_semidefinite(sigma):
        raise ValueError("Measured relative entropy is only defined if sigma is positive semi-definite.")
    if np.array_equal(rho, sigma):
        return 0
    rho, sigma = _restrict_to_support(rho, sigma)
    n = len(rho)
    m, k = _find_mk(rho, sigma, eps)
    w, theta = cvx.Variable((n, n), complex=True), cvx.Variable((n, n), hermitian=True)
    ts = [cvx.Variable((n, n), hermitian=True) for _ in range(m)]
    zs = [cvx.Variable((n, n), hermitian=True) for _ in range(k + 1)]
    nodes, weights = _gauss_legendre_01(m)

    Id = cvx.Constant(np.eye(n))
    zblocks = [cvx.bmat(((zs[i], zs[i + 1]), (zs[i + 1], Id))) for i in range(k)]
    tblocks = [
        cvx.bmat(
            ((zs[k] - Id - ts[j], -np.sqrt(nodes[j]) * ts[j]), (-np.sqrt(nodes[j]) * ts[j], Id - nodes[j] * ts[j]))
        )
        for j in range(m)
    ]

    cons = (
        [zs[0] == w, w >> 0]
        + [(sum(weights[i] * ts[i] for i in range(m))) == 2 ** (-k) * theta]
        + [zblocks[i] >> 0 for i in range(k)]
        + [tblocks[j] >> 0 for j in range(m)]
    )

    rho = cvx.Constant(rho)
    sigma = cvx.Constant(sigma)
    obj = cvx.Maximize(cvx.real(cvx.trace(theta @ rho) - cvx.trace(w @ sigma) + 1))
    problem = cvx.Problem(obj, constraints=cons)
    # When rho is rank deficient the optimal omega has eigenvalues of order eps, and SCS with Anderson
    # acceleration can stall on the resulting flat direction and return an inaccurate solution. Plain
    # ADMM takes more iterations but converges on every such instance tried.
    problem.solve(solver=cvx.SCS, acceleration_lookback=0, verbose=False)
    return obj.value


def _restrict_to_support(rho: np.ndarray, sigma: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    r"""Compress both operators onto the support of `sigma`.

    A measurement outcome supported outside the support of \(\sigma\) has
    \(\operatorname{Tr}[\Lambda_x \sigma] = 0\), so it contributes nothing when
    \(\operatorname{Tr}[\Lambda_x \rho] = 0\) and makes the relative entropy infinite otherwise.
    Compressing onto the support therefore preserves the value exactly and leaves \(\sigma\)
    positive definite, which is what the choice of quadrature size below requires.
    """
    eig_rho, vec_rho = np.linalg.eigh(rho)
    eig_sigma, vec_sigma = np.linalg.eigh(sigma)
    kernel = eig_sigma <= tol
    if not kernel.any():
        return rho, sigma
    weights = eig_rho @ np.abs(vec_rho.conj().T @ vec_sigma) ** 2
    if np.any(weights[kernel] > tol):
        raise ValueError(
            "Measured relative entropy is infinite because the support of rho is not contained in the support of sigma."
        )
    basis = vec_sigma[:, ~kernel]
    return basis.conj().T @ rho @ basis, basis.conj().T @ sigma @ basis


def _find_mk(rho: np.ndarray, sigma: np.ndarray, eps: float) -> tuple[int, int]:
    r"""Find m and k for the desired epsilon value.

    The optimal \(\omega\) in the variational formula has its spectrum inside
    \([\lambda_{\min}, \lambda_{\max}]\) of \(\sigma^{-1/2} \rho \sigma^{-1/2}\), so the quadrature only
    has to be accurate on \([1/a, a]\) with \(a = \max(\lambda_{\max}, 1/\lambda_{\min})\). When
    \(\rho\) is singular on the support of \(\sigma\) the lower end is zero. Lifting the corresponding
    eigenvalues of \(\omega\) to \(1/a\) changes the objective by at most \(\operatorname{Tr}[\sigma]/a\),
    so taking \(a = 2\operatorname{Tr}[\sigma]/\varepsilon\) and leaving \(\varepsilon/2\) for the
    quadrature keeps the total error within \(\varepsilon\). The same bound is used whenever it gives
    the smaller \(a\).
    """
    eig_sigma, vec_sigma = np.linalg.eigh(sigma)
    sigma_half_inv = (vec_sigma / np.sqrt(eig_sigma)) @ vec_sigma.conj().T
    eigs = np.linalg.eigvalsh(sigma_half_inv @ rho @ sigma_half_inv)
    lower, upper = eigs.min(), eigs.max()
    trace_sigma = np.trace(sigma).real
    if lower > 0 and 1 / lower <= 2 * trace_sigma / eps:
        a = max(upper, 1 / lower)
    else:
        eps = eps / 2
        a = max(upper, trace_sigma / eps)
    k1 = int(np.ceil(np.log2(np.log(a))) + 1)
    k2 = int(2 * np.ceil(np.sqrt(np.log2(32 * np.log(a) / eps)) / 2))
    k = k1 + k2
    m = k2 // 2
    return m, k
