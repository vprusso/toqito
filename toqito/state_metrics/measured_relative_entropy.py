"""Measured relative entropy quantifies how well two states can be distinguished by measuring individual copies."""

import cvxpy as cvx
import numpy as np
import scipy.linalg

from toqito.matrix_props import is_density, is_positive_semidefinite


def measured_relative_entropy(rho: np.ndarray, sigma: np.ndarray, eps: float = 1e-5) -> float:
    r"""Compute the measured relative entropy of two quantum states. :footcite:`Huang_2025_Msrd_Rel_Entr`.

    Given a quantum state :math:`\rho` and a positive semi-definite operator :math:`\sigma`,
    the measured relative entropy is defined by optimizing the relative entropy over all
    possible measurements:

    .. math::

        D^M(\rho \| \sigma) := \sup_{\mathcal{X}, (\Lambda_x)_{x \in \mathcal{X}}}
        \sum_{x \in \mathcal{X}} \operatorname{Tr}[\Lambda_x \rho] \ln \left(
        \frac{\operatorname{Tr}[\Lambda_x \rho]}{\operatorname{Tr}[\Lambda_x \sigma]} \right),

    where the supremum is over every finite alphabet :math:`\mathcal{X}` and every
    positive-operator valued measure (POVM) :math:`(\Lambda_x)_{x \in \mathcal{X}}`
    (i.e., satisfying :math:`\Lambda_x \geq 0` for all :math:`x \in \mathcal{X}` and
    :math:`\sum_{x \in \mathcal{X}}\Lambda_x = I`).

    When :math:`\rho` and :math:`\sigma` are :math:`d \times d` matrices, the quantity
    :math:`D^M(\rho \| \sigma)` can be efficiently calculated by means of a semi-definite
    program up to an additive error :math:`\varepsilon`, by means of
    :math:`O(\sqrt{\ln(1/\varepsilon)})` linear matrix inequalities, each of size
    :math:`2d \times 2d`. Specifically, there exist :math:`m, k \in \mathbb{N}` such that
    :math:`m+k = O(\sqrt{\ln(1/\varepsilon)})` and the following inequality holds:

    .. math::

        |D^M(\rho \| \sigma) - D_{m,k}^M(\rho \| \sigma)| \leq \varepsilon,

    where

    .. math::

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


    and, for all :math:`j \in \{1, \dots, m\}`, :math:`w_j` and :math:`t_j`
    are the weights and nodes, respectively, for the :math:`m`-point Gauss--Legendre quadrature
    on the interval :math:`[0, 1]`.

    Examples
    ==========

    Consider the following quantum state :math:`\rho = \frac{1}{2}(I + r \cdot \boldsymbol{\sigma})`
    and the PSD operator :math:`\sigma = \frac{1}{2}(I + s \cdot \boldsymbol{\sigma})`, where
    :math:`r = (0.9, 0.05, -0.02)`, :math:`s = (-0.8, 0.1, 0.1)`, and :math:`\boldsymbol{\sigma} =
    (\sigma_x, \sigma_y, \sigma_z)` are the Pauli operators.

    Calculating the measured relative entropy can be done as follows.

    .. jupyter-execute::

        from toqito.matrices import pauli
        from toqito.state_metrics import measured_relative_entropy
        import numpy as np

        r = np.array([0.9, 0.05, -0.02])
        s = np.array([-0.8, 0.1, 0.1])
        rho = 0.5 * (pauli("I") + r[0] * pauli("X") + r[1] * pauli("Y") + r[2] * pauli("Z"))
        sigma = 0.5 * (pauli("I") + s[0] * pauli("X") + s[1] * pauli("Y") + s[2] * pauli("Z"))
        measured_relative_entropy(rho, sigma, 1e-5)

    References
    ==========
    .. footbibliography::

    :param rho: Density operator.
    :param sigma: Positive semi-definite operator.
    :param err: Tolerance level.
    :return: The measured relative entropy between :math:`\rho` and :math:`\sigma`.

    """
    if not is_density(rho):
        raise ValueError("Measured relative entropy is only defined if rho is a density operator.")
    if not is_positive_semidefinite(sigma):
        raise ValueError("Measured relative entropy is only defined if sigma is positive semi-definite.")
    if np.array_equal(rho, sigma):
        return 0
    n = len(rho)
    m, k = _find_mk(rho, sigma, eps)
    w, theta = cvx.Variable((n, n), complex=True), cvx.Variable((n, n), hermitian=True)
    ts = [cvx.Variable((n, n), hermitian=True) for _ in range(m)]
    zs = [cvx.Variable((n, n), hermitian=True) for _ in range(k + 1)]
    nodes, weights = _gauss_legendre_on_01(m)

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
    problem.solve(verbose=False)
    return obj.value


def _gauss_legendre_on_01(m: int) -> (np.ndarray, np.ndarray):
    """m-point Gauss legendre quadrature weights on the interval [0,1]."""
    x = np.polynomial.legendre.leggauss(m)[0]
    w = np.polynomial.legendre.leggauss(m)[1]
    node = 0.5 * (x + 1)
    weight = 0.5 * w
    return node, weight


def _compute_a(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Find optimal a."""
    rho_half_inv = scipy.linalg.inv(scipy.linalg.sqrtm(rho))
    X = rho_half_inv @ sigma @ rho_half_inv
    eigs = np.linalg.eigvalsh(X)
    a = max(eigs.max(), 1.0 / eigs.min())
    return a


def _find_mk(rho: np.ndarray, sigma: np.ndarray, error: float) -> (int, int):
    """Find m and k for the desired error rate."""
    a = _compute_a(rho, sigma)
    k1 = int(np.ceil(np.log2(np.log(a))) + 1)
    k2 = int(2 * np.ceil(np.sqrt(np.log2(32 * np.log(a) / error)) / 2))
    k = k1 + k2
    m = k2 // 2
    return m, k
