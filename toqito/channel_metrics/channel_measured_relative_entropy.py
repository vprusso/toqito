"""Measured relative entropy (channel) is how well two channels can be distinguished by measuring them individually."""

import cvxpy as cvx
import numpy as np

from toqito.channel_props import is_completely_positive, is_quantum_channel


def channel_measured_relative_entropy(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    in_dim: int,
    m: int,
    k: int,
    hamiltonian: np.ndarray,
    energy: float,
) -> float:
    r"""Compute the measured relative entropy of two quantum channels :footcite:`Huang_2025_Msrd_Rel_Entr`.

    Given a quantum channel :math:`\mathcal{N}_{A \to B}`, a completely positive map :math:`\mathcal{M}_{A \to B}`,
    a Hamiltonian :math:`H_A` (Hermitian operator acting on system :math:`A`), and an energy constraint
    :math:`E \in \mathbb{R}`, the energy-constrained measured relative entropy of channels is defined as

    .. math::

        D^{M}_{H,E}(\mathcal{N}\Vert\mathcal{M}) :=
        \sup_{\substack{d_{R'} \in \mathbb{N},\\ \rho_{R'A} \in \mathcal{D}(\mathcal{H}_{R'A})}}
        \left\{D^{M}\!\left(\mathcal{N}_{A \to B}(\rho_{R'A}) \middle\Vert \mathcal{M}_{A \to B}(\rho_{R'A})\right):
        \operatorname{Tr}[H_A \rho_A] \le E\right\}.

    When their Choi operators :math:`\Gamma^{\mathcal{N}}` and :math:`\Gamma^{\mathcal{M}}` are
    :math:`d_A d_B \times d_A d_B` matrices, the quantity :math:`D^{M}_{H,E}(\mathcal{N}\Vert\mathcal{M})` can be
    efficiently calculated by means of a semi-definite program up to an additive error :math:`\varepsilon`,
    by means of :math:`O(\sqrt{\ln(1/\varepsilon)})` linear matrix inequalities, each of size
    :math:`2d_A d_B \times 2d_A d_B`. Specifically, there exist :math:`m, k \in \mathbb{N}` such that
    :math:`m + k = O(\sqrt{\ln(1/\varepsilon)})` and the following inequality holds:

    .. math::

        \left|D^{M}_{H,E}(\mathcal{N}\Vert\mathcal{M})
        - D^{M}_{H,E,m,k}(\mathcal{N}\Vert\mathcal{M})\right| \le \varepsilon,

    where

    .. math::

        D_{H,E,m,k}^{M}(\mathcal{N} \| \mathcal{M}) :=
        \sup\limits_{\substack{\Omega, \rho > 0, \Theta \in \mathbb{H}, \\
        T_1, \dots, T_m \in \mathbb{H}, \\ Z_0, \dots, Z_k \in \mathbb{H}}}
        \left\{ \begin{gathered}
        \operatorname{Tr}[\Theta \Gamma^{\mathcal{N}}]
        - \operatorname{Tr}[\Omega \Gamma^{\mathcal{M}}] + 1 : \\
        \operatorname{Tr}[\rho] = 1, \operatorname{Tr}[H\rho] \leq E, \\
        Z_0 = \Omega, \sum_{j=1}^m w_j T_j = 2^{-k} \Theta, \\
        \left\{ \begin{bmatrix} Z_i & Z_{i+1} \\ Z_{i+1} & \rho \otimes I \end{bmatrix}
        \geq 0 \right\}_{i=0}^{k-1}, \\
        \left\{ \begin{bmatrix} Z_k - \rho \otimes I - T_j & -\sqrt{t_j} T_j \\
        -\sqrt{t_j} T_j & \rho \otimes I - t_j T_j \end{bmatrix} \geq 0 \right\}_{j=1}^m
        \end{gathered} \right\}

    and, for all :math:`j \in \{1, \dots, m\}`, :math:`w_j` and :math:`t_j`
    are the weights and nodes, respectively, for the :math:`m`-point Gauss--Legendre quadrature
    on the interval :math:`[0, 1]`.

    Examples
    ==========
    We can find the measured relative entropy between a depolarizing channel of dimension 2
    and the identity channel, constrained by a Hamiltonian and energy, as follows:

    .. jupyter-execute::

        from toqito.channel_metrics import channel_measured_relative_entropy
        from toqito.channels import depolarizing
        import numpy as np

        channel_1 = depolarizing(2, 0.2)
        channel_2 = np.eye(4)
        in_dim = 2
        m = 5
        k = 5
        hamiltonian = np.zeros((2, 2))
        energy = 100
        channel_measured_relative_entropy(channel_1, channel_2, in_dim, m, k, hamiltonian, energy)

    References
    ==========
    .. footbibliography::

    :raises ValueError: If :code:`channel_1` is not a quantum channel/:code:`channel_2` is not completely positive.
    :param channel_1: Choi matrix for first channel.
    :param channel_2: Choi matrix for second channel.
    :param in_dim: the dimension of the input of the quantum channels.
    :param m: one of the optimization parameters.
    :param k: the other optimization parameter.
    :param hamiltonian: the Hamiltonian.
    :param energy: the energy constraint.
    :return: The measured relative entropy between :code:`channel_1` and :code:`channel_2`.

    """
    if not is_quantum_channel(channel_1):
        raise ValueError("Measured relative entropy is only defined if channel_1 is a quantum channel.")
    if not is_completely_positive(channel_2):
        raise ValueError("Measured relative entropy is only defined if channel_2 is a completely positive map.")
    if np.array_equal(channel_1, channel_2):
        return 0
    n = len(channel_1)
    out_dim = len(channel_1) // in_dim
    omega = cvx.Variable((n, n), complex=True)
    rho = cvx.Variable((in_dim, in_dim), complex=True)
    theta = cvx.Variable((n, n), hermitian=True)
    ts = [cvx.Variable((n, n), hermitian=True) for _ in range(m)]
    zs = [cvx.Variable((n, n), hermitian=True) for _ in range(k + 1)]
    nodes, weights = _gauss_legendre_on_01(m)

    Id = cvx.Constant(np.eye(out_dim))
    zblocks = [cvx.bmat(([zs[i], zs[i + 1]], [zs[i + 1], cvx.kron(rho, Id)])) for i in range(k)]
    tblocks = [
        cvx.bmat(
            (
                [zs[k] - cvx.kron(rho, Id) - ts[j], -np.sqrt(nodes[j]) * ts[j]],
                [-np.sqrt(nodes[j]) * ts[j], cvx.kron(rho, Id) - nodes[j] * ts[j]],
            )
        )
        for j in range(m)
    ]

    cons = (
        [cvx.trace(rho) == 1]
        + [zs[0] == omega]
        + [cvx.real(cvx.trace(hamiltonian @ rho)) <= energy]
        + [rho >> 0, omega >> 0]
        + [sum([weights[i] * ts[i] for i in range(m)]) == 2 ** (-k) * theta]
        + [zblocks[i] >> 0 for i in range(k)]
        + [tblocks[j] >> 0 for j in range(m)]
    )

    channel_1_cvx = cvx.Constant(channel_1)
    channel_2_cvx = cvx.Constant(channel_2)
    obj = cvx.Maximize(cvx.real(cvx.trace(theta @ channel_1_cvx) - cvx.trace(omega @ channel_2_cvx) + 1))
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
