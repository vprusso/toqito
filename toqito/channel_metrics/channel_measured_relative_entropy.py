"""Measured relative entropy (channel) is how well two channels can be distinguished by measuring them individually."""

import cvxpy as cvx
import numpy as np

from toqito.channel_props import is_completely_positive, is_quantum_channel


def channel_measured_relative_entropy(N, M, dA, m, k, H, E):
    r"""Compute the measured relative entropy of two quantum channels. :footcite:`Huang_2025_Msrd_Rel_Entr`.

    Given a quantum channel :math:`\mathcal{N}_{A \to B}`, a completely positive map :math:`\mathcal{M}_{A \to B}`,
    a Hamiltonian :math:`H_A` (Hermitian operator acting on system :math:`A`), and an energy constraint
    :math:`E \in \mathbb{R}`, the energy-constrained measured relative entropy of channels is defined as

    .. math::
        D^{M}_{H,E}(\mathcal{N}\Vert\mathcal{M}) :=
        \sup_{\substack{d_{R'} \in \mathbb{N},\\ \rho_{R'A} \in \mathcal{D}(\mathcal{H}_{R'A})}}
        \left\{D^{M}\!\left(\mathcal{N}_{A \to B}(\rho_{R'A}) \middle\Vert \mathcal{M}_{A \to B}(\rho_{R'A})\right):
        \operatorname{Tr}[H_A \rho_A] \le E\right\}.

    When their Choi operators :math:`\Gamma^{\mathcal{N}}` and :math:`\Gamma^{\mathcal{M}}` are :math:
    `d_A d_B \times d_A d_B` matrices, the quantity :math:`D^{M}_{H,E}(\mathcal{N}\Vert\mathcal{M})` can be
    efficiently calculated by means of a semi-definite program up to an additive error :math:`\varepsilon`,
    by means of :math:`O(\sqrt{\ln(1/\varepsilon)})` linear matrix inequalities, each of size :math:
    `2d_A d_B \times 2d_A d_B`. Specifically, there exist :math:`m, k \in \mathbb{N}` such that :math:
    `m + k = O(\sqrt{\ln(1/\varepsilon)})` and the following inequality holds:

    ..math::

        \left|D^{M}_{H,E}(\mathcal{N}\Vert\mathcal{M})
        - D^{M}_{H,E,m,k}(\mathcal{N}\Vert\mathcal{M})\right| \le \varepsilon,

    where

    ..math::

        D^{M}_{H,E,m,k}(\mathcal{N}\Vert\mathcal{M}) :=
        \sup_{\substack{
        \Omega,\,\rho > 0,\,\Theta \in \mathbb{H},\\
        T_1,\ldots,T_m \in \mathbb{H},\\
        Z_0,\ldots,Z_k \in \mathbb{H}
        }}
        \left\{
        \operatorname{Tr}[\Theta \Gamma^{\mathcal{N}}]
        -
        \operatorname{Tr}[\Omega \Gamma^{\mathcal{M}}]
        + 1
        :
        \operatorname{Tr}[\rho] = 1,\;
        \operatorname{Tr}[H\rho] \le E,\;
        Z_0 = \Omega,\;
        \sum_{j=1}^m w_j T_j = 2^{-k} \Theta,
        \begin{array}{l}
        \left[\begin{array}{cc}
        Z_i & Z_{i+1} \\
        Z_{i+1} & \rho \otimes I
        \end{array}\right] \ge 0,\quad i=0,\ldots,k-1,\\[1em]
        \left[\begin{array}{cc}
        Z_k - \rho \otimes I - T_j & -\sqrt{t_j} T_j \\
        -\sqrt{t_j} T_j & \rho \otimes I - t_j T_j
        \end{array}\right] \ge 0,\quad j = 1,\ldots, m
        \end{array}
        \right\}.

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

    N = depolarizing(2, 0.2)
    M = np.eye(4)
    dA = 2
    m = 5
    k = 5
    H = np.zeros((2, 2))
    E = 100
    channel_measured_relative_entropy(N, M, dA, m, k, H, E)

    References
    ==========
    .. footbibliography::

    :param N: Choi matrix for first channel.
    :param M: Choi matrix for second channel.
    :param dA: the dimension of the input of the quantum channels.
    :param m: one of the optimization parameters.
    :param k: the other optimization parameter.
    :param H: the Hamiltonian.
    :param E: the energy constraint.
    :return: The measured relative entropy between :math:`\mathcal{N}_{A \to B}` and :math:`\mathcal{M}_{A \to B}`.

    """
    if not is_quantum_channel(N):
        raise ValueError("Measured relative entropy is only defined if N is a quantum channel.")
    if not is_completely_positive(M):
        raise ValueError("Measured relative entropy is only defined if M is a completely positive map.")
    if np.array_equal(N, M):
        return 0
    n = len(N)
    dB = len(N) // dA
    Omega = cvx.Variable((n, n), hermitian=True)
    rho = cvx.Variable((dA, dA), hermitian=True)
    Theta = cvx.Variable((n, n), hermitian=True)
    Ts = [cvx.Variable((n, n), hermitian=True) for i in range(m)]
    Zs = [cvx.Variable((n, n), hermitian=True) for i in range(k + 1)]
    ts, ws = gauss_legendre_on_01(m)

    Id = cvx.Constant(np.eye(dB))
    Zblocks = [cvx.bmat(([Zs[i], Zs[i + 1]], [Zs[i + 1], cvx.kron(rho, Id)])) for i in range(k)]
    Tblocks = [
        cvx.bmat(
            (
                [Zs[k] - cvx.kron(rho, Id) - Ts[j], -np.sqrt(ts[j]) * Ts[j]],
                [-np.sqrt(ts[j]) * Ts[j], cvx.kron(rho, Id) - ts[j] * Ts[j]],
            )
        )
        for j in range(m)
    ]

    cons = (
        [cvx.trace(rho) == 1]
        + [Zs[0] == Omega]
        + [cvx.real(cvx.trace(H @ rho)) <= E]
        + [rho >> 0, Omega >> 0, Theta >> 0]
        + [sum([ws[i] * Ts[i] for i in range(m)]) == 2 ** (-k) * Theta]
        + [Zblocks[i] >> 0 for i in range(k)]
        + [Tblocks[j] >> 0 for j in range(m)]
    )

    Ncvx = cvx.Constant(N)
    Mcvx = cvx.Constant(M)
    obj = cvx.Maximize(cvx.real(cvx.trace(Theta @ Ncvx) - cvx.trace(Omega @ Mcvx) + 1))
    problem = cvx.Problem(obj, constraints=cons)
    problem.solve(verbose=False)
    return obj.value


def gauss_legendre_on_01(m):
    """m-point Gauss legendre quadrature weights on the interval [0,1]."""
    x = np.polynomial.legendre.leggauss(m)[0]
    w = np.polynomial.legendre.leggauss(m)[1]
    T = 0.5 * (x + 1)
    W = 0.5 * w
    return T, W
