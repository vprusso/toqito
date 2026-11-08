"""Measured relative entropy (channel) is how well two channels can be distinguished by measuring them individually."""

import warnings

import cvxpy as cvx
import numpy as np

from toqito._quadrature import _gauss_legendre_01
from toqito._sdp import solve_sdp
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
    r"""Compute the measured relative entropy of two quantum channels [@huang2025semidefinite].

    Given a quantum channel $\mathcal{N}_{A \to B}$, a completely positive map $\mathcal{M}_{A \to B}$,
    a Hamiltonian $H_A$ (Hermitian operator acting on system $A$), and an energy constraint
    $E \in \mathbb{R}$, the energy-constrained measured relative entropy of channels is defined as:

    $$
        D^{M}_{H,E}(\mathcal{N}\Vert\mathcal{M}) :=
        \sup_{\substack{d_{R'} \in \mathbb{N},\\ \rho_{R'A} \in \mathcal{D}(\mathcal{H}_{R'A})}}
        \left\{D^{M}\!\left(\mathcal{N}_{A \to B}(\rho_{R'A}) \middle\Vert \mathcal{M}_{A \to B}(\rho_{R'A})\right):
        \operatorname{Tr}[H_A \rho_A] \le E\right\}.
    $$

    When their Choi operators $\Gamma^{\mathcal{N}}$ and $\Gamma^{\mathcal{M}}$ are
    $d_A d_B \times d_A d_B$ matrices, the quantity $D^{M}_{H,E}(\mathcal{N}\Vert\mathcal{M})$ can be
    efficiently calculated by means of a semi-definite program up to an additive error $\varepsilon$,
    by means of $O(\sqrt{\ln(1/\varepsilon)})$ linear matrix inequalities, each of size
    $2d_A d_B \times 2d_A d_B$. Specifically, there exist $m, k \in \mathbb{N}$ such that
    $m + k = O(\sqrt{\ln(1/\varepsilon)})$ and the following inequality holds:

    $$
        \left|D^{M}_{H,E}(\mathcal{N}\Vert\mathcal{M})
        - D^{M}_{H,E,m,k}(\mathcal{N}\Vert\mathcal{M})\right| \le \varepsilon,
    $$

    where

    $$
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
    $$

    and, for all $j \in \{1, \dots, m\}$, $w_j$ and $t_j$
    are the weights and nodes, respectively, for the $m$-point Gauss--Legendre quadrature
    on the interval $[0, 1]$.

    Args:
        channel_1: Choi matrix for the first channel.
        channel_2: Choi matrix for the second channel.
        in_dim: The dimension of the input of the quantum channels.
        m: One of the optimization parameters.
        k: The other optimization parameter.
        hamiltonian: The Hamiltonian.
        energy: The energy constraint.

    Returns:
        The measured relative entropy between `channel_1` and `channel_2`.

    Raises:
        ValueError: If `channel_1` is not a quantum channel or `channel_2` is not completely positive.
        ValueError: If the semidefinite program does not solve.

    Examples:
        The completely depolarizing channel against the depolarizing channel with parameter \(1/4\), with no
        energy constraint. Both outputs commute for the maximally entangled input, where the measured and the
        Umegaki relative entropies agree at about \(0.0759\) nats.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_metrics import channel_measured_relative_entropy
        from toqito.channels import depolarizing

        channel_1 = np.eye(4) / 2
        channel_2 = depolarizing(2, 0.25)
        hamiltonian = np.zeros((2, 2))
        print(round(channel_measured_relative_entropy(channel_1, channel_2, 2, 6, 6, hamiltonian, np.inf), 4))
        ```

    """
    if not is_quantum_channel(channel_1, dim=[in_dim, len(channel_1) // in_dim]):
        raise ValueError("Measured relative entropy is only defined if channel_1 is a quantum channel.")
    if not is_completely_positive(channel_2):
        raise ValueError("Measured relative entropy is only defined if channel_2 is a completely positive map.")
    if np.array_equal(channel_1, channel_2):
        return 0
    n = len(channel_1)
    out_dim = len(channel_1) // in_dim
    omega = cvx.Variable((n, n), hermitian=True)
    rho = cvx.Variable((in_dim, in_dim), hermitian=True)
    theta = cvx.Variable((n, n), hermitian=True)
    ts = [cvx.Variable((n, n), hermitian=True) for _ in range(m)]
    zs = [cvx.Variable((n, n), hermitian=True) for _ in range(k + 1)]
    nodes, weights = _gauss_legendre_01(m)

    rho_id = cvx.kron(rho, np.eye(out_dim))
    zblocks = [cvx.bmat(([zs[i], zs[i + 1]], [zs[i + 1], rho_id])) for i in range(k)]
    tblocks = [
        cvx.bmat(
            (
                [zs[k] - rho_id - ts[j], -np.sqrt(nodes[j]) * ts[j]],
                [-np.sqrt(nodes[j]) * ts[j], rho_id - nodes[j] * ts[j]],
            )
        )
        for j in range(m)
    ]

    # The quadrature constraint carries the 2^k factor on the T side, as in `measured_relative_entropy`: a
    # residual of size eps in sum_j w_j T_j = 2^{-k} theta would be amplified by 2^k in theta, which enters the
    # objective directly.
    cons = (
        [cvx.trace(rho) == 1]
        + [zs[0] == omega]
        + [rho >> 0, omega >> 0]
        + [2**k * sum(weights[i] * ts[i] for i in range(m)) == theta]
        + [zblocks[i] >> 0 for i in range(k)]
        + [tblocks[j] >> 0 for j in range(m)]
    )
    # Tr[H rho] <= lambda_max(H) for every state, so the energy constraint only binds below that. Leaving it out
    # otherwise matters numerically: a constraint such as 0 <= 1e6 puts a constant of that size into the solver
    # data and SCS then stalls at a value an order of magnitude off.
    hamiltonian = (np.asarray(hamiltonian) + np.asarray(hamiltonian).conj().T) / 2
    if energy < np.linalg.eigvalsh(hamiltonian)[-1]:
        cons.append(cvx.real(cvx.trace(hamiltonian @ rho)) <= energy)

    obj = cvx.Maximize(cvx.real(cvx.trace(theta @ channel_1) - cvx.trace(omega @ channel_2) + 1))
    problem = cvx.Problem(obj, constraints=cons)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Solution may be inaccurate")
        return solve_sdp(problem, cvx.SCS, {"eps": 1e-8})
