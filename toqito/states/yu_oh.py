"""Yu-Oh family of nonlocal bound entangled states."""

import numpy as np

from toqito.states import basis, max_entangled


def yu_oh(dim: int, x_param: float, y_param: float) -> np.ndarray:
    r"""Produce a Yu-Oh nonlocal bound entangled state [@yu2017family].

    The Yu-Oh states form a two-parameter family of PPT entangled states on
    \(\mathbb{C}^d \otimes \mathbb{C}^d\), \(d \geq 3\), that violate a Bell inequality. The construction
    uses the antisymmetric vectors \(|\psi_{ij}\rangle = |ij\rangle - |ji\rangle\) and a family of \(d\)
    unit vectors \(\theta_1, \dots, \theta_d \in \mathbb{R}^d\) that sum to zero and are built by induction
    on the dimension. For \(k = 1, \dots, d - 1\) let

    \[
        |\varphi_k\rangle = \frac{(d-1)^{3/2}}{d\sqrt{d-2}} \sum_{p=1}^{d} \theta_p(k) \,
        |\theta_p\rangle \otimes |\theta_p\rangle,
        \qquad
        |\psi_k\rangle = x |0, k\rangle + y |k, 0\rangle + z |\varphi_k\rangle,
    \]

    with \(z = \sqrt{1 - x^2 - y^2}\). With \(|\Phi_d\rangle = \sum_i |ii\rangle\) the unnormalized
    maximally entangled vector, \(\delta = z^2/(d-2) - xy\) and \(R = dxy + (d-1)(d-2)\delta + d - 1\),
    the state is

    \[
        \rho = \frac{1}{R} \left( xy \, |\Phi_d\rangle\langle\Phi_d|
        + \delta \sum_{1 \leq j < i < d} |\psi_{ij}\rangle\langle\psi_{ij}|
        + \sum_{k=1}^{d-1} |\psi_k\rangle\langle\psi_k| \right).
    \]

    The parameters must satisfy \(x^2 + y^2 \leq 1\) and \(\delta > 0\).

    Args:
        dim: The local dimension \(d \geq 3\).
        x_param: The parameter \(x\).
        y_param: The parameter \(y\).

    Returns:
        The Yu-Oh state on \(\mathbb{C}^d \otimes \mathbb{C}^d\).

    Raises:
        ValueError: If `dim` is smaller than 3, if \(x^2 + y^2 > 1\), or if \(\delta \leq 0\).

    Examples:
        The two-qutrit member with \(x = 0.5\) and \(y = 0.1\) is PPT.

        ```python exec="1" source="above" result="text"
        from toqito.states import yu_oh
        from toqito.state_props import is_ppt
        rho = yu_oh(3, 0.5, 0.1)
        print(rho.shape, is_ppt(rho, dim=[3, 3]))
        ```

    """
    if dim < 3:
        raise ValueError("The Yu-Oh state requires local dimension at least 3.")
    if x_param**2 + y_param**2 > 1:
        raise ValueError("The parameters must satisfy x^2 + y^2 <= 1.")
    z_param = np.sqrt(1 - x_param**2 - y_param**2)
    delta = z_param**2 / (dim - 2) - x_param * y_param
    if delta <= 0:
        raise ValueError("The parameters must satisfy z^2 / (d - 2) - x y > 0.")

    # The vectors theta_p, built by induction on the dimension, are unit vectors summing to zero.
    thetas = [basis(dim, 1).flatten(), -basis(dim, 1).flatten()]
    for sub_dim in range(3, dim + 1):
        last = basis(dim, sub_dim - 1).flatten()
        thetas = [(np.sqrt(sub_dim * (sub_dim - 2)) * theta - last) / (sub_dim - 1) for theta in thetas] + [last]

    norm = dim * x_param * y_param + (dim - 1) * (dim - 2) * delta + dim - 1
    phi_d = max_entangled(dim).flatten() * np.sqrt(dim)
    rho = x_param * y_param * np.outer(phi_d, phi_d)

    for j in range(1, dim - 1):
        for i in range(j + 1, dim):
            psi_ij = (np.kron(basis(dim, i), basis(dim, j)) - np.kron(basis(dim, j), basis(dim, i))).flatten()
            rho += delta * np.outer(psi_ij, psi_ij)

    for k in range(1, dim):
        phi_k = sum(theta[k] * np.kron(theta, theta) for theta in thetas)
        phi_k *= (dim - 1) ** 1.5 / (dim * np.sqrt(dim - 2))
        psi_k = (
            x_param * np.kron(basis(dim, 0), basis(dim, k)).flatten()
            + y_param * np.kron(basis(dim, k), basis(dim, 0)).flatten()
            + z_param * phi_k
        )
        rho += np.outer(psi_k, psi_k)

    return rho / norm
