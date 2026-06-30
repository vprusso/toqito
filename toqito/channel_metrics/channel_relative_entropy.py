"""Quantum relative entropy of channels via semidefinite programming."""

import warnings
from typing import Any

import cvxpy as cvx
import numpy as np

from toqito.channel_props import is_completely_positive, is_quantum_channel
from toqito.cones.integral_relative_entropy import (
    _make_delta,
    _make_gamma,
    _make_grid,
    _sandwich_parameters,
)
from toqito.matrix_ops.partial_trace import partial_trace


def channel_relative_entropy(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    in_dim: int,
    hamiltonian: np.ndarray | None = None,
    energy: float = 0.0,
    epsilon_dec: float = 1e-2,
    mean: bool = False,
    solver: str = "SCS",
    **kwargs: Any,
) -> float | tuple[float, float]:
    r"""Estimate the quantum relative entropy of two channels [@kossmann2025channelrelativeentropy].

    For channels \(\mathcal{N}\) and \(\mathcal{M}\),

    \[
        D(\mathcal{N}\Vert\mathcal{M})
        := \sup_{\rho_{AR}} D\bigl(\mathcal{N}(\rho_{AR})\,\Vert\,\mathcal{M}(\rho_{AR})\bigr).
    \]

    This routine implements SDP lower/upper bounds from the integral representation
    of relative entropy. The formulation assumes that the channels are not too close to
    identical beyond the ``np.allclose`` early exit. When this fails, the auxiliary
    parameters ``mu`` and ``lambda`` from the integral representation are
    degenerate and a ``ValueError`` is raised.

    Args:
        channel_1: Choi matrix for the first channel.
        channel_2: Choi matrix for the second map.
        in_dim: Input dimension \(d_A\) of the channels.
        hamiltonian: Hermitian operator \(H_A\) used in the upper-bound SDP.
            Defaults to the zero matrix of shape ``(in_dim, in_dim)``.
        energy: Energy parameter ``E`` in the upper-bound SDP. Defaults to ``0.0``.
        epsilon_dec: Grid refinement parameter for the discretized integral.
            Defaults to ``1e-2``.
        mean: If ``True``, return the average of the lower and upper bounds; otherwise
            return ``(lower, upper)``.
        solver: The CVXPY solver to use. Defaults to ``"SCS"``.
        kwargs: Additional arguments passed to ``cvxpy.Problem.solve``
            (for example ``eps`` or ``max_iters``).

    Returns:
        Either the midpoint of the bounds or the pair ``(lower, upper)``.

    Raises:
        ValueError: If the Choi matrices have incompatible shapes or dimensions.
        ValueError: If ``hamiltonian`` does not have shape ``(in_dim, in_dim)``.
        ValueError: If ``channel_1`` is not a quantum channel or ``channel_2`` is not CP.
        ValueError: If the integral grid parameters satisfy ``mu <= 0`` or
            ``lambda <= mu`` (channels too close for the bound).

    Examples:
        The qubit dephasing-vs-depolarizing example from [@kossmann2025channelrelativeentropy]
        compares

        \[
            \mathcal{N}_{\mathrm{deph}}(\rho) = 0.4\,\rho + 0.6\,Z\rho Z
        \]

        with

        \[
            \mathcal{M}_{\mathrm{dep}}(\rho)
            = \left(1 - \frac{3p}{4}\right)\rho
            + \frac{p}{4}(X\rho X + Y\rho Y + Z\rho Z).
        \]

        For \(p = 0.05\), the midpoint of the SDP bounds is approximately \(1.97\):

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_metrics import channel_relative_entropy
        from toqito.channels import pauli_channel
        p = 0.05
        channel_1 = np.asarray(pauli_channel([0.4, 0.0, 0.0, 0.6]))
        channel_2 = np.asarray(
            pauli_channel([1 - 3 * p / 4, p / 4, p / 4, p / 4])
        )
        lower, upper = channel_relative_entropy(channel_1, channel_2, in_dim=2)
        print(round((lower + upper) / 2, 3))
        ```

    """
    if channel_1.shape != channel_2.shape:
        raise ValueError("The Choi matrices provided should be of equal dimension.")
    if channel_1.shape[0] != channel_1.shape[1]:
        raise ValueError("The Choi matrix provided must be square.")
    n = channel_1.shape[0]
    if n % in_dim != 0:
        raise ValueError("The Choi dimension must be divisible by in_dim.")
    if hamiltonian is None:
        hamiltonian = np.zeros((in_dim, in_dim), dtype=complex)
    else:
        hamiltonian = np.asarray(hamiltonian, dtype=complex)
    if hamiltonian.shape != (in_dim, in_dim):
        raise ValueError("The Hamiltonian must have shape (in_dim, in_dim).")
    if not is_quantum_channel(channel_1):
        raise ValueError(
            "Channel relative entropy is only defined if channel_1 is a quantum channel."
        )
    if not is_completely_positive(channel_2):
        raise ValueError(
            "Channel relative entropy is only defined if channel_2 is completely positive."
        )
    if np.allclose(channel_1, channel_2):
        if mean:
            return 0.0
        return 0.0, 0.0

    out_dim = n // in_dim
    choi_1 = channel_1
    choi_2 = channel_2
    solve_kwargs = {"eps": 1e-8, "verbose": False, **kwargs}

    mu, lam = _sandwich_parameters(choi_1, choi_2)
    if mu <= 0 or lam <= mu:
        raise ValueError(
            "The integral representation requires 0 < mu < lambda. "
            "This typically means the channels are too close for the bound "
            "(channel_1 may lie in the kernel of channel_2). "
            "If the channels are identical, they are handled by the early return."
        )
    t = _make_grid(mu, lam, epsilon_dec)
    r = len(t)

    alpha = [np.log(t[k] / t[k + 1]) for k in range(r - 1)]
    beta = [t[k + 1] - t[k] for k in range(r - 1)]

    qs = [cvx.Variable((n, n), complex=True) for _ in range(r - 1)]
    rho_a = cvx.Variable((in_dim, in_dim), complex=True)
    eye_out = np.eye(out_dim)
    cons = (
        [cvx.trace(rho_a) == 1, rho_a >> 0]
        + [qs[k] >> 0 for k in range(r - 1)]
        + [cvx.kron(rho_a, eye_out) - qs[k] >> 0 for k in range(r - 1)]
    )
    lower_prob = cvx.Problem(
        cvx.Maximize(
            cvx.real(
                cvx.trace(cvx.kron(rho_a, eye_out) @ (choi_1 - choi_2))
                + cvx.sum(
                    [
                        cvx.trace(qs[k] @ (alpha[k] * choi_1 + beta[k] * choi_2))
                        for k in range(r - 1)
                    ]
                )
            )
        ),
        cons,
    )
    lower_prob.solve(solver=solver, **solve_kwargs)
    if lower_prob.status not in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Lower-bound SDP failed: {lower_prob.status}")
    if lower_prob.status == cvx.OPTIMAL_INACCURATE:
        warnings.warn("Lower-bound SDP returned OPTIMAL_INACCURATE; result may be off.")

    lower = lower_prob.value + np.log(lam) + 1 - lam

    x_var, y_var = cvx.Variable(), cvx.Variable()
    gamma, delta = _make_gamma(t), _make_delta(t)
    ns = [cvx.Variable((n, n), hermitian=True) for _ in range(r + 1)]
    upper_cons = (
        [y_var >= 0]
        + [ns[0] - choi_1 + choi_2 >> 0]
        + [
            ns[k] - gamma[k - 1] * choi_1 - delta[k - 1] * choi_2 >> 0
            for k in range(1, r + 1)
        ]
        + [ns[k] >> 0 for k in range(1, r + 1)]
        + [
            x_var * np.eye(in_dim)
            + y_var * hamiltonian
            - sum(partial_trace(ns[i], dim=[in_dim, out_dim]) for i in range(r + 1))
            >> 0
        ]
    )
    upper_prob = cvx.Problem(cvx.Minimize(cvx.real(x_var + y_var * energy)), upper_cons)
    upper_prob.solve(solver=solver, **solve_kwargs)
    if upper_prob.status not in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Upper-bound SDP failed: {upper_prob.status}")
    if upper_prob.status == cvx.OPTIMAL_INACCURATE:
        warnings.warn("Upper-bound SDP returned OPTIMAL_INACCURATE; result may be off.")

    upper = upper_prob.value + np.log(lam) + 1 - lam
    if mean:
        return (lower + upper) / 2
    return lower, upper
