"""Computes the maximum probability of distinguishing two quantum channels."""

from typing import Any

import numpy as np
import picos as pc

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import kraus_to_choi
from toqito.channel_props.channel_dim import channel_dim


def channel_distinguishability(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    psi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    probs: list[float] | None = None,
    dim: int | list[int] | np.ndarray | None = None,
    strategy: str = "bayesian",
    solver: str = "cvxopt",
    primal_dual: str = "dual",
    **kwargs: Any,
) -> tuple[float, list[np.ndarray]]:
    r"""Compute the optimal probability of distinguishing two quantum channels.

    Bayesian and minimax discrimination of two quantum channels are implemented.

    For Bayesian discrimination, channels to be distinguished should have a given a priori probability distribution.
    The task of discriminating channels can be connected to the completely bounded trace norm
    (Section 3.3.3 of [@watrous2018theory]).
    The problem is finding POVMs for which error probability of discrimination of
    output states is minimized after input state is acted on by the two quantum channels.
    In the language of statistical decision theory, the problem is equivalent to minimizing quantum Bayes' risk.

    In the minimax problem, there are no a priori probabilities.
    Minimax discrimination of two channels consists of finding the
    optimal input state so that the two possible output states are discriminated
    with minimum risk. ([@dariano2005minimax]).

    QETLAB's functionality inspired the Bayesian option [@qetlablink]
    and the minimax option is adapted from QuTIpy [@qutipylink].

    Args:
        phi: A superoperator. It should be provided either as a Choi matrix,
             or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
        psi: A superoperator. It should be provided either as a Choi matrix,
             or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
        probs: Prior weights of the two channels (Bayesian strategy only). If omitted, a uniform
            distribution is used. Weights need not be normalized (matching ``state_exclusion``);
            they are normalized internally.
        dim: Input and output dimensions of the channels.
        strategy: Whether to perform Bayesian or minimax discrimination task. Possible
                  values are "Bayesian" and "minimax". Default option is `strategy="Bayesian"`.
        solver: Optimization option for `picos` solver. Default option is `solver="cvxopt"`.
        primal_dual: Option for the optimization problem. Default option is `solver="cvxopt"`.
        kwargs: Additional arguments to pass to picos' solve method.

    Returns:
        A tuple ``(value, operators)``. ``value`` is the optimal probability of discriminating the
        two channels. ``operators`` holds the optimal strategy operators for the minimax SDP
        branches (the measurement operators for ``primal_dual="primal"`` and the dual operator for
        ``primal_dual="dual"``) and is an empty list for the closed-form Bayesian branch.

    Raises:
        ValueError: If strategy is neither Bayesian nor minimax.
        ValueError: If channels have different input or output dimensions.
        ValueError: If the prior weights are negative or sum to zero.
        ValueError: If number of prior probabilities not equal to 2.

    Examples:
        Optimal probability of distinguishing two amplitude damping channels in the Bayesian setting:

        ```python exec="1" source="above" result="text"
        from toqito.channels import amplitude_damping
        from toqito.channel_ops import kraus_to_choi
        from toqito.channel_metrics import channel_distinguishability
        # Define two amplitude damping channels with gamma=0.25 and gamma=0.5
        choi_ch_1 = kraus_to_choi(amplitude_damping(gamma=0.25, return_kraus_ops=True))
        choi_ch_2 = kraus_to_choi(amplitude_damping(gamma=0.5, return_kraus_ops=True))

        probs = [0.5, 0.5]

        value, _ = channel_distinguishability(choi_ch_1, choi_ch_2, probs)
        print(value)
        ```

        Optimal probability of distinguishing two amplitude damping channels in the minimax setting:

        ```python exec="1" source="above" result="text"
        from toqito.channels import amplitude_damping
        from toqito.channel_ops import kraus_to_choi
        from toqito.channel_metrics import channel_distinguishability
        # Define two amplitude damping channels with gamma=0.25 and gamma=0.5
        choi_ch_1 = kraus_to_choi(amplitude_damping(gamma=0.25, return_kraus_ops=True))
        choi_ch_2 = kraus_to_choi(amplitude_damping(gamma=0.5, return_kraus_ops=True))

        value, _ = channel_distinguishability(
            choi_ch_1, choi_ch_2, None, [2, 2], strategy="minimax", primal_dual="primal"
        )
        print(value)
        ```

    """
    # Get the input and output dimensions of phi and psi. The environment dimension is not used here, so skip the
    # extra matrix_rank computation it would require.
    d_in_phi, d_out_phi, _ = channel_dim(phi, dim=dim, compute_env_dim=False)
    d_in_psi, d_out_psi, _ = channel_dim(psi, dim=dim, compute_env_dim=False)

    # If the variable `phi` and/or `psi` are provided as a list, we assume this is a list
    # of Kraus operators. We convert to choi matrices if not provided as choi matrix.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    if isinstance(psi, list):
        psi = kraus_to_choi(psi)

    dim_phi, dim_psi = np.array([d_in_phi, d_out_phi]), np.array([d_in_psi, d_out_psi])

    # checking for errors.
    if strategy.lower() not in ("bayesian", "minimax"):
        raise ValueError("The strategy must either be Bayesian or Minimax.")

    if primal_dual not in {"primal", "dual"}:
        raise ValueError("The primal_dual option must be either 'primal' or 'dual'.")

    if not np.array_equal(dim_phi, dim_psi):
        raise ValueError("The channels must have the same dimension input and output spaces as each other.")

    if strategy.lower() == "bayesian":
        # Default to a uniform prior, and accept unnormalized weights (matching state_exclusion).
        probs = [1 / 2, 1 / 2] if probs is None else probs

        if len(probs) != 2:
            raise ValueError("probs must be a probability distribution with 2 entries.")

        probs_arr = np.array(probs, dtype=float)
        if np.any(probs_arr < 0):
            raise ValueError("Prior probabilities must be non-negative.")
        probs_sum = float(np.sum(probs_arr))
        if probs_sum <= 0:
            raise ValueError("Prior probabilities must have a positive sum.")
        probs_arr = probs_arr / probs_sum

        if max(probs_arr) >= 1:
            return 1.0, []

        # optimal success probability is minimizing error probability (Bayes risk).
        value = 1 / 2 * (1 + completely_bounded_trace_norm(probs_arr[0] * phi - probs_arr[1] * psi))
        return float(value), []

    if primal_dual == "primal":
        return _minimax_primal(phi, psi, d_in_phi[0], d_out_phi[0], solver=solver, **kwargs)

    return _minimax_dual(phi, psi, d_in_phi[0], d_out_phi[0], solver=solver, **kwargs)


def _minimax_dual(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    psi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    dimA: int,
    dimB: int,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[np.ndarray]]:
    """Find the dual problem for minimax quantum channel distinguishability SDP."""
    J_var = list([phi, psi])

    problem = pc.Problem()

    a_var = pc.RealVariable("a", lower=0)
    P_var = pc.RealVariable("P", 2)
    Y_var = pc.HermitianVariable("Y", (dimA * dimB, dimA * dimB))

    Y0 = pc.partial_trace(Y_var, 1)

    problem.add_list_of_constraints(Y_var >> P_var[i] * J_var[i] for i in range(2))
    problem.add_constraint(pc.sum(P_var) == 1)
    problem.add_constraint(Y_var >> 0)
    problem.add_constraint(Y0 == a_var * np.eye(dimA))
    problem.add_list_of_constraints(p >= 0 for p in P_var)

    problem.set_objective("min", a_var)

    problem.solve(solver=solver, **kwargs)

    return problem.value, [np.array(Y_var.value)]


def _minimax_primal(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    psi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    dimA: int,
    dimB: int,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[np.ndarray]]:
    """Find the primal problem for minimax quantum channel distinguishability SDP."""
    J_var = list([phi, psi])

    problem = pc.Problem()

    a_var = pc.RealVariable("a", lower=0)
    P_var = [pc.HermitianVariable(f"P[{i}]", (dimA * dimB, dimA * dimB)) for i in range(2)]
    rho = pc.HermitianVariable("rho", (dimA, dimA))

    problem.add_list_of_constraints(P_var[i] >> 0 for i in range(2))
    problem.add_list_of_constraints(a_var <= pc.trace(P_var[i] * J_var[i]).real for i in range(2))
    problem.add_constraint(pc.sum(P_var) == rho @ np.eye(dimB))
    problem.add_constraint(rho >> 0)
    problem.add_constraint(pc.trace(rho) == 1)

    problem.set_objective("max", a_var)

    problem.solve(solver=solver, **kwargs)

    return problem.value, [np.array(var.value) for var in P_var]
