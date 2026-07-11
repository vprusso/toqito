"""Computes the maximum probability of distinguishing a collection of quantum channels."""

from typing import Any

import numpy as np
import picos as pc

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import kraus_to_choi
from toqito.channel_props.channel_dim import channel_dim


def channel_distinguishability(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    psi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]] | None = None,
    probs: list[float] | None = None,
    dim: int | list[int] | np.ndarray | None = None,
    strategy: str = "bayesian",
    solver: str = "cvxopt",
    primal_dual: str = "dual",
    **kwargs: Any,
) -> tuple[float, list[np.ndarray]]:
    r"""Compute the optimal probability of distinguishing a collection of quantum channels.

    Bayesian discrimination of \(n \geq 2\) quantum channels and minimax discrimination of two
    quantum channels are implemented.

    For Bayesian discrimination, the channels to be distinguished are given with an a priori
    probability distribution. The task is to find an input state and a POVM on the (reference and)
    output system for which the error probability of identifying which channel acted is minimized.
    In the language of statistical decision theory, the problem is equivalent to minimizing quantum
    Bayes' risk. For two channels, the optimal value admits a closed form in terms of the completely
    bounded trace norm (Section 3.3.3 of [@watrous2018theory]), which is used whenever the channels
    are supplied through the two-argument signature ``channel_distinguishability(phi, psi, ...)``.

    For \(n \geq 2\) channels supplied as a single list (with ``psi=None``), the full semidefinite
    program of Section 3.5 of [@watrous2018theory] is solved. Given channels
    \(\{\Phi_1, \ldots, \Phi_n\}\) with priors \(\{p_1, \ldots, p_n\}\) and Choi matrices
    \(J(\Phi_i)\), the primal problem is

    \[
        \begin{aligned}
            \text{maximize:} \quad & \sum_{i=1}^n p_i \text{Tr}\left[P_i J(\Phi_i)\right] \\
            \text{subject to:} \quad & \sum_{i=1}^n P_i = \rho \otimes \mathbb{I}_{\mathcal{Y}}, \\
            & P_i \succeq 0 \quad \forall i, \\
            & \rho \succeq 0, \; \text{Tr}(\rho) = 1,
        \end{aligned}
    \]

    and the corresponding dual problem is

    \[
        \begin{aligned}
            \text{minimize:} \quad & \lambda \\
            \text{subject to:} \quad & Y \succeq p_i J(\Phi_i) \quad \forall i, \\
            & \text{Tr}_{\mathcal{Y}}(Y) \preceq \lambda \, \mathbb{I}_{\mathcal{X}}.
        \end{aligned}
    \]

    Note on notation: in [@watrous2018theory], each channel maps operators on an input space
    \(\mathcal{X}\) to operators on an output space \(\mathcal{Y}\), and the Choi operator
    \(J(\Phi)\) lives on \(\mathcal{Y} \otimes \mathcal{X}\). In toqito's convention the Choi
    matrix instead lives on \(\mathcal{X} \otimes \mathcal{Y}\) (input \(\otimes\) output); that
    is, subsystem index 0 is the input space \(\mathcal{X}\) (of dimension ``dim_in``) and
    subsystem index 1 is the output space \(\mathcal{Y}\) (of dimension ``dim_out``). Accordingly,
    tracing out the output space \(\mathcal{Y}\) in the dual constraint corresponds to
    ``picos.partial_trace(Y, 1)`` in the implementation.

    In the minimax problem, there are no a priori probabilities.
    Minimax discrimination of two channels consists of finding the
    optimal input state so that the two possible output states are discriminated
    with minimum risk. ([@dariano2005minimax]). Minimax discrimination is only implemented for two
    channels; requesting it for more than two channels raises a ``ValueError``.

    QETLAB's functionality inspired the Bayesian option [@qetlablink]
    and the minimax option is adapted from QuTIpy [@qutipylink].

    Args:
        phi: Either a single superoperator (provided as a Choi matrix, or as a (1d or 2d) list of
             numpy arrays whose entries are its Kraus operators) to be distinguished from ``psi``,
             or, when ``psi`` is ``None``, a list of \(n \geq 2\) superoperators (each a
             Choi matrix or a list of Kraus operators) to be mutually distinguished.
        psi: A superoperator. It should be provided either as a Choi matrix,
             or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
             If ``None``, ``phi`` is interpreted as a list of channels.
        probs: Prior weights of the channels (Bayesian strategy only). If omitted, a uniform
            distribution is used. Weights need not be normalized (matching ``state_exclusion``);
            they are normalized internally.
        dim: Input and output dimensions of the channels.
        strategy: Whether to perform Bayesian or minimax discrimination task. Possible
                  values are "Bayesian" and "minimax". Default option is `strategy="Bayesian"`.
        solver: Optimization option for `picos` solver. Default option is `solver="cvxopt"`.
        primal_dual: Option for the optimization problem. Default option is `primal_dual="dual"`.
        kwargs: Additional arguments to pass to picos' solve method.

    Returns:
        A tuple ``(value, operators)``. ``value`` is the optimal probability of discriminating the
        channels. ``operators`` holds the optimal strategy operators for the SDP branches (the
        measurement operators :math:`P_i` for ``primal_dual="primal"`` and the dual operator for
        ``primal_dual="dual"``) and is an empty list for the closed-form two-channel Bayesian
        branch and for degenerate priors.

    Raises:
        ValueError: If strategy is neither Bayesian nor minimax.
        ValueError: If strategy is minimax and more than 2 channels are provided.
        ValueError: If fewer than 2 channels are provided in list mode.
        ValueError: If channels have different input or output dimensions.
        ValueError: If the prior weights are negative or sum to zero.
        ValueError: If number of prior probabilities is not equal to the number of channels.

    Examples:
        Optimal probability of distinguishing two amplitude damping channels in the Bayesian setting:

        ```python exec="1" source="above" result="text"
        from toqito.channels import amplitude_damping
        from toqito.channel_ops import kraus_to_choi
        from toqito.channel_opt import channel_distinguishability
        # Define two amplitude damping channels with gamma=0.25 and gamma=0.5
        choi_ch_1 = kraus_to_choi(amplitude_damping(gamma=0.25, return_kraus_ops=True))
        choi_ch_2 = kraus_to_choi(amplitude_damping(gamma=0.5, return_kraus_ops=True))

        probs = [0.5, 0.5]

        value, _ = channel_distinguishability(choi_ch_1, choi_ch_2, probs)
        print(value)
        ```

        Optimal probability of distinguishing three depolarizing channels with distinct noise
        parameters, supplied as a single list of channels:

        ```python exec="1" source="above" result="text"
        from toqito.channels import depolarizing
        from toqito.channel_opt import channel_distinguishability
        # Define three depolarizing channels with distinct noise parameters.
        channels = [depolarizing(2, 0.1), depolarizing(2, 0.5), depolarizing(2, 0.9)]

        value, _ = channel_distinguishability(channels, probs=[1 / 3, 1 / 3, 1 / 3])
        print(value)
        ```

        Optimal probability of distinguishing two amplitude damping channels in the minimax setting:

        ```python exec="1" source="above" result="text"
        from toqito.channels import amplitude_damping
        from toqito.channel_ops import kraus_to_choi
        from toqito.channel_opt import channel_distinguishability
        # Define two amplitude damping channels with gamma=0.25 and gamma=0.5
        choi_ch_1 = kraus_to_choi(amplitude_damping(gamma=0.25, return_kraus_ops=True))
        choi_ch_2 = kraus_to_choi(amplitude_damping(gamma=0.5, return_kraus_ops=True))

        value, _ = channel_distinguishability(
            choi_ch_1, choi_ch_2, None, [2, 2], strategy="minimax", primal_dual="primal"
        )
        print(value)
        ```

    """
    # Checking for errors common to both modes.
    if strategy.lower() not in ("bayesian", "minimax"):
        raise ValueError("The strategy must either be Bayesian or Minimax.")

    if primal_dual not in {"primal", "dual"}:
        raise ValueError("The primal_dual option must be either 'primal' or 'dual'.")

    if psi is not None:
        return _two_channel_distinguishability(phi, psi, probs, dim, strategy, solver, primal_dual, **kwargs)

    # n-channel (list) mode: `phi` is a list of channels, each given as a Choi matrix or Kraus list.
    if not isinstance(phi, list) or len(phi) < 2:
        raise ValueError("When psi is None, phi must be a list of at least 2 channels.")

    channels = phi
    num_channels = len(channels)

    # Get the input and output dimensions of every channel and check they all agree. The environment
    # dimension is not used here, so skip the extra matrix_rank computation it would require.
    channel_dims = [channel_dim(channel, dim=dim, compute_env_dim=False) for channel in channels]
    dims = [np.array([d_in, d_out]) for d_in, d_out, _ in channel_dims]
    if any(not np.array_equal(dims[0], dims_i) for dims_i in dims[1:]):
        raise ValueError("The channels must have the same dimension input and output spaces as each other.")
    dim_in, dim_out = int(dims[0][0][0]), int(dims[0][1][0])

    # Convert any Kraus representations to Choi matrices.
    choi_matrices = [kraus_to_choi(channel) if isinstance(channel, list) else channel for channel in channels]

    if strategy.lower() == "minimax":
        if num_channels != 2:
            raise ValueError("Minimax discrimination is only supported for exactly 2 channels.")
        if primal_dual == "primal":
            return _minimax_primal(choi_matrices[0], choi_matrices[1], dim_in, dim_out, solver=solver, **kwargs)
        return _minimax_dual(choi_matrices[0], choi_matrices[1], dim_in, dim_out, solver=solver, **kwargs)

    probs_arr = _validate_probs(probs, num_channels)
    if max(probs_arr) >= 1:
        return 1.0, []

    if primal_dual == "primal":
        return _bayesian_primal(choi_matrices, probs_arr, dim_in, dim_out, solver=solver, **kwargs)
    return _bayesian_dual(choi_matrices, probs_arr, dim_in, dim_out, solver=solver, **kwargs)


def _two_channel_distinguishability(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    psi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    probs: list[float] | None,
    dim: int | list[int] | np.ndarray | None,
    strategy: str,
    solver: str,
    primal_dual: str,
    **kwargs: Any,
) -> tuple[float, list[np.ndarray]]:
    """Distinguish two channels (legacy two-argument mode)."""
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

    if not np.array_equal(dim_phi, dim_psi):
        raise ValueError("The channels must have the same dimension input and output spaces as each other.")

    if strategy.lower() == "bayesian":
        probs_arr = _validate_probs(probs, 2)

        if max(probs_arr) >= 1:
            return 1.0, []

        # optimal success probability is minimizing error probability (Bayes risk).
        value = 1 / 2 * (1 + completely_bounded_trace_norm(probs_arr[0] * phi - probs_arr[1] * psi))
        return float(value), []

    if primal_dual == "primal":
        return _minimax_primal(phi, psi, d_in_phi[0], d_out_phi[0], solver=solver, **kwargs)

    return _minimax_dual(phi, psi, d_in_phi[0], d_out_phi[0], solver=solver, **kwargs)


def _validate_probs(probs: list[float] | None, num_channels: int) -> np.ndarray:
    """Validate and normalize the prior weights of the channels."""
    # Default to a uniform prior, and accept unnormalized weights (matching state_exclusion).
    probs = [1 / num_channels] * num_channels if probs is None else probs

    if len(probs) != num_channels:
        raise ValueError(f"probs must be a probability distribution with {num_channels} entries.")

    probs_arr = np.array(probs, dtype=float)
    if np.any(probs_arr < 0):
        raise ValueError("Prior probabilities must be non-negative.")
    probs_sum = float(np.sum(probs_arr))
    if probs_sum <= 0:
        raise ValueError("Prior probabilities must have a positive sum.")
    return probs_arr / probs_sum


def _bayesian_primal(
    choi_matrices: list[np.ndarray],
    probs: np.ndarray,
    dimA: int,
    dimB: int,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[np.ndarray]]:
    """Find the primal problem for Bayesian discrimination of n quantum channels.

    See Section 3.5 of [@watrous2018theory]. The Choi matrices live on input (x) output
    (subsystem 0 is the input space of dimension dimA, subsystem 1 the output space of
    dimension dimB).
    """
    num_channels = len(choi_matrices)

    problem = pc.Problem()

    P_var = [pc.HermitianVariable(f"P[{i}]", (dimA * dimB, dimA * dimB)) for i in range(num_channels)]
    rho = pc.HermitianVariable("rho", (dimA, dimA))

    problem.add_list_of_constraints(P_var[i] >> 0 for i in range(num_channels))
    problem.add_constraint(pc.sum(P_var) == rho @ np.eye(dimB))
    problem.add_constraint(rho >> 0)
    problem.add_constraint(pc.trace(rho) == 1)

    problem.set_objective(
        "max", pc.sum([probs[i] * pc.trace(P_var[i] * choi_matrices[i]).real for i in range(num_channels)])
    )

    problem.solve(solver=solver, **kwargs)

    return problem.value, [np.array(var.value) for var in P_var]


def _bayesian_dual(
    choi_matrices: list[np.ndarray],
    probs: np.ndarray,
    dimA: int,
    dimB: int,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[np.ndarray]]:
    """Find the dual problem for Bayesian discrimination of n quantum channels.

    See Section 3.5 of [@watrous2018theory]. Minimize lambda subject to Y >= p_i J(Phi_i) for all i
    and Tr_out(Y) <= lambda * I_in, i.e. minimize the largest eigenvalue of Tr_out(Y).
    """
    num_channels = len(choi_matrices)

    problem = pc.Problem()

    a_var = pc.RealVariable("a")
    Y_var = pc.HermitianVariable("Y", (dimA * dimB, dimA * dimB))

    # Trace out the output space (subsystem index 1 in toqito's input (x) output Choi convention).
    Y0 = pc.partial_trace(Y_var, subsystems=1, dimensions=[dimA, dimB])

    problem.add_list_of_constraints(Y_var >> probs[i] * choi_matrices[i] for i in range(num_channels))
    problem.add_constraint(Y0 << a_var * np.eye(dimA))

    problem.set_objective("min", a_var)

    problem.solve(solver=solver, **kwargs)

    return problem.value, [np.array(Y_var.value)]


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
