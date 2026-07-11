"""Channel exclusion SDP implementations.

This module implements channel-level exclusion tasks: the minimum-error channel
exclusion (primal + dual) and the unambiguous exclusion primal. The implementations
follow the lifted-interactive-measurement (tester) formulation from Watrous,
The Theory of Quantum Information, Section 3.5.

Notation and mapping to toqito variables
- `J(Φ)` : Choi matrix of channel Φ. In toqito `kraus_to_choi` and channel
    constructors use the `input ⊗ output` ordering for Choi matrices. Therefore
    inner products of the form `Tr[μ Φ(ρ)]` map to
    `⟨J(Φ), ρ^T ⊗ μ⟩` under this ordering.
- `ρ` : input state. In the lifted formulation we use `X = ρ^T` as the input-side
    marginal variable so that the marginal constraint reads
    `Σ_i W_i = X ⊗ I_Y` where the first tensor factor corresponds to the input.
- `W_i` : lifted positive operator representing the joint `M_i ⊗ ρ^T` strategy.

Careful points (followed in tests and code)
- Choi normalization: Choi matrices returned by toqito channel constructors are
    *unnormalized* (trace = d for trace-preserving maps on d-dimensional inputs).
    When comparing a channel-level optimization against `state_exclusion` on Choi
    states, normalize the Choi matrices by `1/d` to form valid density matrices.
- Unambiguous exclusion: the implementation includes an inconclusive operator
    `W_inc ⪰ 0` with `Σ_i W_i + W_inc = X ⊗ I_Y`, and the objective minimizes
    `Σ_j p_j ⟨J(Φ_j), W_inc⟩` subject to `⟨J(Φ_i), W_i⟩ = 0` for conclusive outcomes.

"""

from typing import Any

import numpy as np
import picos as pc

from toqito.channel_ops import kraus_to_choi
from toqito.channel_props.channel_dim import channel_dim


def channel_exclusion(
    channels: list[np.ndarray | list[np.ndarray] | list[list[np.ndarray]]],
    probs: list[float] | None = None,
    strategy: str = "min_error",
    solver: str = "cvxopt",
    primal_dual: str = "dual",
    probability_tolerance: float = 1e-8,
    **kwargs: Any,
) -> tuple[float, list[np.ndarray]]:
    r"""Compute minimum-error channel exclusion for a collection of channels.

    Given channels \(\Phi_1, \ldots, \Phi_n\) with prior probabilities \(p_1, \ldots, p_n\),
    this function computes the minimum probability of incorrectly excluding the true channel.

    For `strategy="min_error"`, this function solves the lifted SDP:

    \[
        \begin{aligned}
            \text{minimize:} \quad & \sum_{i=1}^{n} p_i \langle J(\Phi_i), W_i \rangle \\
            \text{subject to:} \quad & W_i \succeq 0 \quad \forall i, \\
            & X \succeq 0, \; \text{Tr}(X)=1, \\
            & \sum_{i=1}^{n} W_i = X \otimes \mathbb{I}_{\mathcal{Y}}.
        \end{aligned}
    \]

    where \(J(\Phi_i)\) are Choi matrices in input \(\otimes\) output order and
    \(X = \rho^T\) is the transposed input state variable.

    The dual SDP solved when `primal_dual="dual"` is:

    \[
        \begin{aligned}
            \text{maximize:} \quad & \lambda \\
            \text{subject to:} \quad & Y \preceq p_i J(\Phi_i) \quad \forall i, \\
            & \text{Tr}_{\mathcal{Y}}(Y) \succeq \lambda \mathbb{I}_{\mathcal{X}}, \\
            & Y \in \text{Herm}(\mathcal{X}\otimes\mathcal{Y}).
        \end{aligned}
    \]

    Args:
        channels: List of channels, each provided as a Choi matrix or as Kraus operators.
        probs: Prior weights for the channels. If omitted, a uniform distribution is used. Weights
            need not be normalized (matching ``state_exclusion``); they are normalized internally.
        strategy: Exclusion strategy. In Phase 1, only `"min_error"` is implemented.
        solver: Optimization option for `picos` solver. Default is `"cvxopt"`.
        primal_dual: Option for the optimization problem (`"primal"` or `"dual"`).
        probability_tolerance: Numerical tolerance used when validating the prior weights: weights
            below ``-probability_tolerance`` are rejected as negative, and a weight sum at or below
            ``probability_tolerance`` is rejected as non-positive. Default is ``1e-8``.
        kwargs: Additional arguments to pass to picos' solve method.

    Returns:
        The optimal exclusion probability and a list of optimal strategy operators.

    Raises:
        ValueError: If fewer than 2 channels are provided.
        ValueError: If channels have mismatched dimensions.
        ValueError: If probabilities are invalid.
        ValueError: If `primal_dual` is not `"primal"` or `"dual"`.
        ValueError: If `strategy` is not supported.
        ValueError: If `strategy="unambiguous"` is requested with `primal_dual="dual"`.

    Examples:
        Consider the two identical depolarizing channels

        \[
            \Phi_1(\rho) = 0.3 \rho + 0.7 \operatorname{Tr}(\rho) \frac{\mathbb{I}}{2}, \quad
            \Phi_2(\rho) = 0.3 \rho + 0.7 \operatorname{Tr}(\rho) \frac{\mathbb{I}}{2}.
        \]

        Since the channels are identical, it is not possible to exclude either
        with better than random guessing. The optimal error probability is 0.5.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_opt import channel_exclusion
        from toqito.channels import depolarizing

        channels = [depolarizing(2, 0.3), depolarizing(2, 0.3)]
        val, _ = channel_exclusion(channels, probs=[0.5, 0.5], primal_dual="dual")

        print(np.around(val, decimals=2))
        ```

        Two different depolarizing channels can be excluded with a lower error
        probability.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_opt import channel_exclusion
        from toqito.channels import depolarizing

        channels = [depolarizing(2, 0.2), depolarizing(2, 0.7)]
        val, _ = channel_exclusion(channels, probs=[0.5, 0.5], primal_dual="dual")

        print(np.around(val, decimals=2))
        ```

        Channels can also be provided as Kraus operators.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_opt import channel_exclusion

        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        Z = np.array([[1.0, 0.0], [0.0, -1.0]])

        val, _ = channel_exclusion([[X], [Z]], probs=[0.5, 0.5], primal_dual="dual")

        print(np.around(val, decimals=2))
        ```

    """
    if len(channels) < 2:
        raise ValueError("At least 2 channels are required for channel exclusion.")

    if strategy not in ("min_error", "unambiguous"):
        raise ValueError("The strategy must be either 'min_error' or 'unambiguous'.")

    # `unambiguous` is supported (primal only) but handled after we normalize probabilities
    # and convert Kraus inputs to Choi matrices.

    if primal_dual not in ("primal", "dual"):
        raise ValueError("The primal_dual option must be either 'primal' or 'dual'.")

    n_channels = len(channels)
    probs = [1 / n_channels] * n_channels if probs is None else probs

    if len(probs) != n_channels:
        raise ValueError("The number of probabilities must match the number of channels.")

    probs_arr = np.array(probs, dtype=float)
    if np.any(probs_arr < -probability_tolerance):
        raise ValueError("Prior probabilities must be non-negative.")

    probs_sum = float(np.sum(probs_arr))
    if probs_sum <= probability_tolerance:
        raise ValueError("Prior probabilities must have a positive sum.")
    # Accept unnormalized weights (matching state_exclusion, e.g. antidistinguishability passes
    # [1, ..., 1]) by normalizing internally.
    probs_arr = probs_arr / probs_sum

    choi_channels: list[np.ndarray] = []
    channel_dims = []
    for channel in channels:
        # The environment dimension is unused here, so skip its matrix_rank computation.
        dim_in, dim_out, _ = channel_dim(channel, compute_env_dim=False)
        channel_dims.append(np.array([dim_in, dim_out]))
        choi_channels.append(kraus_to_choi(channel) if isinstance(channel, list) else channel)

    first_dim = channel_dims[0]
    if not all(np.array_equal(first_dim, chan_dim) for chan_dim in channel_dims[1:]):
        raise ValueError("All channels must have the same input and output dimensions.")

    dim_in = int(first_dim[0][0])
    dim_out = int(first_dim[1][0])

    if strategy == "unambiguous":
        if primal_dual != "primal":
            raise ValueError("Unambiguous strategy supports only the primal formulation for Phase 2.")
        return _unambiguous_primal(choi_channels, probs_arr.tolist(), dim_in, dim_out, solver=solver, **kwargs)

    if primal_dual == "primal":
        return _min_error_primal(choi_channels, probs_arr.tolist(), dim_in, dim_out, solver=solver, **kwargs)

    return _min_error_dual(choi_channels, probs_arr.tolist(), dim_in, dim_out, solver=solver, **kwargs)


def _min_error_primal(
    channels: list[np.ndarray],
    probs: list[float],
    dim_in: int,
    dim_out: int,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[np.ndarray]]:
    """Solve the primal SDP for minimum-error channel exclusion."""
    n_channels = len(channels)
    problem = pc.Problem()

    strategy_ops = [
        pc.HermitianVariable(f"W[{idx}]", (dim_in * dim_out, dim_in * dim_out)) for idx in range(n_channels)
    ]
    x_var = pc.HermitianVariable("X", (dim_in, dim_in))

    problem.add_list_of_constraints(strategy_ops[idx] >> 0 for idx in range(n_channels))
    problem.add_constraint(x_var >> 0)
    problem.add_constraint(pc.trace(x_var) == 1)

    # Choi operators are ordered as input x output, so the lifted marginal is X x I_out.
    problem.add_constraint(pc.sum(strategy_ops) == x_var @ np.eye(dim_out))

    objective = pc.sum([probs[idx] * (channels[idx] | strategy_ops[idx]).real for idx in range(n_channels)])
    problem.set_objective("min", objective)

    solution = problem.solve(solver=solver, **kwargs)
    return solution.value, [np.array(var.value) for var in strategy_ops]


def _min_error_dual(
    channels: list[np.ndarray],
    probs: list[float],
    dim_in: int,
    dim_out: int,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[np.ndarray]]:
    """Solve the dual SDP for minimum-error channel exclusion."""
    n_channels = len(channels)
    problem = pc.Problem()

    y_var = pc.HermitianVariable("Y", (dim_in * dim_out, dim_in * dim_out))
    lambda_var = pc.RealVariable("lambda")

    dual_constraints = [problem.add_constraint(y_var << probs[idx] * channels[idx]) for idx in range(n_channels)]
    problem.add_constraint(pc.partial_trace(y_var, 1, (dim_in, dim_out)) >> lambda_var * np.eye(dim_in))

    problem.set_objective("max", lambda_var)
    solution = problem.solve(solver=solver, **kwargs)

    strategy_ops = [np.array(constraint.dual) for constraint in dual_constraints]
    return solution.value, strategy_ops


def _unambiguous_primal(
    channels: list[np.ndarray],
    probs: list[float],
    dim_in: int,
    dim_out: int,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[np.ndarray]]:
    """Solve the primal SDP for unambiguous channel exclusion.

    Returns the minimal inconclusive probability and the set of strategy operators
    (W_1, ..., W_n, W_inc).
    """
    n_channels = len(channels)
    problem = pc.Problem()

    W_vars = [pc.HermitianVariable(f"W[{i}]", (dim_in * dim_out, dim_in * dim_out)) for i in range(n_channels)]
    W_inc = pc.HermitianVariable("W_inc", (dim_in * dim_out, dim_in * dim_out))
    x_var = pc.HermitianVariable("X", (dim_in, dim_in))

    problem.add_list_of_constraints(W_vars[i] >> 0 for i in range(n_channels))
    problem.add_constraint(W_inc >> 0)
    problem.add_constraint(x_var >> 0)
    problem.add_constraint(pc.trace(x_var) == 1)

    problem.add_constraint(pc.sum(W_vars) + W_inc == x_var @ np.eye(dim_out))

    # Zero-error constraints for conclusive outcomes
    problem.add_list_of_constraints((channels[i] | W_vars[i]) == 0 for i in range(n_channels))

    objective = pc.sum([probs[i] * (channels[i] | W_inc) for i in range(n_channels)])
    problem.set_objective("min", objective)

    solution = problem.solve(solver=solver, **kwargs)

    return solution.value, [np.array(var.value) for var in W_vars] + [np.array(W_inc.value)]
