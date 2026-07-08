"""Compute the probability of error of excluding a unitary from a collection of unitaries."""

from typing import Any

import numpy as np

from toqito.channel_metrics.channel_exclusion import channel_exclusion
from toqito.matrix_props import is_unitary
from toqito.state_opt.state_exclusion import state_exclusion


def unitary_exclusion(
    unitaries: list[np.ndarray],
    probe: np.ndarray | None = None,
    probs: list[float] | None = None,
    strategy: str = "min_error",
    primal_dual: str = "dual",
    solver: str = "cvxopt",
    **kwargs: Any,
) -> tuple[float, list]:
    r"""Compute the optimal probability of error for unitary exclusion (antidistinguishability).

    In the *unitary exclusion* problem, one of the unitaries \(\{U_1, \ldots, U_n\} \subset
    \text{U}(\mathbb{C}^d)\) is applied (with prior weights \(p_1, \ldots, p_n\)) to one share of a
    bipartite probe state \(|\phi\rangle \in \mathbb{C}^{d_A} \otimes \mathbb{C}^d\), producing the
    output states

    \[
        |\psi_k\rangle = (\mathbb{I}_{d_A} \otimes U_k)|\phi\rangle,
    \]

    and the goal is to conclusively rule out one unitary that was *not* applied by measuring the
    output. The set of unitaries is called *antidistinguishable with respect to the probe* when the
    optimal error probability is zero. This is the unitary analogue of quantum state exclusion
    [@bandyopadhyay2014conclusive] and was studied for qubit unitaries in [@manna2025single].

    Two scenarios are supported:

    - **Fixed probe** (`probe` is provided): the output states \(|\psi_k\rangle\) are formed and the
      problem reduces to the state exclusion SDP
      [`state_exclusion`][toqito.state_opt.state_exclusion.state_exclusion]. The probe may include an
      ancilla of any dimension \(d_A \geq 1\): a vector of length \(d_A \cdot d\) is interpreted as an
      element of \(\mathbb{C}^{d_A} \otimes \mathbb{C}^d\) with the unitary acting on the *second*
      subsystem.
    - **Optimal probe** (`probe=None`): the error probability is minimized over *all* input
      strategies simultaneously (probe states of unbounded ancilla dimension together with output
      measurements) by solving the lifted tester SDP
      [`channel_exclusion`][toqito.channel_metrics.channel_exclusion.channel_exclusion] with each
      unitary treated as the channel \(\rho \mapsto U_k \rho U_k^*\). This yields the exclusion value
      of the unitaries themselves, certifying a global optimum that no choice of probe can improve
      upon.

    Args:
        unitaries: A list of \(n \geq 2\) unitary matrices, all of the same dimension \(d\).
        probe: Optional pure probe state, given as a vector of length \(d_A \cdot d\) for some
            ancilla dimension \(d_A \geq 1\). If `None`, the exclusion value is optimized over all
            input strategies via the channel exclusion SDP.
        probs: Prior weights for the unitaries. If omitted, a uniform distribution is assumed.
        strategy: Either `"min_error"` (default) or `"unambiguous"`. For `probe=None`, the
            unambiguous strategy is supported in its primal formulation only (matching
            `channel_exclusion`).
        primal_dual: Option for the optimization problem (`"primal"` or `"dual"`).
        solver: Optimization option for `picos` solver. Default is `"cvxopt"`.
        kwargs: Additional arguments to pass to picos' solve method.

    Returns:
        The optimal probability of error together with the optimal strategy operators (measurement
        operators for a fixed probe; lifted tester operators for `probe=None`).

    Raises:
        ValueError: If fewer than 2 unitaries are provided.
        ValueError: If any provided matrix is not unitary.
        ValueError: If the unitaries do not all have the same dimension.
        ValueError: If the probe length is not a positive multiple of the unitary dimension.

    Examples:
        The three Pauli matrices are antidistinguishable when probed with the maximally entangled
        state, as shown in [@manna2025single]:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_metrics import unitary_exclusion
        from toqito.states import bell

        paulis = [
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]]),
        ]
        value, _ = unitary_exclusion(paulis, probe=bell(0))
        print(np.around(value, decimals=6))
        ```

        For the cube roots of the Pauli matrices, perfect exclusion is impossible for *any* input
        strategy. Omitting the probe optimizes over all of them:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_metrics import unitary_exclusion

        theta = np.pi / 3
        paulis = [
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]]),
        ]
        roots = [np.cos(theta / 2) * np.eye(2) + 1j * np.sin(theta / 2) * p for p in paulis]
        value, _ = unitary_exclusion(roots, cvxopt_kktsolver="ldl")
        print(np.around(value, decimals=6))
        ```

    """
    if len(unitaries) < 2:
        raise ValueError("At least 2 unitaries are required for unitary exclusion.")

    dim = unitaries[0].shape[0]
    for unitary in unitaries:
        if not is_unitary(unitary):
            raise ValueError("All matrices provided must be unitary.")
        if unitary.shape[0] != dim:
            raise ValueError("All unitaries must have the same dimension.")

    if probe is None:
        return channel_exclusion(
            [[unitary] for unitary in unitaries],
            probs=probs,
            strategy=strategy,
            primal_dual=primal_dual,
            solver=solver,
            **kwargs,
        )

    probe_vec = np.asarray(probe, dtype=complex).flatten()
    dim_ancilla, remainder = divmod(probe_vec.shape[0], dim)
    if dim_ancilla < 1 or remainder != 0:
        raise ValueError(
            f"The probe length ({probe_vec.shape[0]}) must be a positive multiple of the unitary dimension ({dim})."
        )

    output_states = [(np.kron(np.eye(dim_ancilla), unitary) @ probe_vec).reshape(-1, 1) for unitary in unitaries]
    return state_exclusion(
        output_states,
        probs=probs,
        strategy=strategy,
        primal_dual=primal_dual,
        solver=solver,
        **kwargs,
    )
