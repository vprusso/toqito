"""Check if a set of unitaries is antidistinguishable."""

from typing import Any

import numpy as np

from toqito.channel_metrics.unitary_exclusion import unitary_exclusion


def is_unitary_antidistinguishable(
    unitaries: list[np.ndarray],
    probe: np.ndarray | None = None,
    solver: str = "cvxopt",
    atol: float = 1e-6,
    **kwargs: Any,
) -> bool:
    r"""Check whether a collection of unitaries is antidistinguishable.

    A set of unitaries \(\{U_1, \ldots, U_n\} \subset \text{U}(\mathbb{C}^d)\) is
    *antidistinguishable* if there is a single-shot strategy---an input probe state, possibly
    entangled with an ancilla, together with a measurement on the output---that, for whichever
    unitary was actually applied, never reports that unitary. Equivalently, the set is
    antidistinguishable if and only if the minimum-error unitary *exclusion* problem has optimal
    value zero. Antidistinguishability of qubit unitaries was studied in [@manna2025single]; see
    also state antidistinguishability [@heinosaari2018antidistinguishability].

    When `probe` is provided, antidistinguishability is decided *with respect to that probe*: the
    output states \((\mathbb{I} \otimes U_k)|\phi\rangle\) are formed and checked for
    antidistinguishability. When `probe=None`, the exclusion value is optimized over all input
    strategies, so the result depends only on the unitaries themselves.

    The optimal value is obtained from
    [`unitary_exclusion`][toqito.channel_metrics.unitary_exclusion.unitary_exclusion] with uniform
    prior weights; the set is antidistinguishable exactly when that value is (numerically) zero. The
    less computationally intensive dual formulation is used, matching
    [`is_antidistinguishable`][toqito.state_props.is_antidistinguishable.is_antidistinguishable].

    Args:
        unitaries: A list of \(n \geq 2\) unitary matrices, all of the same dimension \(d\).
        probe: Optional pure probe state, given as a vector of length \(d_A \cdot d\) for some
            ancilla dimension \(d_A \geq 1\), with the unitary acting on the second subsystem. If
            `None`, the exclusion value is optimized over all input strategies.
        solver: Optimization solver passed to `picos`. Default is `"cvxopt"`.
        atol: Absolute tolerance for deciding whether the optimal exclusion value is zero. The
            unitaries are reported antidistinguishable when the value is within `atol` of zero.
            Default is `1e-6`.
        kwargs: Additional keyword arguments forwarded to the `picos` solve method.

    Returns:
        `True` if the unitaries are antidistinguishable; `False` otherwise.

    Examples:
        The three Pauli matrices are antidistinguishable (a maximally entangled probe achieves
        perfect exclusion [@manna2025single]):

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_props import is_unitary_antidistinguishable

        paulis = [
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]]),
        ]
        print(is_unitary_antidistinguishable(paulis, cvxopt_kktsolver="ldl"))
        ```

        The cube roots of the Pauli matrices are not antidistinguishable for any input strategy:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_props import is_unitary_antidistinguishable

        theta = np.pi / 3
        paulis = [
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]]),
        ]
        roots = [np.cos(theta / 2) * np.eye(2) + 1j * np.sin(theta / 2) * p for p in paulis]
        print(is_unitary_antidistinguishable(roots, cvxopt_kktsolver="ldl"))
        ```

    """
    opt_val, _ = unitary_exclusion(
        unitaries,
        probe=probe,
        probs=[1] * len(unitaries),
        primal_dual="dual",
        solver=solver,
        **kwargs,
    )
    return bool(np.isclose(opt_val, 0, atol=atol))
