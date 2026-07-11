"""Check if a set of quantum channels is antidistinguishable."""

from typing import Any

import numpy as np

from toqito.channel_opt import channel_exclusion


def is_channel_antidistinguishable(
    channels: list[np.ndarray | list[np.ndarray] | list[list[np.ndarray]]],
    solver: str = "cvxopt",
    atol: float = 1e-6,
    **kwargs: Any,
) -> bool:
    r"""Check whether a collection of quantum channels is antidistinguishable.

    A set of channels \(\{\Xi_1, \ldots, \Xi_n\}\) is *antidistinguishable* if there is a single-shot
    strategy (an input state, possibly entangled with an ancilla, together with a measurement on the
    output) that, for whichever channel was actually applied, never reports that channel. Equivalently,
    the set is antidistinguishable if and only if the minimum-error channel *exclusion* problem has
    optimal value zero, i.e. the true channel can always be ruled out. This mirrors state
    antidistinguishability [@heinosaari2018antidistinguishability] lifted to channels, and is decided
    here via the channel exclusion SDP (Section 3.5 of [@watrous2018theory]).

    The optimal value is obtained from
    [`channel_exclusion`][toqito.channel_opt.channel_exclusion.channel_exclusion] with uniform
    prior weights; the set is antidistinguishable exactly when that value is (numerically) zero. The
    less computationally intensive dual formulation is used, matching
    [`is_antidistinguishable`][toqito.state_props.is_antidistinguishable.is_antidistinguishable].

    Args:
        channels: A list of channels, each given either as a Choi matrix or as a list of Kraus
            operators. All channels must share the same input and output dimensions.
        solver: Optimization solver passed to `picos`. Default is `"cvxopt"`.
        atol: Absolute tolerance for deciding whether the optimal exclusion value is zero. The
            channels are reported antidistinguishable when the value is within `atol` of zero.
            Default is `1e-6`.
        kwargs: Additional keyword arguments forwarded to the `picos` solve method.

    Returns:
        `True` if the channels are antidistinguishable; `False` otherwise.

    Examples:
        The Choi states of the four Pauli channels are precisely the four Bell states, which are
        antidistinguishable. Hence the Pauli channels are antidistinguishable:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_props import is_channel_antidistinguishable
        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        pauli_channels = [[np.eye(2)], [pauli_x], [pauli_y], [pauli_z]]
        print(is_channel_antidistinguishable(pauli_channels))
        ```

        Two identical channels can never be antidistinguishable, since the true channel can never be
        ruled out:

        ```python exec="1" source="above" result="text"
        from toqito.channels import depolarizing
        from toqito.channel_props import is_channel_antidistinguishable
        print(is_channel_antidistinguishable([depolarizing(2, 0.3), depolarizing(2, 0.3)]))
        ```

    """
    opt_val, _ = channel_exclusion(
        channels,
        probs=[1] * len(channels),
        primal_dual="dual",
        solver=solver,
        **kwargs,
    )
    return bool(np.isclose(opt_val, 0, atol=atol))
