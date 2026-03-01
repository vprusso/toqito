"""Computes the common quantum overlap quantum states."""

import numpy as np

from toqito.state_opt.state_exclusion import state_exclusion


def common_quantum_overlap(states: list[np.ndarray]) -> float:
    r"""Calculate the common quantum overlap of a collection of quantum states.

    For more information, see [@campos2024epistemic].

    The common quantum overlap \(\omega_Q[n]\) quantifies the "overlap" between \(n\) quantum states
    based on their antidistinguishability properties. It is related to the
    antidistinguishability probability \(A_Q[n]\) by the formula:

    \[
        \omega_Q[n] = n(1 - A_Q[n])
    \]

    For two pure states with inner product \(|\langle\psi|\phi\rangle| = p\), the common quantum overlap is:

    \[
        \omega_Q = 1 - \sqrt{1 - p^2}
    \]

    The common quantum overlap is a key concept in analyzing epistemic models of quantum
    mechanics and understanding quantum state preparation contextuality.

    Examples:
        Consider the Bell states:

        ```python exec="1" source="above"
        from toqito.states import bell
        from toqito.state_props import common_quantum_overlap
        bell_states = [bell(0), bell(1), bell(2), bell(3)]
        print(common_quantum_overlap(bell_states))
        ```

        For maximally mixed states in any dimension:

        ```python exec="1" source="above"
        import numpy as np
        from toqito.state_props import common_quantum_overlap
        dim = 2
        states = [np.eye(dim) / dim, np.eye(dim) / dim, np.eye(dim) / dim]
        print(common_quantum_overlap(states))
        ```

        The common quantum overlap \(\omega_Q\) for two pure states
        with inner product \(|\langle \psi | \phi \rangle| = \cos(\theta)\) is given by:

        \[
            \omega_Q = 1 - \sqrt{1 - \cos(\theta)^2}
        \]

        where \(\theta\) represents the angle between the two states in Hilbert space.
        For two pure states with a known inner product:

        ```python exec="1" source="above"
        import numpy as np
        from toqito.state_props import common_quantum_overlap
        theta = np.pi/4
        states = [np.array([1, 0]), np.array([np.cos(theta), np.sin(theta)])]
        print(common_quantum_overlap(states)) # Should approximate (1-sqrt(1-cos²(π/4)))
        ```

    Args:
        states: A list of quantum states represented as numpy arrays. States can be pure states
            (represented as state vectors) or mixed states (represented as density matrices).

    Returns:
        The common quantum overlap value.

    """
    n = len(states)
    opt_val, _ = state_exclusion(vectors=states, probs=[1] * n, primal_dual="dual")
    return n * (1 - (1 - opt_val / n))
