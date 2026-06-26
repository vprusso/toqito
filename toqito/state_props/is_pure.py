"""Checks if a quantum state is a pure state."""

import numpy as np


def is_pure(state: list[np.ndarray] | np.ndarray, tol: float | None = None) -> bool:
    r"""Determine if a given state is pure or list of states are pure [@wikipediapurestate].

    A state is said to be pure if it is a density matrix with rank equal to 1. Equivalently, the
    state \(\rho\) is pure if there exists a unit vector \(u\) such that:

    \[
        \rho = u u^*.
    \]

    Args:
        state: The density matrix representing the quantum state or a list of density matrices representing quantum
            states.
        tol: Tolerance used when computing the matrix rank. If `None` (default), numpy's relative tolerance based on
            the largest singular value is used.

    Returns:
        `True` if state is pure and `False` otherwise.

    Examples:
        Consider the following Bell state:

        \[
            u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.
        \]

        The corresponding density matrix of \(u\) may be calculated by:

        \[
            \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                             1 & 0 & 0 & 1 \\
                             0 & 0 & 0 & 0 \\
                             0 & 0 & 0 & 0 \\
                             1 & 0 & 0 & 1
                           \end{pmatrix} \in \text{D}(\mathcal{X}).
        \]

        Calculating the rank of \(\rho\) yields that the \(\rho\) is a pure state. This can be
        confirmed in `|toqito⟩` as follows:

        ```python exec="1" source="above" result="text"
        from toqito.states import bell
        from toqito.state_props import is_pure
        u = bell(0)
        rho = u @ u.conj().T
        print(is_pure(rho))
        ```

        It is also possible to determine whether a set of density matrices are pure. For instance, we
        can see that the density matrices corresponding to the four Bell states yield a result of
        `True` indicating that all states provided to the function are pure.

        ```python exec="1" source="above" result="text"
        from toqito.states import bell
        from toqito.state_props import is_pure
        u0, u1, u2, u3 = bell(0), bell(1), bell(2), bell(3)
        rho0 = u0 @ u0.conj().T
        rho1 = u1 @ u1.conj().T
        rho2 = u2 @ u2.conj().T
        rho3 = u3 @ u3.conj().T
        print(is_pure([rho0, rho1, rho2, rho3]))
        ```

    """
    # A state is pure if and only if its density matrix has rank one. Check each provided state; allow the user to
    # enter either a single state or a list of states.
    states = state if isinstance(state, list) else [state]
    return all(np.linalg.matrix_rank(rho, tol=tol) == 1 for rho in states)
