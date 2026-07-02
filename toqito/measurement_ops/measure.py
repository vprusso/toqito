"""Apply measurement to a quantum state."""

import numpy as np

from toqito.matrix_props import is_density


def measure(
    state: np.ndarray,
    measurement: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    tol: float = 1e-10,
    state_update: bool = False,
) -> float | tuple[float, np.ndarray] | list[float | tuple[float, np.ndarray]]:
    r"""Apply measurement to a quantum state.

    The measurement can be provided as a single operator or as a list of operators describing a complete quantum
    measurement.

    A single operator is always treated as a Kraus operator \(K\): the outcome probability is
    \(\mathrm{Tr}(K \rho K^\dagger) = \mathrm{Tr}(K^\dagger K \rho)\) and the post-measurement state is
    \(K \rho K^\dagger / \mathrm{Tr}(K \rho K^\dagger)\). For a POVM element \(E\) the Born probability is
    \(\mathrm{Tr}(E \rho)\); this equals \(\mathrm{Tr}(K \rho K^\dagger)\) only when the operator is a projector
    (\(E^\dagger E = E\)), so to measure with a non-projector POVM element pass its Kraus operators in the list form.

    When a single operator is provided:

      - Returns the measurement outcome probability if ``state_update`` is False.
      - Returns a tuple (probability, post_state) if ``state_update`` is True.

    When a list of operators is provided, the function verifies that they satisfy the completeness relation (which
    depends only on the operators, not on the state or ``state_update``).

    \[
        \sum_i K_i^\dagger K_i = \mathbb{I},
    \]

    when ``state_update`` is True. Then, for each operator \(K_i\), the outcome probability is computed as

    \[
        p_i = \mathrm{Tr}\Bigl(K_i^\dagger K_i\, \rho\Bigr),
    \]

    and, if \(p_i > tol\), the post‐measurement state is updated via

    \[
        u = \frac{1}{\sqrt{3}} e_0 + \sqrt{\frac{2}{3}} e_1
    \]

    where we define \(u u^* = \rho \in \text{D}(\mathcal{X})\).

    Define measurement operators

    \[
        P_0 = e_0 e_0^* \quad \text{and} \quad P_1 = e_1 e_1^*.
    \]

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.states import basis
        from toqito.measurement_ops import measure

        e_0, e_1 = basis(2, 0), basis(2, 1)

        u = 1/np.sqrt(3) * e_0 + np.sqrt(2/3) * e_1
        rho = u @ u.conj().T

        proj_0 = e_0 @ e_0.conj().T
        proj_1 = e_1 @ e_1.conj().T
        print(measure(proj_0, rho))
        ```

    Then the probability of obtaining outcome \(0\) is given by

    \[
        \langle P_0, \rho \rangle = \frac{1}{3}.
    \]


    Similarly, the probability of obtaining outcome \(1\) is given by

    \[
        \langle P_1, \rho \rangle = \frac{2}{3}.
    \]

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.measurement_ops.measure import measure

        rho = np.array([[0.5, 0.5], [0.5, 0.5]])
        K0 = np.array([[1, 0], [0, 0]])
        K1 = np.array([[0, 0], [0, 1]])

        # Returns list of probabilities.
        print("```")
        print(measure(rho, [K0, K1]))

        # Returns list of (probability, post_state) tuples.
        print(measure(rho, [K0, K1], state_update=True))
        print("```")
        ```


    Raises:
        ValueError: If a list of operators does not satisfy the completeness relation.

    Args:
        state: Quantum state as a density matrix shape (d, d) where d is the dimension of the Hilbert space.
        measurement: Either a single measurement operator (an np.ndarray) or a list/tuple of operators. When providing a
            list, they are assumed to be Kraus operators satisfying the completeness relation.
        tol: Tolerance for numerical precision (default is 1e-10).
        state_update: If True, also return the post-measurement state(s); otherwise, only the probability or
            probabilities are returned.

    Returns:
        If a single operator is provided, returns a float (probability) or a tuple (probability, post_state) if
        ``state_update`` is True. If a list is provided, returns a list of probabilities or a list of tuples if
        ``state_update`` is True.

    """
    if not is_density(state):
        raise ValueError("Input must be a valid density matrix.")

    # Single-operator case.
    if not isinstance(measurement, (list, tuple)):
        if not state_update:
            # Only the probability Tr(M rho M^dagger) = Tr(M^dagger M rho) is needed.
            return np.trace(measurement.conj().T @ measurement @ state).real
        result = measurement @ state @ measurement.conj().T
        prob = np.trace(result).real
        if prob > tol:
            post_state = result / prob
        else:
            post_state = np.zeros_like(state)
        return prob, post_state

    # List-of-operators case.
    if len(measurement) == 0:
        raise ValueError("At least one measurement operator is required.")

    # The completeness relation depends only on the operators, not on the state or the resulting probabilities, so
    # validate it unconditionally whenever a list/tuple of Kraus operators is provided.
    d = state.shape[0]
    completeness = sum(op.conj().T @ op for op in measurement)
    if not np.allclose(completeness, np.eye(d), atol=tol):
        raise ValueError("Kraus operators do not satisfy completeness relation: ∑ Kᵢ†Kᵢ ≠ I.")

    outcomes: list[float | tuple[float, np.ndarray]] = []
    for op in measurement:
        result = op @ state @ op.conj().T
        prob = np.trace(result).real

        if prob > tol:
            post_state = result / prob
        else:
            post_state = np.zeros_like(state)

        outcomes.append((prob, post_state) if state_update else prob)

    return outcomes
