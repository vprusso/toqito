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

    The measurement can be provided as a single operator (POVM element or Kraus operator) or as a
    list of operators (assumed to be Kraus operators) describing a complete quantum measurement.

    When a single operator is provided:

      - Returns the measurement outcome probability if ``state_update`` is False.
      - Returns a tuple (probability, post_state) if ``state_update`` is True.

    When a list of operators is provided, the function verifies that they satisfy the completeness relation when
    ``state_update`` is True.

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

    ```python exec="1" source="above"
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

    ```python exec="1" source="above"
    import numpy as np
    from toqito.measurement_ops.measure import measure
    
    rho = np.array([[0.5, 0.5], [0.5, 0.5]])
    K0 = np.array([[1, 0], [0, 0]])
    K1 = np.array([[0, 0], [0, 1]])
    
    # Returns list of probabilities.
    print(measure(rho, [K0, K1]))
    
    # Returns list of (probability, post_state) tuples.
    print(measure(rho, [K0, K1], state_update=True))
    ```


    Raises:
        ValueError: If a list of operators does not satisfy the completeness relation.

    Args:
        state: Quantum state as a density matrix shape (d, d) where d is the dimension of the Hilbert space.
        measurement: Either a single measurement operator (an np.ndarray) or a list/tuple of operators. When providing a list, they are assumed to be Kraus operators satisfying the completeness relation.
        tol: Tolerance for numerical precision (default is 1e-10).
        state_update: If True, also return the post-measurement state(s); otherwise, only the probability or probabilities are returned.

    Returns:
        If a single operator is provided, returns a float (probability) or a tuple (probability, post_state) if ``state_update`` is True. If a list is provided, returns a list of probabilities or a list of tuples if ``state_update`` is True.

    """
    if not is_density(state):
        raise ValueError("Input must be a valid density matrix.")

    # Single-operator case.
    if not isinstance(measurement, (list, tuple)):
        result = measurement @ state @ measurement.conj().T
        prob = np.trace(result).real
        if prob > tol:
            post_state = result / prob
        else:
            post_state = np.zeros_like(state)
        return (prob, post_state) if state_update else prob

    # List-of-operators case.
    outcomes: list[float | tuple[float, np.ndarray]] = []
    probs: list[float] = []

    for op in measurement:
        result = op @ state @ op.conj().T
        prob = np.trace(result).real
        probs.append(prob)

        if prob > tol:
            post_state = result / prob
        else:
            post_state = np.zeros_like(state)

        outcomes.append((prob, post_state) if state_update else prob)

    # Only enforce completeness if we're doing the update AND every outcome was nonzero.
    if state_update and all(p > tol for p in probs):
        d = state.shape[0]
        completeness = sum(op.T.conj() @ op for op in measurement)
        if not np.allclose(completeness, np.eye(d), atol=tol):
            raise ValueError("Kraus operators do not satisfy completeness relation: ∑ Kᵢ†Kᵢ ≠ I.")

    return outcomes
