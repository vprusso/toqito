"""Apply measurement to a quantum state."""

import numpy as np


def measure(
    state: np.ndarray,
    measurement: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    tol: float = 1e-10,
    update: bool = False,
) -> float | tuple[float, np.ndarray] | list[float | tuple[float, np.ndarray]]:
    r"""Apply measurement to a quantum state.

    The measurement can be provided as a single operator (POVM element or Kraus operator) or as a
    list of operators (assumed to be Kraus operators) describing a complete quantum measurement.

    When a single operator is provided:
      - Returns the measurement outcome probability if ``update`` is False.
      - Returns a tuple (probability, post_state) if ``update`` is True.

    When a list of operators is provided, the function verifies that they satisfy the completeness relation in  Eq. :ref:`completeness_rel_equation` when ``update`` is True. 

    .. math::
      :label: completeness_rel_equation
       \sum_i K_i^\dagger K_i = \mathbb{I},

    when `update` is True. Then, for each operator :math:`K_i`, the outcome probability is computed as

    .. math::
       p_i = \mathrm{Tr}\Bigl(K_i^\dagger K_i\, \rho\Bigr),

    and, if :math:`p_i > tol`, the post‐measurement state is updated via

    .. math::
       \rho_i = \frac{K_i\, \rho\, K_i^\dagger}{p_i}.

    If :math:`p_i \le tol`, the corresponding post‐measurement state is a zero matrix.

    Examples
    ========
    **Single operator (POVM element):**

    .. jupyter-execute::

        import numpy as np
        from toqito.measurement_ops.measure import measure
        rho = np.array([[0.5, 0.5], [0.5, 0.5]])
        proj_0 = np.array([[1, 0], [0, 0]])

        # Without update; simply returns the probability.
        print(measure(rho, proj_0))

        # With state update; returns (probability, post_state).
        p, post_state = measure(rho, proj_0, update=True)
        print(p)

    **Multiple operators (Kraus operators):**

    .. jupyter-execute::

        import numpy as np
        from toqito.measurement_ops.measure import measure
        rho = np.array([[0.5, 0.5], [0.5, 0.5]])
        K0 = np.array([[1, 0], [0, 0]])
        K1 = np.array([[0, 0], [0, 1]])

        # Returns list of probabilities.
        print(measure(rho, [K0, K1]))

        # Returns list of (probability, post_state) tuples.
        print(measure(rho, [K0, K1], update=True))

    :param state: Quantum state as a density matrix shape (d, d) where d is......
    :param measurement: Either a single measurement operator (an np.ndarray) or a list/tuple of operators.
                        When providing a list, they are assumed to be Kraus operators satisfying the completeness
                        relation.
    :param tol: Tolerance for numerical precision (default is 1e-10).
    :param update: If True, also return the post-measurement state(s); otherwise, only the probability or
                   probabilities are returned.
    :raises ValueError: If a list of operators does not satisfy the completeness relation.
    :return: If a single operator is provided, returns a float (probability) or a tuple (probability, post_state)
             if ``update`` is True. If a list is provided, returns a list of probabilities or a list of tuples if
             `update` is True.

    """
    # Single-operator case
    if not isinstance(measurement, (list, tuple)):
        result = measurement @ state @ measurement.conj().T
        prob = np.trace(result).real
        if prob > tol:
            post_state = result / prob
        else:
            post_state = np.zeros_like(state)
        return (prob, post_state) if update else prob

    # List-of-operators case
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

        outcomes.append((prob, post_state) if update else prob)

    # Only enforce completeness if we're doing the update AND every outcome was nonzero
    if update and all(p > tol for p in probs):
        d = state.shape[0]
        completeness = sum(op.T.conj() @ op for op in measurement)
        if not np.allclose(completeness, np.eye(d), atol=tol):
            raise ValueError("Kraus operators do not satisfy completeness relation: ∑ Kᵢ†Kᵢ ≠ I.")

    return outcomes
