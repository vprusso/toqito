"""Computes the common quantum overlap quantum states."""

import numpy as np

from toqito.state_opt.state_exclusion import state_exclusion


def common_quantum_overlap(states: list[np.ndarray]) -> float:
    r"""Calculate the common quantum overlap of a collection of quantum states.

    For more information, see :cite:`Campos_2024_Epistemic`.

    The common quantum overlap :math:`\omega_Q[n]` quantifies the "overlap" between :math:`n` quantum states
    based on their antidistinguishability properties. It is related to the
    antidistinguishability probability :math:`A_Q[n]` by the formula:

    .. math::
        \omega_Q[n] = n(1 - A_Q[n])

    For two pure states with inner product :math:`|\langle\psi|\phi\rangle| = p`, the common quantum overlap is:

    .. math::
        \omega_Q = 1 - \sqrt{1 - p^2}

    The common quantum overlap is a key concept in analyzing epistemic models of quantum
    mechanics and understanding quantum state preparation contextuality.

    Examples
    ==========
    Consider the Bell states:

    >>> from toqito.states import bell
    >>> from toqito.state_props import common_quantum_overlap
    >>> bell_states = [bell(0), bell(1), bell(2), bell(3)]
    >>> round(common_quantum_overlap(bell_states),4)
    0.0

    For maximally mixed states in any dimension:

    >>> import numpy as np
    >>> from toqito.state_props import common_quantum_overlap
    >>> dim = 2
    >>> states = [np.eye(dim) / dim, np.eye(dim) / dim, np.eye(dim) / dim]
    >>> round(common_quantum_overlap(states),4)
    1.0

    The common quantum overlap :math:`\omega_Q` for two pure states
    with inner product :math:`|\langle \psi | \phi \rangle| = \cos(\theta)` is given by:

    .. math::
        \omega_Q = 1 - \sqrt{1 - \cos(\theta)^2}

    where :math:`\theta` represents the angle between the two states in Hilbert space.
    For two pure states with a known inner product:

    >>> import numpy as np
    >>> from toqito.state_props import common_quantum_overlap
    >>> theta = np.pi/4
    >>> states = [np.array([1, 0]), np.array([np.cos(theta), np.sin(theta)])]
    >>> round(common_quantum_overlap(states),4)  # Should approximate (1-sqrt(1-cos²(π/4)))
    0.2929

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param states: A list of quantum states represented as numpy arrays. States can be
                  pure states (represented as state vectors) or mixed states
                  (represented as density matrices).
    :return: The common quantum overlap value.

    """
    n = len(states)
    opt_val, _ = state_exclusion(vectors=states, probs=[1] * n, primal_dual="dual")
    return n * (1 - (1 - opt_val / n))

