"""Generates the Generalised Amplitude Damping Channel."""

import numpy as np


def GeneralisedAmplitudeDamping(
    input_mat: np.ndarray | None = None,
    gamma: float = 0,
    prob: float = 0,
) -> np.ndarray:
    r"""Apply the generalized amplitude damping channel to a quantum state.

    The generalized amplitude damping channel is a quantum channel that models energy dissipation
    in a quantum system, where the system can lose energy to its environment with a certain
    probability. This channel is defined by two parameters: `gamma` (the damping rate) and `prob`
    (the probability of energy loss).

    The Kraus operators for the generalized amplitude damping channel are given by:

    .. math::
        K_0 = \sqrt{p} \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix},
        K_1 = \sqrt{p} \sqrt{\gamma} \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix},
        K_2 = \sqrt{1 - p} \begin{pmatrix} \sqrt{1 - \gamma} & 0 \\ 0 & 1 \end{pmatrix},
        K_3 = \sqrt{1 - p} \sqrt{\gamma} \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}.

    These operators describe the evolution of a quantum state under the generalized amplitude
    damping process.

    Examples
    ==========

    Apply the generalized amplitude damping channel to a qubit state:

    >>> import numpy as np
    >>> from toqito.channels import GeneralisedAmplitudeDamping
    >>> rho = np.array([[1, 0], [0, 0]])  # |0><0|
    >>> gamma = 0.1  # Damping rate
    >>> prob = 0.5  # Probability of energy loss
    >>> result = GeneralisedAmplitudeDamping(rho, gamma, prob)
    >>> print(result)
    [[0.95 0.  ]
     [0.   0.05]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param input_mat: The input quantum state (density matrix) to which the channel is applied.
                      If `None`, the function returns the Kraus operators of the channel.
    :param gamma: The damping rate, a float between 0 and 1. Represents the probability of
                  energy dissipation.
    :param prob: The probability of energy loss, a float between 0 and 1. Represents the
                 likelihood of the system transitioning to a lower energy state.
    :raises ValueError: If `gamma` or `prob` are not in the range [0, 1], or if `input_mat`
                        is not a valid 2x2 density matrix.
    :return: The evolved quantum state after applying the generalized amplitude damping channel.
             If `input_mat` is `None`, returns the list of Kraus operators.

    """
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    if not (0 <= gamma <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    if input_mat is not None:
        if input_mat.shape[0] != 2:
            raise ValueError("Generalised Amplitude Damping Channel is only defined for qubits (dim=2).")

    dim = 2

    k0 = np.sqrt(prob) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1 = np.sqrt(prob) * np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
    k2 = np.sqrt(1 - prob) * np.array([np.sqrt(1 - gamma), 0][0, 1])
    k3 = np.sqrt(1 - prob) * np.sqrt(gamma) * np.array([[0, 0], [1, 0]])

    kraus_ops = [k0, k1, k2, k3]

    if input_mat is None:
        return kraus_ops

    input_mat = np.asarray(input_mat, dtype=complex)

    result = np.zeros_like(input_mat, dtype=complex)

    for op in kraus_ops:
        result += op @ input_mat @ op.conj().T
    return result
