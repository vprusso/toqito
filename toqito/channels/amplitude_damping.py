"""Generates the (generalized) amplitude damping channel."""

import numpy as np


def amplitude_damping(
    input_mat: np.ndarray | None = None,
    gamma: float = 0,
    prob: float = 1,
) -> np.ndarray:
    r"""Apply the generalized amplitude damping channel to a quantum state.

    The generalized amplitude damping channel is a quantum channel that models energy dissipation
    in a quantum system, where the system can lose energy to its environment with a certain
    probability. This channel is defined by two parameters: `gamma` (the damping rate) and `prob`
    (the probability of energy loss).

    To also include standard implementation of amplitude damping, we have set `prob = 1` as the default implementation.

    .. note::
          This channel is defined for qubit systems in the standard literature :footcite:`Khatri_2020_Information`.


    The Kraus operators for the generalized amplitude damping channel are given by:

    .. math::
        K_0 = \sqrt{p} \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \\
        K_1 = \sqrt{p}  \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}, \\
        K_2 = \sqrt{1 - p} \begin{pmatrix} \sqrt{1 - \gamma} & 0 \\ 0 & 1 \end{pmatrix}, \\
        K_3 = \sqrt{1 - p}  \begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}, \\

    These operators describe the evolution of a quantum state under the generalized amplitude
    damping process.

    Examples
    ==========

    Apply the generalized amplitude damping channel to a qubit state:

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import amplitude_damping

     rho = np.array([[1, 0], [0, 0]])  # |0><0|
     result = amplitude_damping(rho, gamma=0.1, prob=0.5)

     print(result)

    References
    ==========
    .. footbibliography::


    :param input_mat: The input matrix to which the channel is applied.
                      If `None`, the function returns the Kraus operators of the channel.
    :param gamma: The damping rate, a float between 0 and 1. Represents the probability of
                  energy dissipation.
    :param prob: The probability of energy loss, a float between 0 and 1.
    :return: The evolved quantum state after applying the generalized amplitude damping channel.
             If `input_mat` is `None`, it returns the list of Kraus operators.

    """
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    if not (0 <= gamma <= 1):
        raise ValueError("Gamma (damping rate) must be between 0 and 1.")

    k0 = np.sqrt(prob) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1 = np.sqrt(prob) * np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
    k2 = np.sqrt(1 - prob) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]])
    k3 = np.sqrt(1 - prob) * np.sqrt(gamma) * np.array([[0, 0], [1, 0]])

    if input_mat is not None and input_mat.shape != (2, 2):
        raise ValueError("Input matrix must be 2x2 for the generalized amplitude damping channel.")
    elif input_mat is None:
        return [k0, k1, k2, k3]

    input_mat = np.asarray(input_mat, dtype=complex)

    return (
        k0 @ input_mat @ k0.conj().T
        + k1 @ input_mat @ k1.conj().T
        + k2 @ input_mat @ k2.conj().T
        + k3 @ input_mat @ k3.conj().T
    )
