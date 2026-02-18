"""Generates the (generalized) amplitude damping channel."""

import numpy as np

from toqito.channel_ops.apply_channel import apply_channel as apply_op


def amplitude_damping(
    input_mat: np.ndarray | None = None,
    gamma: float = 0,
    prob: float = 1,
    apply_channel: bool = False,
) -> np.ndarray | list[np.ndarray]:
    r"""Generate the generalized amplitude damping channel or apply it to a quantum state.

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
        K_3 = \sqrt{1 - p}  \begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}, 

    These operators describe the evolution of a quantum state under the generalized amplitude
    damping process.

    Examples
    ==========

    Generate the Kraus operators for the generalized amplitude damping channel:

    .. jupyter-execute::

     from toqito.channels import amplitude_damping

     kraus_ops = amplitude_damping(gamma=0.1, prob=0.5)
     print(kraus_ops)

    Apply the generalized amplitude damping channel to a qubit state:

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import amplitude_damping

     rho = np.array([[1, 0], [0, 0]])  # |0><0|
     result = amplitude_damping(rho, gamma=0.1, prob=0.5, apply_channel=True)

     print(result)

    References
    ==========
    .. footbibliography::


    :param input_mat: The input matrix to which the channel is applied (optional).
                      If provided with `apply_channel=True`, the channel is applied to this matrix.
    :param gamma: The damping rate, a float between 0 and 1. Represents the probability of
                  energy dissipation.
    :param prob: The probability of energy loss, a float between 0 and 1.
    :param apply_channel: If True, apply the channel to `input_mat` using the Kraus operators.
                          If False (default), return the Kraus operators as a list.
                          When `apply_channel=True`, `input_mat` must be provided.
    :return: If `apply_channel=False` (default), returns the list of Kraus operators
             :code:`[K0, K1, K2, K3]`. If `apply_channel=True` and `input_mat` is provided,
             returns the result of applying the channel to `input_mat`.

    """
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    if not (0 <= gamma <= 1):
        raise ValueError("Gamma (damping rate) must be between 0 and 1.")

    k0 = np.sqrt(prob) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1 = np.sqrt(prob) * np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
    k2 = np.sqrt(1 - prob) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]])
    k3 = np.sqrt(1 - prob) * np.sqrt(gamma) * np.array([[0, 0], [1, 0]])
    kraus_ops = [k0, k1, k2, k3]

    # If apply_channel is True, we need input_mat
    if apply_channel:
        if input_mat is None:
            raise ValueError("input_mat is required when apply_channel=True.")
        if input_mat.shape != (2, 2):
            raise ValueError("Input matrix must be 2x2 for the generalized amplitude damping channel.")
        return apply_op(input_mat, kraus_ops)

    # apply_channel=False: return Kraus operators (regardless of input_mat)
    return kraus_ops