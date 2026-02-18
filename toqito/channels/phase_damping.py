"""phase damping channel."""

import numpy as np

from toqito.channel_ops.apply_channel import apply_channel as apply_op


def phase_damping(
    input_mat: np.ndarray | None = None,
    gamma: float = 0,
    apply_channel: bool = False,
) -> np.ndarray | list[np.ndarray]:
    r"""Generate the phase damping channel or apply it to a quantum state :footcite:`Chuang_2011_Quantum`.

    The phase damping channel describes how quantum information is lost due to environmental interactions,
    causing dephasing in the computational basis without losing energy.

    The Kraus operators for the phase damping channel are:

    .. math::
        K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \\
        K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix},

    Examples
    ==========
    Generate the Kraus operators for the phase damping channel:

    .. jupyter-execute::

     from toqito.channels.phase_damping import phase_damping

     kraus_ops = phase_damping(gamma=0.2)
     print(kraus_ops)

    Apply the phase damping channel to a qubit state:

    .. jupyter-execute::

     import numpy as np
     from toqito.channels.phase_damping import phase_damping

     rho = np.array([[1, 0.5], [0.5, 1]])
     result = phase_damping(rho, gamma=0.2, apply_channel=True)

     print(result)

    References
    ==========
    .. footbibliography::


    :param input_mat: The input matrix to apply the channel to (optional).
                      If provided with `apply_channel=True`, the channel is applied to this matrix.
    :param gamma: The dephasing rate (between 0 and 1), representing the probability of phase decoherence.
    :param apply_channel: If True, apply the channel to `input_mat` using the Kraus operators.
                          If False (default), return the Kraus operators as a list.
                          When `apply_channel=True`, `input_mat` must be provided.
    :return: If `apply_channel=False` (default), returns the list of Kraus operators
             :code:`[K0, K1]`. If `apply_channel=True` and `input_mat` is provided,
             returns the result of applying the channel to `input_mat`.

    """
    if not (0 <= gamma <= 1):
        raise ValueError("Gamma must be between 0 and 1.")

    k0 = np.diag([1, np.sqrt(1 - gamma)])
    k1 = np.diag([0, np.sqrt(gamma)])
    kraus_ops = [k0, k1]

    # If apply_channel is True, we need input_mat
    if apply_channel:
        if input_mat is None:
            raise ValueError("input_mat is required when apply_channel=True.")
        if input_mat.shape != (2, 2):
            raise ValueError("Input matrix must be 2x2 for the phase damping channel.")
        return apply_op(input_mat, kraus_ops)

    # apply_channel=False: return Kraus operators (regardless of input_mat)
    return kraus_ops