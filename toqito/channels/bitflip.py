"""Implements the bitflip quantum gate channel."""

import numpy as np

from toqito.channel_ops.apply_channel import apply_channel as apply_op


def bitflip(
    input_mat: np.ndarray | None = None,
    prob: float = 0,
    apply_channel: bool = False,
) -> np.ndarray | list[np.ndarray]:
    r"""Generate the bitflip quantum channel or apply it to a state.

    The *bitflip channel* is a quantum channel that flips a qubit from :math:`|0\rangle` to :math:`|1\rangle`
    and from :math:`|1\rangle` to :math:`|0\rangle` with probability :math:`p`.
    It is defined by the following operation:

    .. math::

        \mathcal{E}(\rho) = (1-p) \rho + p X \rho X

    where :math:`X` is the Pauli-X (NOT) gate given by:

    .. math::

        X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}

    The Kraus operators for this channel are:

    .. math::

        K_0 = \sqrt{1-p} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
        K_1 = \sqrt{p} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}

    Examples
    ==========

    Generate the Kraus operators for the bitflip channel with probability 0.3:

    .. jupyter-execute::

     from toqito.channels import bitflip

     kraus_ops = bitflip(prob=0.3)
     print(kraus_ops)


    Apply the bitflip channel to a quantum state. For the state :math:`|0\rangle`,
    the bitflip channel with probability 0.3 produces:

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import bitflip

     rho = np.array([[1, 0], [0, 0]])  # |0><0|
     result = bitflip(rho, prob=0.3, apply_channel=True)
     print(result)

    References
    ==========
    .. footbibliography::


    :param input_mat: A matrix or state to apply the channel to (optional).
                      If provided with `apply_channel=True`, the channel is applied to this matrix.
    :param prob: The probability of a bitflip occurring.
    :param apply_channel: If True, apply the channel to `input_mat` using the Kraus operators.
                          If False (default), return the Kraus operators as a list.
                          When `apply_channel=True`, `input_mat` must be provided.
    :return: If `apply_channel=False` (default), returns the list of Kraus operators
             :code:`[K0, K1]`. If `apply_channel=True` and `input_mat` is provided,
             returns the result of applying the channel to `input_mat`.

    """
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    k0 = np.sqrt(1 - prob) * np.eye(2)
    k1 = np.sqrt(prob) * np.array([[0, 1], [1, 0]])
    kraus_ops = [k0, k1]

    # If apply_channel is True, we need input_mat
    if apply_channel:
        if input_mat is None:
            raise ValueError("input_mat is required when apply_channel=True.")
        if input_mat.shape != (2, 2):
            raise ValueError("Input matrix must be 2x2 for the bitflip channel.")
        return apply_op(input_mat, kraus_ops)

    # apply_channel=False: return Kraus operators (regardless of input_mat)
    return kraus_ops