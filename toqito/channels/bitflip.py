"""Implements the bitflip quantum gate channel."""

import numpy as np


def bitflip(
    input_mat: np.ndarray | None = None,
    prob: float = 0,
) -> np.ndarray:
    r"""Apply the bitflip quantum channel to a state or return the Kraus operators.

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

    We can generate the Kraus operators for the bitflip channel with probability 0.3:

    .. jupyter-execute::

     from toqito.channels import bitflip

     bitflip(prob=0.3)


    We can also apply the bitflip channel to a quantum state. For the state :math:`|0\rangle`,
    the bitflip channel with probability 0.3 produces:

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import bitflip

     rho = np.array([[1, 0], [0, 0]])  # |0><0|
     bitflip(rho, prob=0.3)

    References
    ==========
    .. footbibliography::


    :param input_mat: A matrix or state to apply the channel to. If `None`, returns the Kraus operators.
    :param prob: The probability of a bitflip occurring.
    :return: Either the Kraus operators of the bitflip channel if `input_mat` is `None`,
             or the result of applying the channel to `input_mat`.

    """
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    k0 = np.sqrt(1 - prob) * np.eye(2)
    k1 = np.sqrt(prob) * np.array([[0, 1], [1, 0]])

    if input_mat is not None and input_mat.shape != (2, 2):
        raise ValueError("Input matrix must be 2x2 for the bitflip channel.")
    elif input_mat is None:
        return [k0, k1]

    input_mat = np.asarray(input_mat, dtype=complex)

    return k0 @ input_mat @ k0.conj().T + k1 @ input_mat @ k1.conj().T
