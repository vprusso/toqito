"""Implements the bitflip quantum gate channel."""

import numpy as np

from toqito.helper import expr_as_np_array, np_array_as_expr


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

    >>> from toqito.channels import bitflip
    >>> bitflip(prob=0.3)
    [array([[0.83666003, 0.        ],
            [0.        , 0.83666003]]),
     array([[0.        , 0.54772256],
            [0.54772256, 0.        ]])]

    We can also apply the bitflip channel to a quantum state. For the state :math:`|0\rangle`,
    the bitflip channel with probability 0.3 produces:

    >>> from toqito.channels import bitflip
    >>> import numpy as np
    >>> rho = np.array([[1, 0], [0, 0]])  # |0><0|
    >>> bitflip(rho, prob=0.3)
    array([[0.7, 0. ],
           [0. , 0.3]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames
    

    :param input_mat: A matrix or state to apply the channel to. If `None`, returns the Kraus operators.
    :param prob: The probability of a bitflip occurring.
    :return: Either the Kraus operators of the bitflip channel if `input_mat` is `None`,
             or the result of applying the channel to `input_mat`.

    """
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    if input_mat is not None:
        if input_mat.shape[0] != 2:
              raise ValueError("Bitflip channel is only defined for qubits (dim=2).")
    
      
    dim = 2

    # Define the Kraus operators for the bitflip channel
    no_flip_prob = np.sqrt(1 - prob)
    flip_prob = np.sqrt(prob)

    # Identity matrix for the no-flip case
    k0 = no_flip_prob * np.eye(dim)

    # X gate for the flip case
    k1 = flip_prob * np.array([[0, 1], [1, 0]])

    kraus_ops = [k0, k1]

    # If no input matrix is provided, return the Kraus operators
    if input_mat is None:
        return kraus_ops

    # Apply the channel to the input state
    input_mat = np.asarray(input_mat, dtype=complex)

    result = np.zeros_like(input_mat, dtype=complex)
    for op in kraus_ops:
        result += op @ input_mat @ op.conj().T

    return result
