"""Module for checking whether a quantum channel is extremal."""

import numpy as np
from numpy.linalg import matrix_rank

from toqito.channel_ops.choi_to_kraus import choi_to_kraus


def _convert_to_kraus(channel):
    r"""Convert a quantum channel into its Kraus representation.

    The input channel can be provided as:
      - A list of Kraus operators (each a numpy.ndarray),
      - A Choi matrix (numpy.ndarray),
      - A dictionary with a 'kraus' or 'choi' key,
      - An object with a callable kraus() method.

    Parameters
    ----------
    channel : list, numpy.ndarray, or dict
        The quantum channel representation.

    Returns
    -------
    list
        A list of numpy.ndarray objects representing the Kraus operators.

    Raises
    ------
    ValueError
        If the channel input type is not supported.

    """
    if isinstance(channel, list):
        return channel
    if isinstance(channel, np.ndarray):
        return choi_to_kraus(channel)
    if isinstance(channel, dict):
        if "kraus" in channel:
            return channel["kraus"]
        if "choi" in channel:
            return choi_to_kraus(channel["choi"])
        raise ValueError("Dictionary channel representation must have either 'kraus' or 'choi' key.")
    if hasattr(channel, "kraus") and callable(channel.kraus):
        return channel.kraus()
    raise ValueError(
        "Unsupported channel input type. Please provide a list of Kraus operators, "
        "a Choi matrix, or a dict with a 'kraus' or 'choi' key."
    )


def is_extremal(channel):
    r"""Check whether a quantum channel is extremal.

    According to Section 2.2.4 of Watrous's *Theory of Quantum Information*,
    a channel with Kraus operators \(\{A_i\}_{i=1}^{r}\) is extremal if and only if
    the set \(\{A_i^\dagger A_j \,:\, i,j=1,\dots,r\}\) is linearly independent.

    Parameters
    ----------
    channel : list, numpy.ndarray, or dict
        The quantum channel representation, which can be:
        - A list of Kraus operators
        - A Choi matrix
        - A dictionary with 'kraus' or 'choi' key
        - An object with a callable kraus() method

    Returns
    -------
    bool
        True if the channel is extremal; False otherwise.

    Raises
    ------
    ValueError
        If no Kraus operators can be extracted.

    """
    kraus_ops = _convert_to_kraus(channel)

    if not kraus_ops or len(kraus_ops) == 0:
        raise ValueError("The channel must contain at least one Kraus operator.")

    if isinstance(kraus_ops[0], list):
        kraus_ops = [op[0] for op in kraus_ops]

    r = len(kraus_ops)

    if r == 1:
        return True

    flattened_products = [np.dot(A.conj().T, B).flatten() for A in kraus_ops for B in kraus_ops]

    M = np.column_stack(flattened_products)

    return matrix_rank(M) == r * r
