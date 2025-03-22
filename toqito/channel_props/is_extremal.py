"""Determines whether a quantum channel is extremal."""

import numpy as np
from numpy.linalg import matrix_rank

from toqito.channel_ops.choi_to_kraus import choi_to_kraus


def is_extremal(channel):
    r"""Determine whether a quantum channel is extremal.

    A channel with Kraus operators :math:`\{A_i\}_{i=1}^{r}` is **extremal** if and only if
    the set :math:`\{A_i^\dagger A_j : i,j=1,\dots,r\}` is linearly independent.

    For more details, see Section 2.2.4 of :cite:`Watrous_2018_TQI`.

    :param channel: The quantum channel representation, which can be:
        - A list of Kraus operators,
        - A Choi matrix,
        - A dictionary with ``'kraus'`` or ``'choi'`` key,
        - An object with a callable ``kraus()`` method.
    :type channel: list[numpy.ndarray] | numpy.ndarray | dict
    :raises ValueError: If no Kraus operators can be extracted.
    :return: ``True`` if the channel is extremal; ``False`` otherwise.
    :rtype: bool
    """
    # Convert the input into a list of Kraus operators
    if isinstance(channel, list):
        kraus_ops = channel
    elif isinstance(channel, np.ndarray):
        kraus_ops = choi_to_kraus(channel)
    elif isinstance(channel, dict):
        if "kraus" in channel:
            kraus_ops = channel["kraus"]
        elif "choi" in channel:
            kraus_ops = choi_to_kraus(channel["choi"])
        else:
            raise ValueError("Dictionary must have a 'kraus' or 'choi' key.")
    elif hasattr(channel, "kraus") and callable(channel.kraus):
        kraus_ops = channel.kraus()
    else:
        raise ValueError("Unsupported channel format. Provide Kraus operators, a Choi matrix, or a dictionary.")

    if not kraus_ops or len(kraus_ops) == 0:
        raise ValueError("The channel must contain at least one Kraus operator.")

    # If nested lists (non-completely positive maps), extract the first set of Kraus operators
    if isinstance(kraus_ops[0], list):
        kraus_ops = [op[0] for op in kraus_ops]

    r = len(kraus_ops)

    # A single Kraus operator (e.g., unitary channel) is always extremal.
    if r == 1:
        return True

    # Compute the set {A_i^† A_j}
    flattened_products = [np.dot(A.conj().T, B).flatten() for A in kraus_ops for B in kraus_ops]

    # Form a matrix whose columns are these vectorized operators.
    M = np.column_stack(flattened_products)

    # The channel is extremal if and only if the operators {A_i^† A_j} are linearly independent.
    return matrix_rank(M) == r * r
