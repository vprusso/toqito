"""Converts Kraus operators into the corressponding quantum channel (i.e. superoperator)."""

import numpy as np

from toqito.matrix_ops import tensor


def kraus_to_channel(
    kraus_list: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Convert a collection of Kraus operators into the corresponding quantum channel (superoperator).

    :param kraus_list: List of tuples (A, B) where A and B are Kraus operators as numpy arrays
    :return: The superoperator as a numpy array
    """
    super_op = sum(tensor(B, A.conj()) for A, B in kraus_list)
    return super_op
