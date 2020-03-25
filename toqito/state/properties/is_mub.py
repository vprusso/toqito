"""Check if list of vectors constitute a mutually unbiased basis."""
from typing import Any, List, Union
import numpy as np


def is_mub(vec_list: List[Union[np.ndarray, List[Union[float, Any]]]]) -> bool:
    """
    Check if list of vectors constitute a mutually unbiased basis.

    :param vec_list: The list of vectors to check.
    :return: True if `vec_list` constitutes a mutually unbiased basis, and
             False otherwise.
    """
    if len(vec_list) <= 1:
        raise ValueError("There must be at least two bases provided as input.")

    dim = vec_list[0][0].shape[0]
    for i, _ in enumerate(vec_list):
        for j, _ in enumerate(vec_list):
            for k in range(dim):
                if i != j:
                    if not np.isclose(
                        np.abs(
                            np.inner(
                                vec_list[i][k].conj().T[0], vec_list[j][k].conj().T[0]
                            )
                        )
                        ** 2,
                        1 / dim,
                    ):
                        return False
    return True
