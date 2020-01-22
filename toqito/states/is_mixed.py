import numpy as np
from toqito.states.is_pure import is_pure


def is_mixed(state: np.ndarray) -> bool:
    return not is_pure(state)
