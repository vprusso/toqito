import numpy as np
from toqito.helper.constants import e0, e1


def bell(idx: int) -> np.ndarray:
    """
    Produces a Bell state.
    :param idx: A parameter in [0, 1, 2, 3]

    Returns one of the following four Bell states depending on the value
    of IDX:
        0: (|0>|0> + |1>|1>)/sqrt(2)
        1: (|0>|0> - |1>|1>)/sqrt(2)
        2: (|0>|1> + |1>|0>)/sqrt(2) 
        3: (|0>|1> - |1>|0>)/sqrt(2)
    """
    if idx == 0:
        return 1/np.sqrt(2) * (np.kron(e0, e0) + np.kron(e1, e1))
    elif idx == 1:
        return 1/np.sqrt(2) * (np.kron(e0, e0) - np.kron(e1, e1))
    elif idx == 2:
        return 1/np.sqrt(2) * (np.kron(e0, e1) + np.kron(e1, e0))
    elif idx == 3:
        return 1/np.sqrt(2) * (np.kron(e0, e1) - np.kron(e1, e0))
    else:
        raise ValueError("Invalid integer value for Bell state.")
