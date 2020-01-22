import numpy as np
from toqito.helper.constants import e0, e1


def bell(idx: int) -> np.ndarray:
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
