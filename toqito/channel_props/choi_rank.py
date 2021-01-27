import numpy as np
from toqito.channel_ops import kraus_to_choi


def choi_rank(phi: Union[np.ndarray, List[List[np.ndarray]]]) -> int:
    """Calculates the Choi rank from the Choi matrix of a channel or a list
        of Kraus operators defining the channel."""
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    return(np.linalg.matrix_rank(phi))
