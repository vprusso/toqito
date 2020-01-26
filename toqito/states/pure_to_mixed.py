import numpy as np


def pure_to_mixed(phi: np.ndarray) -> np.ndarray:
    """
    Converts a state vector or density matrix representation of a state to a
    density matrix.

    :param phi: A density matrix or a pure state vector.
    :return: density matrix representation of PHI, regardless of
             whether PHI is itself already a density matrix or if
             if is a pure state vector.
    """

    # Compute the size of PHI. If it's already a mixed state, leave it alone.
    # If it's a vector (pure state), make it into a density matrix.
    m, n = phi.shape[0], phi.shape[1]

    # It's a pure state vector.
    if min(m, n) == 1:
        return phi * phi.conj().T
    # It's a density matrix.
    elif m == n:
        return phi
    # It's neither.
    else:
        msg = """
            InvalidDim: PHI must be either a vector or square matrix.
        """
        raise ValueError(msg)


