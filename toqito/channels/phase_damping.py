"""phase damping channel."""

import numpy as np


def phase_damping(
    input_mat: np.ndarray | None = None,
    gamma: float = 0,
) -> np.ndarray | list[np.ndarray]:
    r"""Apply the phase damping channel to a quantum state [@Chuang_2011_Quantum].

    The phase damping channel describes how quantum information is lost due to environmental interactions,
    causing dephasing in the computational basis without losing energy.

    The Kraus operators for the phase damping channel are:

    \[
        K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \\
        K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix},
    \]

    Examples:
    Applying the phase damping channel to a qubit state:

    ```python exec="1" source="above"
    import numpy as np
    from toqito.channels.phase_damping import phase_damping
    
    rho = np.array([[1, 0.5], [0.5, 1]])
    result = phase_damping(rho, gamma=0.2)
    
    print(result)
    ```

    Args:
        input_mat: The input matrix to apply the channel to. If `None`, the function returns the Kraus operators.
        gamma: The dephasing rate (between 0 and 1), representing the probability of phase decoherence.

    Returns:
        The transformed quantum state after applying the phase damping channel. If `input_mat` is `None`, returns the list of Kraus operators.

    """
    if not (0 <= gamma <= 1):
        raise ValueError("Gamma must be between 0 and 1.")

    k0 = np.diag([1, np.sqrt(1 - gamma)])
    k1 = np.diag([0, np.sqrt(gamma)])

    if input_mat is not None and input_mat.shape != (2, 2):
        raise ValueError("Input matrix must be 2x2 for the phase damping channel.")
    elif input_mat is None:
        return [k0, k1]

    input_mat = np.asarray(input_mat, dtype=complex)

    return k0 @ input_mat @ k0.conj().T + k1 @ input_mat @ k1.conj().T
